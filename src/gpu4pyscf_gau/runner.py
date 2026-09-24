"""Run Gaussian and one resident worker on the current allocated compute node."""
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from .config import executable
from .external_client import request
from .fchk import read_fchk

BOHR = 0.529177210903


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def read_xyz(filename):
    lines = Path(filename).read_text().splitlines()
    n = int(lines[0]); atoms = []
    if n < 1 or len(lines) < n+2 or any(s.strip() for s in lines[n+2:]):
        raise ValueError('Expected one complete XYZ geometry, coordinates in Angstrom')
    for line in lines[2:n+2]:
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r'[A-Z][a-z]?', fields[0]):
            raise ValueError('XYZ lines must be: Element x y z')
        xyz = list(map(float, fields[1:]))
        if not all(map(math.isfinite, xyz)):
            raise ValueError('XYZ contains non-finite coordinates')
        atoms.append((fields[0], *xyz))
    return atoms


def gaussian_status(text, task, returncode):
    normal = returncode == 0 and 'Normal termination of Gaussian' in text
    optimized = bool(re.search(r'^\s*Optimization completed(?: on the basis of negligible forces)?\.', text, re.M))
    fallback = 'FormBX had a problem' in text or 'Switching to Cartesian' in text
    completed = normal and (task not in ('opt','tsopt') or optimized and not fallback)
    return dict(completed=completed, normal_termination=normal, optimization_completed=optimized,
                cartesian_fallback=fallback, returncode=returncode,
                frequencies_cm1=[float(x) for line in re.findall(r'Frequencies --([^\n]+)', text)
                                  for x in line.split()],
                irc_scope='local_path_only; endpoint identity not verified' if task=='irc' else None)


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Worker:
    def __init__(self, directory, cfg, socket_dir):
        directory.mkdir()
        self.directory = directory
        self.socket = str(socket_dir/'worker.sock')
        Path(self.socket).unlink(missing_ok=True)
        save(directory/'config.json', cfg['gpu'])
        env = os.environ.copy()
        env.update({k:str(v) for k,v in cfg['runtime']['worker_environment'].items()})
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[key] = str(cfg['gpu']['threads'])
        env['PYTHONUNBUFFERED'] = '1'
        python = executable(cfg['runtime']['worker_python'] or sys.executable)
        # Include this installed/source package for a separately selected GPU Python.
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
        self.log = (directory/'worker.log').open('w')
        self.process = subprocess.Popen(self.command(python, directory), env=env,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.perf_counter()
        try:
            while not (directory/'worker.ready').exists():
                if self.process.poll() is not None:
                    raise RuntimeError('Worker initialization failed; see worker.log')
                if time.perf_counter()-start > cfg['runtime']['startup_timeout_seconds']:
                    raise TimeoutError('Worker initialization timeout')
                time.sleep(.1)
        except BaseException:
            self.close(); raise
        self.startup_seconds = time.perf_counter()-start

    def command(self, python, directory):
        return [python, '-m', 'gpu4pyscf_gau.worker', '--root', str(directory),
                '--config', str(directory/'config.json'), '--socket', self.socket]

    def reset(self):
        request({'operation':'reset'}, self.socket)

    def close(self):
        # Termination also works if a calculation is stuck inside a GPU request.
        stop_process(self.process)
        self.log.close()


def run_gaussian(directory, job, atoms, cfg, worker, shim_dir):
    directory.mkdir()
    scratch = directory/'scratch'; scratch.mkdir()
    gauss = cfg['gaussian']
    env = os.environ.copy()
    env.update({k:str(v) for k,v in gauss['environment'].items()})
    env.pop('LD_PRELOAD', None)
    env['GPU_GAU_SOCKET'] = worker.socket
    env['GPU_GAU_REQUEST_TIMEOUT'] = str(cfg['runtime']['timeout_seconds'])
    env['GPU_GAU_CLIENT_TRACE'] = str(directory/'client_profile.jsonl')
    env['GAUSS_SCRDIR'] = str(scratch)
    if gauss['exedir']:
        env['GAUSS_EXEDIR'] = gauss['exedir']
    env['PATH'] = str(shim_dir)+os.pathsep+env.get('PATH','')
    route = f'External="gpu_gau_external" {cfg["routes"][job["task"]]} NoSymm'
    text = f'%nprocshared={gauss["threads"]}\n%mem={gauss["memory"]}\n%chk=gaussian.chk\n#p {route}\n\nGPU4PySCF Gaussian External\n\n'
    text += f'{job["charge"]} {job["multiplicity"]}\n'
    text += '\n'.join(f'{s} {x:.12f} {y:.12f} {z:.12f}' for s,x,y,z in atoms)+'\n\n'
    (directory/'input.gjf').write_text(text)
    start = time.perf_counter()
    with (directory/'input.gjf').open('rb') as inp, (directory/'gaussian.log').open('wb') as log:
        process = subprocess.Popen([gauss['executable']], cwd=directory, env=env, stdin=inp,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = process.wait(timeout=cfg['runtime']['timeout_seconds'])
        except subprocess.TimeoutExpired:
            stop_process(process); rc = 124
        except BaseException:
            stop_process(process); raise
    status = gaussian_status((directory/'gaussian.log').read_text(errors='replace'), job['task'], rc)
    status['gaussian_wall_seconds'] = time.perf_counter()-start
    if not status['completed']:
        return status
    # Formatted checkpoint provides the accepted final geometry, not the last trial geometry.
    formchk = gauss.get('formchk')
    if formchk:
        fmt = subprocess.run([formchk,'gaussian.chk','gaussian.fchk'], cwd=directory, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        (directory/'formchk.log').write_text(fmt.stdout)
        if fmt.returncode:
            status.update(completed=False, error='formchk failed after normal Gaussian termination')
            return status
        data = read_fchk(directory/'gaussian.fchk')
        save(directory/'fchk_data.json', data)
        coords = data.get('Current cartesian coordinates', [])
        if len(coords) == 3*len(atoms):
            final = str(len(atoms))+'\nAccepted Gaussian geometry; Angstrom\n'
            final += '\n'.join(f'{atom[0]} '+ ' '.join(f'{x*BOHR:.12f}' for x in coords[3*i:3*i+3])
                               for i,atom in enumerate(atoms))+'\n'
            (directory/'final.xyz').write_text(final)
        elif job['task'] in ('opt','tsopt'):
            status.update(completed=False, error='Accepted final geometry missing from formatted checkpoint')
        status['energy_hartree'] = data.get('Total Energy', data.get('SCF Energy'))
    return status


def run_jobs(cfg, jobs, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    cfg['gaussian']['executable'] = executable(cfg['gaussian']['executable'])
    if cfg['gaussian']['formchk']:
        value = cfg['gaussian']['formchk']
        if value == 'formchk' and cfg['gaussian']['exedir']:
            value = str(Path(cfg['gaussian']['exedir'])/'formchk')
        cfg['gaussian']['formchk'] = executable(value)
    names = set()
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', job['name']) or job['name'] in names:
            raise ValueError('Each job needs a unique name containing only letters, digits, _ or -')
        names.add(job['name'])
        if job['task'] not in cfg['routes'] or job['multiplicity'] < 1:
            raise ValueError('Invalid task/multiplicity')
        job['atoms'] = read_xyz(job['xyz'])
    output.mkdir(parents=True)
    save(output/'config.resolved.json', cfg)
    save(output/'jobs.json', jobs)
    scratch = cfg['runtime']['scratch_dir']
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='gpu-gau-', dir=scratch))
    # Keep Unix socket short even when the user supplies a long scratch/output path.
    sockets = tempfile.TemporaryDirectory(prefix='gg-sock-')
    shim_dir = Path(sockets.name)
    shim = shim_dir/'gpu_gau_external'
    callback = Path(__file__).with_name('external_client.py').resolve()
    shim.write_text('#!/bin/sh\nexec env -u LD_PRELOAD '+shlex.quote(sys.executable)+' -S '+shlex.quote(str(callback))+' "$@"\n')
    shim.chmod(0o700)
    results = []; worker = None; worker_id = 0
    start = time.perf_counter()
    try:
        for job in jobs:
            status = dict(name=job['name'], task=job['task'], completed=False)
            save(output/'progress.json', dict(active=job['name']))
            try:
                if worker is None:
                    worker_id += 1
                    worker = Worker(work/f'worker_{worker_id:03d}', cfg, shim_dir)
                    status['worker_startup_seconds'] = worker.startup_seconds
                else:
                    worker.reset()
                    status['worker_startup_seconds'] = 0
                status['worker'] = worker.directory.name
                events_file = worker.directory/'events.jsonl'
                before = len(events_file.read_text().splitlines()) if events_file.exists() else 0
                status.update(run_gaussian(work/job['name'], job, job['atoms'], cfg, worker, shim_dir))
                events = [json.loads(x) for x in events_file.read_text().splitlines()[before:]] if events_file.exists() else []
                status['successful_evaluations'] = len(events)
                if status['completed'] and not events:
                    status.update(completed=False, error='No successful External callback recorded')
                status['derivative_orders'] = [e['derivative_order'] for e in events]
                for key in ['scf_seconds','gradient_seconds','hessian_seconds']:
                    status[key] = sum(e[key] for e in events)
                status['timing_note'] = 'Component sums cover successful callbacks; failed callbacks may have consumed additional time.'
            except Exception as exc:
                status.update(error=str(exc), completed=False)
                (output/f'{job["name"]}.error.txt').write_text(traceback.format_exc())
            results.append(status)
            if not status['completed'] and worker is not None:
                worker.close(); worker = None
            save(output/'summary.json', dict(completed=all(x['completed'] for x in results), results=results))
    finally:
        if worker is not None:
            worker.close()
        sockets.cleanup()
        # Preserve evidence, excluding Gaussian scratch (checkpoints remain in calculation directories).
        try:
            shutil.copytree(work, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns('scratch'))
        except Exception:
            save(output/'archive_error.json', dict(retained_work_directory=str(work), error=traceback.format_exc()))
            raise
        else:
            shutil.rmtree(work)
        save(output/'progress.json', dict(state='finished' if len(results)==len(jobs) else 'interrupted',
                                          elapsed_seconds=time.perf_counter()-start))
    return all(x['completed'] for x in results)

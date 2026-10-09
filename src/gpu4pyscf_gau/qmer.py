"""Resumable, disjoint HDF5 reaction shards using the existing Gaussian runner."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import threading

from .config import load_config
from .core_dump import disable_core_dumps, cleanup_core_dumps
from .fchk import read_fchk
from .runner import BOHR, save


# Threads only orchestrate isolated Gaussian/worker subprocesses; CUDA is never
# imported in this scheduler. Every reaction owns its state, scratch and socket.
_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()
_CANCEL = threading.Event()
PIPELINE_VERSION = 3
ENDPOINT_IMAGINARY_THRESHOLD = 0.0
CHECKPOINT_NAMES = {
    'tsopt': 'ts_opt',
    'ts_freq': 'ts_freq',
    'irc': 'irc',
    'endpoint_opt_reverse': 'endpoint_reverse_opt',
    'endpoint_opt_forward': 'endpoint_forword_opt',
    'endpoint_freq_reverse': 'endpoint_reverse_freq',
    'endpoint_freq_forward': 'endpoint_forword_freq',
}


def cancel_active():
    from .runner import stop_process
    with _ACTIVE_LOCK:
        _CANCEL.set()
        processes = list(_ACTIVE)
    for process in processes:
        try:
            stop_process(process)
        except ProcessLookupError:
            pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(path, handle):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                dataset_version=str(handle.attrs.get('dataset_version', '')),
                count=len(handle['records/id']))


def prepare(source, output, shards=8, chunk_size=100):
    import h5py
    import numpy as np
    source = Path(source).resolve(); output = Path(output).resolve()
    if shards < 1 or chunk_size < 1: raise ValueError('shards/chunk-size must be positive')
    if output.exists(): raise ValueError('Manifest directory already exists; reuse it or choose a new one')
    output.mkdir(parents=True)
    with h5py.File(source, 'r') as f:
        meta = identity(source, f)
        if not bool(f.attrs.get('complete', False)): raise ValueError('HDF5 export is incomplete')
        if f['TS/coordinates'].attrs.get('unit') != 'angstrom': raise ValueError('Expected TS coordinates in angstrom')
        offsets = f['offsets/atom'][:]
        count = meta['count']
        if len(offsets) != count+1 or offsets[0] != 0 or np.any(np.diff(offsets) < 1):
            raise ValueError('Invalid atom offsets')
        if offsets[-1] != len(f['atoms/atomic_numbers']) or offsets[-1] != len(f['TS/coordinates']):
            raise ValueError('Atom and TS coordinate offsets disagree')
        ids = f['records/id'][:]; nums = f['records/id_num'][:]
        if len(set(ids.tolist())) != count: raise ValueError('Duplicate reaction IDs')
        paths = [output/f'shard_{i:02d}.jsonl' for i in range(shards)]
        streams = [p.open('w') for p in paths]
        counts = [0]*shards
        try:
            for i in range(count):
                shard = i % shards; ordinal = counts[shard]
                row = dict(index=i, id=ids[i].decode(), id_num=nums[i].decode(),
                           natoms=int(offsets[i+1]-offsets[i]), shard=shard,
                           chunk=ordinal//chunk_size)
                if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']): raise ValueError('Unsafe reaction ID')
                streams[shard].write(json.dumps(row)+'\n'); counts[shard]+=1
        finally:
            for stream in streams: stream.close()
        meta.update(shards=shards, chunk_size=chunk_size, counts=counts,
                    assignment='selected.h5 row index modulo shards',
                    charge=int(f.attrs['charge']), multiplicity=int(f.attrs['multiplicity']),
                    manifests=[p.name for p in paths])
        meta['manifest_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        save(output/'dataset.json', meta)
    return meta


def write_xyz(path, numbers, coords, comment):
    import numpy as np
    ELEMENTS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
    array = np.asarray(coords, dtype=float)
    if array.shape != (len(numbers), 3) or not np.isfinite(array).all(): raise ValueError('Invalid geometry')
    if any(int(z)<1 or int(z)>=len(ELEMENTS) for z in numbers): raise ValueError('Invalid atomic number')
    path.write_text(str(len(numbers))+'\n'+comment+'\n'+'\n'.join(
        f'{ELEMENTS[int(z)]} '+ ' '.join(f'{x:.12f}' for x in row) for z,row in zip(numbers,array))+'\n')


def irc_endpoints(fchk, numbers, output):
    """Use accepted fchk IRC geometries, paired with signed reaction coordinates."""
    import numpy as np
    data = read_fchk(fchk); nvar = data.get('IRC Num geometry variables'); nres = data.get('IRC Num results per geometry')
    if nvar != len(numbers)*3 or not isinstance(nres, int) or nres < 2:
        raise ValueError('IRC formatted checkpoint dimensions do not match Cartesian geometry')
    coords=[]; results=[]
    for name in sorted(data):
        if re.match(r'IRC point\s+\d+ Geometries$', name):
            stem=name[:-len('Geometries')]
            result=next((v for k,v in data.items() if k.startswith(stem) and 'Results for each geome' in k),None)
            if result is None: raise ValueError('Missing paired IRC results')
            c=np.asarray(data[name],float).reshape(-1,nvar)
            r=np.asarray(result,float).reshape(-1,nres)
            if len(c)!=len(r):raise ValueError('IRC result/geometry counts differ')
            coords.extend(c);results.extend(r)
    if not coords:raise ValueError('No accepted IRC geometries in formatted checkpoint')
    coords=np.asarray(coords);results=np.asarray(results);s=results[:,1]
    if not np.isfinite(coords).all() or not np.isfinite(results).all() or not (np.any(s<0) and np.any(s>0)):
        raise ValueError('Both finite forward and reverse IRC paths are required')
    report={}
    for label,index in [('reverse',int(np.argmin(s))),('forward',int(np.argmax(s)))]:
        path=output/(label+'.xyz');write_xyz(path,numbers,coords[index].reshape(-1,3)*BOHR,'Accepted IRC '+label+' endpoint')
        report[label]=dict(xyz=str(path),reaction_coordinate=float(s[index]),energy_hartree=float(results[index,0]))
    return report


def stage_fchk(output, task, name):
    directory = Path(output)/task
    for stem in (CHECKPOINT_NAMES[name], 'gaussian'):
        path = directory/(stem+'.fchk')
        if path.exists():
            return path
    raise FileNotFoundError('Stage formatted checkpoint missing: '+str(directory))


def stage(cfg, reaction, name, task, xyz, charge, mult, state, retry_failed, timeout_attempts):
    fingerprint=digest(dict(config=cfg,task=task,input=Path(xyz).read_text(),charge=charge,multiplicity=mult))
    entry=state['stages'].get(name)
    if entry and entry['fingerprint'] != fingerprint:raise ValueError('Stage configuration/input changed; use another output root')
    if entry and entry['status']=='complete':
        result=Path(entry['output'])/task/'final.xyz'
        if not result.exists():raise ValueError('Completed stage geometry is missing')
        return Path(entry['output'])
    if entry and entry['status']=='failed' and not retry_failed:raise RuntimeError('Recorded failed stage; use --retry-failed explicitly')
    attempts=len(entry.get('attempts',[])) if entry else 0
    # Recover a normal completed runner output after a crash before state was committed.
    if entry and entry['status']=='running':
        previous=Path(entry['output']);summary=previous/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('completed') and (previous/task/'final.xyz').exists():
            entry['status']='complete';save(reaction/'state.json',state);return previous
    if attempts >= timeout_attempts:raise RuntimeError('Stage attempt limit reached')
    target=reaction/'stages'/name/f'attempt_{attempts+1:03d}'
    target.parent.mkdir(parents=True,exist_ok=True)
    attempt=dict(output=str(target),started=time.time())
    entry=dict(status='running',fingerprint=fingerprint,output=str(target),checkpoint_name=CHECKPOINT_NAMES[name],attempts=(entry or {}).get('attempts',[])+[attempt])
    state['stages'][name]=entry;save(reaction/'state.json',state)
    command=[sys.executable,'-m','gpu4pyscf_gau','run','--config',str(reaction/'config.json'),
             '--xyz',str(xyz),'--task',task,'--checkpoint-name',CHECKPOINT_NAMES[name],'--charge',str(charge),'--multiplicity',str(mult),'--output',str(target)]
    with (target.parent/f'attempt_{attempts+1:03d}.launcher.log').open('w') as log:
        with _ACTIVE_LOCK:
            if _CANCEL.is_set():raise KeyboardInterrupt('Scheduler cancelled')
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            _ACTIVE.add(p)
        attempt['core_cleanup'] = dict(files=[], bytes=0)
        try:
            while True:
                try:
                    rc=p.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    report=cleanup_core_dumps(target)
                    attempt['core_cleanup']['files'].extend(report['files'])
                    attempt['core_cleanup']['bytes']+=report['bytes']
        except BaseException:
            from .runner import stop_process
            stop_process(p);raise
        finally:
            report=cleanup_core_dumps(target)
            attempt['core_cleanup']['files'].extend(report['files'])
            attempt['core_cleanup']['bytes']+=report['bytes']
            with _ACTIVE_LOCK:_ACTIVE.discard(p)
    summary=target/'summary.json';passed=rc==0 and summary.exists() and json.loads(summary.read_text()).get('completed')
    passed=passed and (target/task/'final.xyz').exists()
    if passed:
        # Successful intermediate SCF checkpoints are not needed for stage resume.
        removed=0
        for checkpoint in target.glob('worker_*/evaluations/*/scf.chk'):
            checkpoint.unlink();removed+=1
        attempt['removed_intermediate_scf_checkpoints']=removed
    entry['status']='complete' if passed else 'failed';attempt.update(returncode=rc,ended=time.time());save(reaction/'state.json',state)
    if not passed:raise RuntimeError(f'{name} failed; inspect {target}')
    return target


def run_reaction(cfg, row, numbers, coords, source_index, meta, control, args):
    reaction=control/f'chunk_{row["chunk"]:05d}'/f'{row["index"]:06d}_{row["id"]}'
    reaction.mkdir(parents=True,exist_ok=True)
    cleanup=cleanup_core_dumps(reaction)
    path=reaction/'state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(record=row,stages={},status='pending')
    if cleanup['files']:
        state.setdefault('core_cleanup',[]).append(cleanup);save(path,state)
    if state['status']=='complete':return 'already_complete'
    if state['status'] in ['failed','rejected'] and not args.retry_failed:return 'already_'+state['status']
    try:
        if _CANCEL.is_set():raise KeyboardInterrupt('Scheduler cancelled')
        ts=reaction/'ts_input.xyz';write_xyz(ts,numbers,coords,row['id'])
        save(reaction/'config.json',cfg)
        save(reaction/'input.json',dict(record=row,charge=meta['charge'],multiplicity=meta['multiplicity'],source_index=source_index))
        state['status']='running';save(path,state)
        def calculate(name,task,xyz):
            return stage(cfg,reaction,name,task,xyz,meta['charge'],meta['multiplicity'],state,args.retry_failed,args.max_attempts)
        tsopt=calculate('tsopt','tsopt',ts)
        freq=calculate('ts_freq','freq',tsopt/'tsopt/final.xyz')
        frequencies=json.loads((freq/'summary.json').read_text())['results'][0]['frequencies_cm1']
        imaginary=[v for v in frequencies if v<args.imaginary_threshold]
        state['ts_validation']=dict(frequencies_cm1=frequencies,imaginary_threshold_cm1=args.imaginary_threshold,imaginary_count=len(imaginary))
        if not frequencies or len(imaginary)!=1:
            state.update(status='rejected',reason='Optimized TS does not have exactly one significant imaginary frequency')
            save(path,state);return 'rejected'
        irc=calculate('irc','irc',tsopt/'tsopt/final.xyz')
        endpoints=irc_endpoints(stage_fchk(irc,'irc','irc'),numbers,reaction)
        state['irc_endpoints']=endpoints;save(path,state)
        optimized={}
        for direction in ['reverse','forward']:
            optimized[direction]=calculate('endpoint_opt_'+direction,'opt',endpoints[direction]['xyz'])/'opt/final.xyz'
        # Both OPTs must succeed before either endpoint FREQ starts.
        state['endpoint_validation']={}
        for direction in ['reverse','forward']:
            freq=calculate('endpoint_freq_'+direction,'freq',optimized[direction])
            frequencies=json.loads((freq/'summary.json').read_text())['results'][0]['frequencies_cm1']
            imaginary=[v for v in frequencies if v<ENDPOINT_IMAGINARY_THRESHOLD]
            state['endpoint_validation'][direction]=dict(frequencies_cm1=frequencies,imaginary_threshold_cm1=ENDPOINT_IMAGINARY_THRESHOLD,imaginary_count=len(imaginary))
            save(path,state)
        if any(not v['frequencies_cm1'] or v['imaginary_count'] for v in state['endpoint_validation'].values()):
            state.update(status='rejected',reason='Endpoint FREQ has negative frequencies (<0 cm^-1) or no frequencies')
            save(path,state);return 'rejected'
        state.pop('reason',None)
        state.update(status='complete',completed=time.time(),scope='Single-imaginary TS, bidirectional IRC, converged endpoint OPTs with no negative frequencies; reference R/P identity not verified')
        save(path,state);return 'complete'
    except (KeyboardInterrupt,SystemExit):raise
    except Exception as exc:
        state.update(status='failed',reason=str(exc));save(path,state);return 'failed'


def bounded_results(pool, items, submit, capacity):
    """Keep at most capacity reactions in flight; refill after each completion."""
    pending={};items=iter(items)
    while True:
        while len(pending)<capacity:
            try:item=next(items)
            except StopIteration:break
            pending[submit(pool,item)]=item
        if not pending:return
        done,_=wait(pending,return_when=FIRST_COMPLETED)
        for future in done:
            item=pending.pop(future)
            yield item,future.result()


def run(args):
    import h5py
    cfg=load_config(args.config)
    if not cfg['gaussian']['formchk']:raise ValueError('Pipeline requires formchk and accepted geometries')
    shards=getattr(args,'shards',None) or [args.shard]
    concurrency=getattr(args,'concurrency',4)
    if len(set(shards))!=len(shards) or any(s is None or s<0 for s in shards) or concurrency not in range(1,5):
        raise ValueError('Specify distinct shards and concurrency 1..4')
    if args.limit is not None and args.limit<1 or args.chunk is not None and args.chunk<0:raise ValueError('Invalid limit/chunk')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    manifests=Path(args.manifests).resolve();meta=json.loads((manifests/'dataset.json').read_text())
    if any(s>=meta['shards'] for s in shards):raise ValueError('Shard out of range')
    selected=set(args.indices) if args.indices else None
    if selected is not None and any(i<0 or i>=meta['count'] or i%meta['shards'] not in shards for i in selected):
        raise ValueError('Selected indices are outside selected shards')
    _CANCEL.clear()
    controls={};counts={s:Counter() for s in shards};processed=Counter()
    with ExitStack() as stack:
        for shard in sorted(shards):
            manifest=manifests/meta['manifests'][shard]
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=meta['manifest_sha256'][manifest.name]:raise ValueError('Manifest changed')
            settings=dict(dataset=meta,config=cfg,imaginary_threshold=args.imaginary_threshold,endpoint_imaginary_threshold=ENDPOINT_IMAGINARY_THRESHOLD,shard=shard,pipeline_version=PIPELINE_VERSION)
            signature=digest(settings);control=root/f'shard_{shard:02d}';control.mkdir(exist_ok=True)
            lock=stack.enter_context((control/'run.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            metadata=control/'run.json'
            if metadata.exists() and json.loads(metadata.read_text())['fingerprint']!=signature:raise ValueError('Run settings changed; use another output root')
            save(metadata,dict(fingerprint=signature,settings=settings,concurrency=concurrency));controls[shard]=control
        f=stack.enter_context(h5py.File(meta['path'],'r'))
        if identity(Path(meta['path']),f)!={k:meta[k] for k in ['path','size','mtime_ns','dataset_version','count']}:
            raise ValueError('Source HDF5 identity changed')
        def records():
            emitted=0
            for shard in shards:
                with (manifests/meta['manifests'][shard]).open() as rows:
                    for line in rows:
                        row=json.loads(line)
                        if selected is not None and row['index'] not in selected:continue
                        if args.chunk is not None and row['chunk']!=args.chunk:continue
                        if args.limit is not None and emitted>=args.limit:return
                        emitted+=1;yield row
        def submit(pool,row):
            i=row['index'];a,b=map(int,f['offsets/atom'][i:i+2])
            if f['records/id'][i].decode()!=row['id'] or b-a!=row['natoms']:raise ValueError('Manifest does not match HDF5 row')
            # Only the scheduler thread accesses HDF5. Copy just one geometry per slot.
            return pool.submit(run_reaction,cfg,row,f['atoms/atomic_numbers'][a:b],f['TS/coordinates'][a:b],int(f['records/source_index'][i]),meta,controls[row['shard']],args)
        started=time.time();pool=ThreadPoolExecutor(max_workers=concurrency)
        try:
            for row,status in bounded_results(pool,records(),submit,concurrency):
                shard=row['shard'];processed[shard]+=1;counts[shard][status]+=1
                save(controls[shard]/'progress.json',dict(processed=processed[shard],counts=dict(counts[shard]),last_record=row,updated=time.time()))
                print(json.dumps(dict(index=row['index'],id=row['id'],status=status),ensure_ascii=False),flush=True)
        except BaseException:
            cancel_active();raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        for shard,control in controls.items():
            save(control/'summary.json',dict(processed=processed[shard],counts=dict(counts[shard]),started=started,finished=time.time(),concurrency=concurrency))
        return 1 if any(c[k] for c in counts.values() for k in ['failed','rejected','already_failed','already_rejected']) else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--h5',required=True);a.add_argument('--output',required=True);a.add_argument('--shards',type=int,default=8);a.add_argument('--chunk-size',type=int,default=100)
    a=sub.add_parser('run');a.add_argument('--config',required=True);a.add_argument('--manifests',required=True);g=a.add_mutually_exclusive_group(required=True);g.add_argument('--shard',type=int);g.add_argument('--shards',type=int,nargs='+');a.add_argument('--concurrency',type=int,default=4);a.add_argument('--output',required=True);a.add_argument('--chunk',type=int);a.add_argument('--limit',type=int);a.add_argument('--indices',type=int,nargs='+');a.add_argument('--retry-failed',action='store_true');a.add_argument('--max-attempts',type=int,default=2);a.add_argument('--imaginary-threshold',type=float,default=-20.)
    args=p.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('Scheduler termination')
    signal.signal(signal.SIGTERM,stop)
    try:
        if args.command=='prepare':print(json.dumps(prepare(args.h5,args.output,args.shards,args.chunk_size),indent=2));return 0
        if args.max_attempts<1 or not math.isfinite(args.imaginary_threshold) or args.imaginary_threshold>=0:raise ValueError('Invalid validation/attempt limit')
        disable_core_dumps()
        return run(args)
    except KeyboardInterrupt:return 130
    except (ValueError,OSError,RuntimeError) as exc:print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())

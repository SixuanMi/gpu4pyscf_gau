"""User-facing direct runner and optional qzcli adapter."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from .config import DEFAULT, load_config
from .runner import run_jobs


def qz_command(platform, run_args):
    required = ['workspace','project','compute_group','spec','image','python','repository']
    missing = [key for key in required if not platform.get(key)]
    if missing:
        raise ValueError('qzcli configuration missing: '+', '.join(missing))
    repository = str(Path(platform['repository']).expanduser())
    # Paths describe the mounted compute-node filesystem, not the submission host.
    run = ['env', f'PYTHONPATH={repository}/src', platform['python'], '-m', 'gpu4pyscf_gau', *run_args]
    command = shlex.join(run)
    argv = [platform.get('qzcli','qzcli'), 'create', '--name', platform['name'], '--command', command,
            '--workspace',platform['workspace'], '--project',platform['project'],
            '--compute-group',platform['compute_group'], '--spec',platform['spec'],
            '--image',platform['image'], '--image-type',platform.get('image_type','SOURCE_PRIVATE'),
            '--instances','1', '--shm',str(platform.get('shm_gib',64)),
            '--priority',str(platform.get('priority',6)), '--framework','pytorch', '--json']
    return argv


def build_parser():
    parser = argparse.ArgumentParser(description='GPU4PySCF electronic structure + Gaussian External geometry workflows')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run one calculation on the current compute node')
    run.add_argument('--config', required=True)
    run.add_argument('--xyz', required=True)
    run.add_argument('--task', choices=list(DEFAULT['routes']), required=True)
    run.add_argument('--charge', type=int, required=True)
    run.add_argument('--multiplicity', type=int, required=True)
    run.add_argument('--output', required=True)
    batch = sub.add_parser('batch', help='Run independent cases serially with a shared worker')
    batch.add_argument('--config', required=True)
    batch.add_argument('--manifest', required=True, help='JSON array; xyz paths relative to this manifest')
    batch.add_argument('--output', required=True)
    check = sub.add_parser('check-config', help='Validate configuration without importing GPU libraries')
    check.add_argument('--config', required=True)
    qz = sub.add_parser('qz-submit', help='Preview or submit the same run/batch command using qzcli')
    qz.add_argument('--platform', required=True)
    qz.add_argument('--submit', action='store_true', help='Actually invoke qzcli; default only prints the command')
    qz.add_argument('run_args', nargs=argparse.REMAINDER, help='After --, pass run/batch and its normal options')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'qz-submit':
            p = json.loads(Path(args.platform).read_text())
            command = args.run_args[1:] if args.run_args[:1] == ['--'] else args.run_args
            if not command or command[0] not in ('run','batch'):
                raise ValueError('After -- provide a run or batch command')
            build_parser().parse_args(command)
            cmd = qz_command(p, command)
            if p.get('mode','cli') == 'console':
                from .qzconsole import payload, submit
                template = Path(p['resource_template']).expanduser()
                if not template.is_absolute():
                    p['resource_template'] = str(Path(args.platform).resolve().parent/template)
                body = payload(p, cmd[cmd.index('--command')+1])
                print(json.dumps(submit(p,body) if args.submit else body, indent=2))
                return 0
            if p.get('mode','cli') != 'cli':
                raise ValueError('qzcli mode must be cli or console')
            if not args.submit:
                print(shlex.join(cmd)); return 0
            return subprocess.run(cmd).returncode
        cfg = load_config(args.config)
        if args.command == 'check-config':
            print(json.dumps(cfg, indent=2)); return 0
        if args.command == 'run':
            jobs = [dict(name=args.task, task=args.task, xyz=str(Path(args.xyz).resolve()),
                         charge=args.charge, multiplicity=args.multiplicity)]
        else:
            path = Path(args.manifest).resolve()
            jobs = json.loads(path.read_text())
            if not isinstance(jobs, list) or not jobs:
                raise ValueError('Batch manifest must be a nonempty JSON array')
            for job in jobs:
                for key in ['name','task','xyz','charge','multiplicity']:
                    if key not in job:
                        raise ValueError('Manifest case missing '+key)
                job['xyz'] = str((path.parent/job['xyz']).resolve())
        # Ensure scheduler cancellation closes both subprocess groups and archives diagnostics.
        def terminate(signum, frame):
            raise KeyboardInterrupt('Termination requested')
        signal.signal(signal.SIGTERM, terminate)
        return 0 if run_jobs(cfg, jobs, args.output) else 1
    except KeyboardInterrupt:
        print('Interrupted; child processes stopped and available diagnostics archived.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'gpu-gau: {exc}', file=sys.stderr)
        return 2

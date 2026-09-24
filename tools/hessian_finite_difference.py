"""Check one analytic Hessian column against central differences of gradients."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--axis',type=int,default=3)
    p.add_argument('--steps',type=float,nargs='+',default=[.0005,.00025])
    p.add_argument('--conv-tol',type=float)
    p.add_argument('--conv-tol-grad',type=float)
    args = p.parse_args()
    import numpy as np
    from gpu4pyscf_gau.worker import Engine
    args.output.mkdir(parents=True,exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='hessian-fd-'))
    cfg = json.loads((args.case/'config.json').read_text())
    if args.conv_tol is not None:cfg['conv_tol'] = args.conv_tol
    if args.conv_tol_grad is not None:cfg['conv_tol_grad'] = args.conv_tol_grad
    if any(not np.isfinite(step) or step <= 0 for step in args.steps):
        raise ValueError('Finite difference steps must be finite and positive')
    req = json.loads((args.case/'request.json').read_text());req['deriv']=1
    n = len(req['numbers'])*3
    if not 0 <= args.axis < n:
        raise ValueError('axis out of range')
    ref = np.load(args.case/'derivatives.npz')['hessian'][:,args.axis]
    engine = Engine(cfg,work)
    results = []
    try:
        center = engine.evaluate(req)
        seed = engine.previous
        for step in args.steps:
            gradients = []
            for sign in [1,-1]:
                displaced = copy.deepcopy(req)
                displaced['coords_bohr'][args.axis//3][args.axis%3] += sign*step
                engine.previous = seed
                value = engine.evaluate(displaced)
                gradients.append(np.asarray(value['gradient_hartree_bohr']).reshape(-1))
            column = (gradients[0]-gradients[1])/(2*step)
            error = column-ref
            results.append(dict(step_bohr=step,max_abs_error=float(abs(error).max()),
                rms_error=float(np.sqrt(np.mean(error**2))),column=column.tolist()))
        report = dict(axis=args.axis,configuration=cfg,center_energy=center['energy_hartree'],results=results,
            reference_column=ref.tolist(),passed=all(x['max_abs_error']<1e-5 for x in results))
        (args.output/'finite_difference.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['reference_column','results']}))
        print(json.dumps([{k:v for k,v in x.items() if k!='column'} for x in results]))
        if not report['passed']:
            raise RuntimeError('Hessian finite difference error exceeds 1e-5 Eh/Bohr^2')
    finally:
        for path in work.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.jsonl','.log']:
                dst = args.output/'worker'/path.relative_to(work)
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        shutil.rmtree(work)


if __name__ == '__main__':
    main()

"""Fixed-geometry Hessian validation; run with a GPU Python, one fresh process per case."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    args = p.parse_args()
    case = args.case.resolve()
    import cupy as cp
    import numpy as np
    import gpu4pyscf
    from gpu4pyscf_gau.worker import Engine
    from gpu4pyscf_gau.hessian_memory import backend_diagnostics

    work = Path(tempfile.mkdtemp(prefix='hessian-check-'))
    cfg = json.loads((case/'config.json').read_text())
    req = json.loads((case/'request.json').read_text())
    engine = Engine(cfg,work)
    diagnostics = backend_diagnostics()
    diagnostics['lowmem_references'] = []
    package = Path(gpu4pyscf.__file__).parent
    for path in package.rglob('*.py'):
        hits = [dict(line=i,text=s.strip()) for i,s in enumerate(path.read_text().splitlines(),1)
                if 'lowmem' in s.lower() or 'low_mem' in s.lower()]
        if hits:
            diagnostics['lowmem_references'].append(dict(file=str(path.relative_to(package)),hits=hits))
    (case/'backend.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    phase = 'ready'
    def mark(value):
        nonlocal phase
        phase = value
        (case/'phase.txt').write_text(value)
        with (case/'phases.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=value,time=time.time()))+'\n')
    source = textwrap.dedent(inspect.getsource(Engine.evaluate))
    for original,replacement in [
        ('energy = float(mf.kernel(dm0=dm0))', "mark('scf'); energy = float(mf.kernel(dm0=dm0))"),
        ('gradient = self.array(grad.kernel())', "mark('gradient'); gradient = self.array(grad.kernel())"),
        ('hessian = self.array(hdriver.kernel())', "mark('hessian'); hessian = self.array(hdriver.kernel())"),
    ]:
        assert source.count(original) == 1
        source = source.replace(original,replacement)
    namespace = dict(Engine.evaluate.__globals__,mark=mark)
    exec(compile(source,'<phase-marked Engine.evaluate>','exec'),namespace)
    engine.evaluate = namespace['evaluate'].__get__(engine,Engine)
    started = time.perf_counter()
    result = dict(completed=False)
    try:
        value = engine.evaluate(req)
        result.update(completed=True, **{k:value[k] for k in ['energy_hartree','scf_cycles',
            'scf_seconds','gradient_seconds','hessian_seconds','hessian_max_asymmetry','nao']})
        np.savez(case/'derivatives.npz',energy=value['energy_hartree'],
            gradient=value['gradient_hartree_bohr'],hessian=value['hessian_hartree_bohr2'])
    except Exception as exc:
        result.update(error=repr(exc),failed_phase=phase)
        (case/'traceback.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        result['total_seconds'] = time.perf_counter()-started
        (case/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        for path in work.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.jsonl','.log','.txt']:
                dest = case/'worker'/path.relative_to(work)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
        # Successful large checkpoints are unnecessary for this numerical replay.
        shutil.rmtree(work)
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()

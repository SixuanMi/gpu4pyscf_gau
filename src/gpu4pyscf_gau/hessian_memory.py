"""Opt-in, source-guarded DF Hessian blocking for the audited GPU4PySCF build.

Only audited J/K derivative and response functions are replaced in this serial worker.
No installed files, SCF settings, gradient settings, or contraction backends change.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
import threading

DEFAULT = dict(policy='off', memory_fraction=0.6, reserve_gib=8.0,
               aux_batch_cap=256, aux_block_cap=128, integral_block_cap=128,
               response_aux_cap=256, response_dm_cap=8, h1_aux_cap=256)
SOURCE_HASHES = {
    'rhf': '45ef4f502fcc8a13de662c995b7dbf7be70f9f89c1e17f4f877b339b08c43b43',
    'uhf': 'e4f8e50be108fee472b4832c6ab849cd0620d8abae21c26b84bcef0b2193a3d0',
}
_LOCK = threading.Lock()


def validate(value):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value)-set(DEFAULT):
        raise ValueError('gpu.hessian_memory must contain only documented policy keys')
    cfg = dict(DEFAULT, **value)
    if cfg['policy'] not in ('off', 'conservative'):
        raise ValueError('hessian_memory.policy must be off or conservative')
    for key in ('memory_fraction', 'reserve_gib'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (float, int)) or not math.isfinite(cfg[key]):
            raise ValueError(f'hessian_memory.{key} must be finite and numeric')
    if not 0 < cfg['memory_fraction'] <= 0.8 or cfg['reserve_gib'] < 0:
        raise ValueError('Hessian budget fraction must be in (0, 0.8]; reserve must be nonnegative')
    for key, alignment in [('aux_batch_cap',16), ('aux_block_cap',8), ('integral_block_cap',8),
                           ('response_aux_cap',8), ('response_dm_cap',1), ('h1_aux_cap',8)]:
        n = cfg[key]
        if isinstance(n, bool) or not isinstance(n, int) or n < alignment or n % alignment:
            raise ValueError(f'hessian_memory.{key} must be a positive multiple of {alignment}')
    return cfg


def budget(free_bytes, cfg):
    result = min(int(free_bytes * cfg['memory_fraction']),
                 int(free_bytes - cfg['reserve_gib'] * 2**30))
    if result <= 0:
        raise MemoryError('Insufficient free GPU memory for the Hessian safety reserve')
    return result


def cap_block(original, limit):
    # Do not enlarge a small tail or an upstream-selected smaller block.
    if original <= 0:
        raise MemoryError('Upstream Hessian selected a nonpositive block')
    return min(original, limit)


class _Blocking(ast.NodeTransformer):
    def __init__(self, phase='jk'):
        self.response = phase == 'response'
        self.h1 = phase == 'h1'
        self.counts = dict(budget=0, aux_batch_cap=0, aux_block_cap=0, integral_block_cap=0)
        if self.response:self.counts = dict(budget=0, response_aux_cap=0, response_dm_cap=0)
        if self.h1:self.counts = dict(budget=0, h1_aux_cap=0, integral_block_cap=0)

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name, value = node.targets[0].id, node.value
        if not isinstance(value, ast.Call):
            return node
        func = value.func.id if isinstance(value.func, ast.Name) else getattr(value.func, 'attr', '')
        key = None
        if name == 'mem_avail' and func == 'get_avail_mem':
            key = 'budget'
        elif self.response and name == 'blksize' and func == 'get_blksize':
            key = 'response_aux_cap'
        elif self.response and name == 'dm_batch_size' and func == 'min':
            key = 'response_dm_cap'
        elif self.h1 and name == 'batch_size' and func == 'min' and any(
                isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id == 'max'
                for a in value.args):
            key = 'h1_aux_cap'
        elif name == 'aux_batch_size' and func == '_balance_batch_size':
            key = 'aux_batch_cap'
        elif name == 'aux_blksize' and func == 'min':
            key = 'aux_block_cap'
        elif name == 'blksize' and func == 'min' and any(
                isinstance(a, ast.Name) and a.id == 'batch_size' for a in value.args):
            key = 'integral_block_cap'
        if key:
            self.counts[key] += 1
            args = [ast.Constant(key), value]
            if key == 'h1_aux_cap':args.append(ast.Name(id='largest_shell_nao',ctx=ast.Load()))
            node.value = ast.Call(func=ast.Name(id='_gpu_gau_memory_adjust', ctx=ast.Load()),
                                  args=args, keywords=[])
        return node


def _prepare(module, cfg, events, function_name='_jk_energy_per_atom'):
    source_file = Path(inspect.getsourcefile(module))
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    name = module.__name__.rsplit('.',1)[-1]
    if digest != SOURCE_HASHES.get(name):
        raise RuntimeError(f'Unsupported GPU4PySCF DF Hessian source: {source_file} sha256={digest}; '
                           'disable the policy or validate this build before adding support')
    original = getattr(module,function_name)
    tree = ast.parse(inspect.getsource(original))
    transform = _Blocking(phase={'_get_jk':'response','_get_veff':'h1'}.get(function_name,'jk'))
    tree = ast.fix_missing_locations(transform.visit(tree))
    if any(n == 0 for n in transform.counts.values()):
        raise RuntimeError('Incomplete Hessian patch; no function has been changed')

    def adjust(key, value, minimum=1):
        if key != 'budget' and cfg[key] < minimum:
            raise ValueError(f'hessian_memory.{key} must be at least {minimum} for this basis shell')
        new = budget(value, cfg) if key == 'budget' else cap_block(value, cfg[key])
        events.append(dict(module=name, function=function_name, kind=key,
                           upstream=int(value), selected=int(new)))
        return new

    namespace = dict(original.__globals__, _gpu_gau_memory_adjust=adjust)
    exec(compile(tree, f'<gpu-gau conservative {source_file}>', 'exec'), namespace)
    replacement = namespace[function_name]
    return original, replacement, dict(file=str(source_file), function=function_name, sha256=digest,
                                       transformations=transform.counts)


@contextmanager
def hessian_memory_scope(value, report):
    cfg = validate(value)
    report.update(configuration=cfg, applied=False, events=[])
    if cfg['policy'] == 'off':
        yield
        return
    # A serial isolated worker is required; never allow competing global patches.
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Concurrent Hessian memory scopes are unsupported')
    installed = []
    try:
        from gpu4pyscf.lib import multi_gpu
        if multi_gpu.num_devices != 1:
            raise ValueError('Conservative Hessian policy currently supports one GPU')
        modules = [importlib.import_module('gpu4pyscf.df.hessian.'+n) for n in ('rhf','uhf')]
        targets = [(m,f) for m in modules for f in ('_jk_energy_per_atom','_get_veff')]
        targets.append((modules[0],'_get_jk'))
        prepared = [(m,f,*_prepare(m,cfg,report['events'],f)) for m,f in targets]
        report['sources'] = [x[4] for x in prepared]
        for module, function_name, original, replacement, _ in prepared:
            setattr(module,function_name,replacement)
            installed.append((module,function_name,original))
        report['applied'] = True
        yield
    finally:
        for module, function_name, original in reversed(installed):
            setattr(module,function_name,original)
        _LOCK.release()


def backend_diagnostics():
    """Run in the GPU worker, never during config loading on a submit host."""
    from gpu4pyscf.lib import cutensor
    info = dict(contract_engine_environment=os.environ.get('CONTRACT_ENGINE'),
                effective_backend=cutensor.contract_engine or 'cutensor',
                wrapper_file=cutensor.__file__, imports={})
    for name in ('cupyx.cutensor', 'cupy_backends.cuda.libs.cutensor'):
        try:
            mod = importlib.import_module(name)
            info['imports'][name] = dict(ok=True, file=getattr(mod,'__file__',None))
        except (ImportError, AttributeError) as exc:
            info['imports'][name] = dict(ok=False, error=repr(exc))
    return info

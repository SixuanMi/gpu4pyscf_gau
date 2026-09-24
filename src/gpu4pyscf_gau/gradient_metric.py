"""Scoped, opt-in stable DF gradient metric application for an audited RKS build."""
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import threading

SOURCE_HASH = '1b5655cf302c8f3206e7cd264d0022153592857bd34304524507a31ce50cb172'
_LOCK = threading.Lock()
_ORIGINAL = """    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'CD')(aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    j2c = aux_coeff = None
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    metric = j3c_oo = None
"""
_SOLVE = """    if mol.omega <= 0 and not auxmol.mol.cart:
        metric_solver = _gen_metric_solver(j2c, 'CD')
    else:
        metric_solver = _gen_metric_solver(j2c, 'ED')
    rhs = cp.einsum('uv,vij->uij', aux_coeff.T, j3c_oo)
    dm_oo = metric_solver(rhs)
    rhs = None
    dm_oo = cp.einsum('uv,vij->uij', aux_coeff, dm_oo)
    j2c = aux_coeff = j3c_oo = metric_solver = None
"""


def validate(value):
    if value not in ('original', 'solve'):
        raise ValueError('gpu.df_gradient_metric must be original or solve')
    return value


@contextmanager
def gradient_metric_scope(value, report):
    mode = validate(value)
    report.update(mode=mode, applied=False)
    if mode == 'original':
        yield
        return
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Concurrent DF gradient metric scopes are unsupported')
    module = original = None
    try:
        from gpu4pyscf.lib import multi_gpu
        if multi_gpu.num_devices != 1:
            raise ValueError('Stable DF gradient metric currently supports one GPU')
        module = importlib.import_module('gpu4pyscf.df.grad.rhf')
        path = Path(inspect.getsourcefile(module))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SOURCE_HASH:
            raise RuntimeError(f'Unsupported DF gradient source: {path} sha256={digest}; '
                               'validate this build before enabling df_gradient_metric=solve')
        function = module._jk_energy_per_atom
        source = inspect.getsource(function)
        if source.count(_ORIGINAL) != 1:
            raise RuntimeError('Unexpected DF gradient metric expression; nothing changed')
        namespace = dict(function.__globals__)
        exec(compile(source.replace(_ORIGINAL, _SOLVE),
                     f'<gpu-gau stable gradient metric {path}>', 'exec'), namespace)
        original = function
        module._jk_energy_per_atom = namespace['_jk_energy_per_atom']
        report.update(applied=True, source=str(path), sha256=digest,
                      function='_jk_energy_per_atom',
                      operation='C solve(J2C, C.T B); original decomposition and cutoff retained')
        yield
    finally:
        if original is not None:
            module._jk_energy_per_atom = original
        _LOCK.release()

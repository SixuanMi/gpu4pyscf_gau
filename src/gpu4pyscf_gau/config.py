"""Portable configuration; importing this module never imports CUDA libraries."""
import copy
import json
import math
import os
from pathlib import Path
import shutil

DEFAULT = {
    'gaussian': {'executable': 'g16', 'exedir': None, 'memory': '32GB', 'threads': 1,
                 'formchk': 'formchk', 'environment': {}},
    'runtime': {'scratch_dir': None, 'worker_python': None, 'worker_environment': {},
                'timeout_seconds': 43200, 'startup_timeout_seconds': 180},
    'gpu': {'method': 'b3lyp', 'dispersion': 'd3bj', 'basis': 'def2-svp',
            'density_fit': True, 'auxbasis': 'def2-universal-jkfit', 'with_solvent': False,
            'atom_grid': [99, 590], 'pruning': 'nwchem', 'conv_tol': 1e-10,
            'conv_tol_grad': 1e-7, 'direct_scf_tol': 1e-14, 'max_cycle': 100,
            'threads': 1, 'memory_mb': 32000, 'reuse_guess': True,
            'reset_at_initial_geometry': True, 'hessian_memory': {'policy':'off'}},
    'routes': {'sp': '', 'opt': 'Opt=(NoMicro,Redundant,MaxCycles=100)',
               'tsopt': 'Opt=(TS,CalcFC,NoEigenTest,NoMicro,Redundant,MaxCycles=100)',
               'irc': 'IRC=(CalcFC,HPC,MaxPoints=10,StepSize=10)',
               'freq': 'Freq', 'force': 'Force'},
}


def load_config(filename):
    filename = Path(filename).resolve()
    custom = json.loads(filename.read_text())
    if not isinstance(custom, dict):
        raise ValueError('Configuration must be a JSON object')
    result = copy.deepcopy(DEFAULT)
    for section, values in custom.items():
        if section not in result or not isinstance(values, dict):
            raise ValueError(f'Unknown/invalid configuration section: {section}')
        for key, value in values.items():
            if key not in result[section]:
                raise ValueError(f'Unknown configuration key: {section}.{key}')
            result[section][key] = value
    gpu = result['gpu']
    from .hessian_memory import validate
    gpu['hessian_memory'] = validate(gpu['hessian_memory'])
    if gpu['hessian_memory']['policy'] != 'off' and not gpu['density_fit']:
        raise ValueError('Conservative Hessian memory policy requires density fitting')
    if gpu['with_solvent']:
        raise ValueError('Solvent is not implemented by this External bridge')
    if gpu['pruning'] not in ('nwchem', 'none'):
        raise ValueError('gpu.pruning must be nwchem or none')
    for key in ['conv_tol', 'conv_tol_grad', 'direct_scf_tol', 'memory_mb']:
        if not math.isfinite(float(gpu[key])) or float(gpu[key]) <= 0:
            raise ValueError(f'gpu.{key} must be positive and finite')
    for key in ['threads', 'max_cycle']:
        if not isinstance(gpu[key], int) or isinstance(gpu[key], bool) or gpu[key] < 1:
            raise ValueError(f'gpu.{key} must be a positive integer')
    for key in ['timeout_seconds', 'startup_timeout_seconds']:
        if not math.isfinite(float(result['runtime'][key])) or result['runtime'][key] <= 0:
            raise ValueError(f'runtime.{key} must be positive')
    for section, key in [('gaussian', 'executable'), ('gaussian', 'exedir'),
                         ('gaussian', 'formchk'), ('runtime', 'scratch_dir'),
                         ('runtime', 'worker_python')]:
        value = result[section][key]
        if value:
            value = os.path.expandvars(os.path.expanduser(str(value)))
            if '/' in value or key in ('scratch_dir','exedir'):
                value = str((filename.parent / value).resolve())
            result[section][key] = value
    if not result['gaussian']['exedir'] and '/' in result['gaussian']['executable']:
        result['gaussian']['exedir'] = str(Path(result['gaussian']['executable']).parent)
    return result


def executable(value):
    found = shutil.which(value)
    if not found:
        raise ValueError(f'Executable not found or not executable: {value}')
    return str(Path(found).absolute())

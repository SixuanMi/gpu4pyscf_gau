"""Serial per-calculation GPU worker; keeps only the last successful SCF guess."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback

from .protocol import read_input, format_output


class Engine:
    def __init__(self, config, root):
        import numpy as np
        import cupy as cp
        import pyscf
        import gpu4pyscf
        from pyscf import gto, dft, scf, lib
        self.np, self.cp, self.gto, self.dft, self.scf = np, cp, gto, dft, scf
        self.config, self.root = config, root
        lib.num_threads(config.get('threads', 1))
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA GPU visible to worker')
        self.previous = None
        self.anchor = None
        self.counter = 0
        (root / 'evaluations').mkdir()
        (root / 'versions.json').write_text(json.dumps(dict(
            pyscf=pyscf.__version__, gpu4pyscf=gpu4pyscf.__version__, cupy=cp.__version__,
            device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()), indent=2))

    def array(self, x):
        return self.cp.asnumpy(x) if isinstance(x, self.cp.ndarray) else self.np.asarray(x)

    def evaluate(self, req):
        np, cp, cfg = self.np, self.cp, self.config
        if cfg.get('with_solvent', False):
            raise ValueError('This benchmark is gas phase only')
        self.counter += 1
        directory = self.root / 'evaluations' / f'{self.counter:04d}'
        directory.mkdir()
        (directory / 'request.json').write_text(json.dumps(req, indent=2))
        start = time.perf_counter()
        timing = {}
        phase = start
        mol = self.gto.M(atom=list(zip(req['numbers'], req['coords_bohr'])), unit='Bohr',
                         basis=cfg['basis'], charge=req['charge'], spin=req['multiplicity']-1,
                         symmetry=False, verbose=4, output=str(directory / 'pyscf.log'),
                         max_memory=cfg.get('memory_mb', 16000))
        signature = (tuple(req['numbers']), req['charge'], req['multiplicity'])
        if cfg['method'].lower() == 'hf':
            mf = self.scf.UHF(mol) if mol.spin else self.scf.RHF(mol)
        else:
            mf = self.dft.UKS(mol) if mol.spin else self.dft.RKS(mol)
            mf.xc = cfg['method']
            mf.grids.atom_grid = tuple(cfg.get('atom_grid', [99, 590]))
            pruning = cfg.get('pruning', 'none')
            mf.grids.prune = {'none':None, 'nwchem':self.dft.gen_grid.nwchem_prune}[pruning]
            if cfg.get('dispersion'):
                mf.disp = cfg['dispersion']
        if cfg.get('density_fit', False):
            mf = mf.density_fit(auxbasis=cfg.get('auxbasis', 'def2-universal-jkfit'))
        mf = mf.to_gpu()
        mf.conv_tol = cfg.get('conv_tol', 1e-10)
        mf.conv_tol_grad = cfg.get('conv_tol_grad', 1e-7)
        mf.direct_scf_tol = cfg.get('direct_scf_tol', 1e-14)
        mf.max_cycle = cfg.get('max_cycle', 100)
        mf.chkfile = str(directory / 'scf.chk')
        timing['molecule_and_scf_object_seconds'] = time.perf_counter()-phase
        phase = time.perf_counter()
        dm0 = None
        guess = 'default'
        if self.previous is not None and cfg.get('reuse_guess', True) and not req.get('cold', False):
            old_signature, old_mol, old_dm = self.previous
            if cfg.get('reset_at_initial_geometry') and self.anchor is not None:
                anchor_sig, anchor_mol, anchor_dm = self.anchor
                if (signature == anchor_sig and self.counter > 2
                    and np.max(np.abs(mol.atom_coords()-anchor_mol.atom_coords())) < 1e-7
                    and np.max(np.abs(mol.atom_coords()-old_mol.atom_coords())) > 1e-3):
                    old_signature, old_mol, old_dm = self.anchor
                    guess = 'initial_geometry_converged_density'
            if signature != old_signature:
                raise ValueError('Atom order, charge or multiplicity changed within one calculation')
            if old_dm.ndim == 2:
                dm = self.scf.addons.project_dm_nr2nr(old_mol, old_dm, mol)
            else:
                dm = np.array([self.scf.addons.project_dm_nr2nr(old_mol, d, mol) for d in old_dm])
            # Normalize the projected density to the correct spin electron counts.
            overlap = mol.intor_symmetric('int1e_ovlp')
            if dm.ndim == 2:
                dm *= mol.nelectron / np.einsum('ij,ji->', dm, overlap)
            else:
                for i, electrons in enumerate(mol.nelec):
                    count = np.einsum('ij,ji->', dm[i], overlap)
                    if electrons == 0:
                        dm[i] = 0
                    else:
                        dm[i] *= electrons / count
            dm0 = cp.asarray(dm)
            if guess == 'default':
                guess = 'projected_previous_converged_density'
        cycles = []
        timing['guess_seconds'] = time.perf_counter()-phase
        mf.callback = lambda env: cycles.append(int(env['cycle']) + 1)
        t = time.perf_counter()
        energy = float(mf.kernel(dm0=dm0))
        cp.cuda.get_current_stream().synchronize()
        scf_seconds = time.perf_counter() - t
        if not mf.converged or not np.isfinite(energy):
            raise RuntimeError('SCF failed; no result returned to Gaussian, prior good guess preserved')
        phase = time.perf_counter()
        dm = self.array(mf.make_rdm1())
        dm_total = dm if dm.ndim == 2 else dm.sum(axis=0)
        dipole = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
        dipole -= np.einsum('xij,ji->x', mol.intor_symmetric('int1e_r', comp=3), dm_total)
        timing['density_and_dipole_seconds'] = time.perf_counter()-phase
        gradient = None
        grad_seconds = 0
        if req['deriv'] in (1, 2):
            grad = mf.nuc_grad_method()
            if cfg.get('density_fit'):
                grad.auxbasis_response = True
            if cfg['method'].lower() != 'hf':
                grad.grid_response = True
            t = time.perf_counter()
            gradient = self.array(grad.kernel())
            cp.cuda.get_current_stream().synchronize()
            grad_seconds = time.perf_counter() - t
            if gradient.shape != (mol.natm, 3) or not np.isfinite(gradient).all():
                raise RuntimeError('Invalid gradient; result rejected')
        elif req['deriv'] != 0:
            raise ValueError('Unsupported derivative order')
        hessian = None
        hess_seconds = 0
        asymmetry = None
        if req['deriv'] == 2:
            hdriver = mf.Hessian()
            if cfg.get('density_fit'):
                hdriver.auxbasis_response = 2
            if cfg['method'].lower() != 'hf':
                hdriver.grid_response = True
            t = time.perf_counter()
            hessian = self.array(hdriver.kernel()).transpose(0,2,1,3).reshape(3*mol.natm,3*mol.natm)
            cp.cuda.get_current_stream().synchronize()
            hess_seconds = time.perf_counter()-t
            asymmetry = float(np.max(np.abs(hessian-hessian.T)))
            if not np.isfinite(hessian).all() or asymmetry > 5e-5:
                raise RuntimeError(f'Invalid/asymmetric Hessian: {asymmetry}')
            hessian = (hessian+hessian.T)*0.5
        if not np.isfinite(dipole).all():
            raise RuntimeError('Invalid dipole')
        spin_squared = float(mf.spin_square()[0]) if mol.spin else 0.0
        result = dict(energy_hartree=energy, dipole_au=dipole.tolist(),
                      spin_squared=spin_squared, multiplicity=req['multiplicity'],
                      nelectron=mol.nelectron, nao=mol.nao_nr(),
                      ecp_core_electrons=[mol.atom_nelec_core(i) for i in range(mol.natm)],
                      gradient_hartree_bohr=None if gradient is None else gradient.tolist(),
                      hessian_hartree_bohr2=None if hessian is None else hessian.tolist(),
                      hessian_max_asymmetry=asymmetry, hessian_seconds=hess_seconds,
                      dispersion_energy_hartree=float(mf.scf_summary.get('dispersion', 0)),
                      electrical_response_derivatives='not computed; zero placeholders for External Hessian protocol',
                      derivative_order=req['deriv'],
                      coords_bohr=req['coords_bohr'], numbers=req['numbers'],
                      evaluation=self.counter, scf_converged=True, guess=guess,
                      scf_cycles=max(cycles, default=getattr(mf, 'cycles', 0)),
                      scf_seconds=scf_seconds, gradient_seconds=grad_seconds,
                      total_seconds=time.perf_counter()-start)
        phase = time.perf_counter()
        result['profile'] = timing
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        # Commit state only after the requested computation is successful.
        if req.get('commit_guess', True):
            self.previous = (signature, mol, dm.copy())
            if self.anchor is None:
                self.anchor = self.previous
            np.savez(directory / 'converged_density.npz', dm=dm, coords_bohr=mol.atom_coords())
            (self.root / 'last_successful.json').write_text(json.dumps(dict(evaluation=self.counter,
                directory=str(directory), config=cfg, signature=signature), indent=2))
        mol.stdout.close()
        timing['result_and_checkpoint_io_seconds'] = time.perf_counter()-phase
        timing['evaluate_complete_seconds'] = time.perf_counter()-start
        with (self.root/'profile.jsonl').open('a') as f:
            f.write(json.dumps(dict(evaluation=self.counter,**timing))+'\n')
        return result


def serve(args):
    root = Path(args.root).resolve()
    engine = Engine(json.loads(Path(args.config).read_text()), root)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(args.socket)
        os.chmod(args.socket, 0o600)
        server.listen(4)
        (root / 'worker.ready').write_text(str(os.getpid()))
        running = True
        while running:
            with server.accept()[0] as conn:
                request_start = time.perf_counter()
                req = {}
                try:
                    req = json.loads(conn.makefile('rb').readline())
                    if req.get('operation') == 'stop':
                        reply = dict(ok=True); running = False
                    elif req.get('operation') == 'reset':
                        engine.previous = None
                        engine.anchor = None
                        reply = dict(ok=True, reset=True)
                    elif req.get('operation') == 'ping':
                        reply = dict(ok=True)
                    else:
                        if req.get('operation') == 'external':
                            if req['layer'] != 'R':
                                raise ValueError('Unsupported External layer; isolated molecules only')
                            parsed = read_input(req['input'])
                        else:
                            parsed = req['geometry']
                        result = engine.evaluate(parsed)
                        if req.get('operation') == 'external':
                            text = format_output(result, parsed['deriv'])
                            output = Path(req['output'])
                            temp = output.with_name(output.name + '.tmp')
                            temp.write_text(text); temp.replace(output)
                            (root / 'evaluations' / f'{engine.counter:04d}' / 'gaussian_input.txt').write_text(Path(req['input']).read_text())
                            (root / 'evaluations' / f'{engine.counter:04d}' / 'gaussian_output.txt').write_text(text)
                        reply = dict(ok=True, result=result, summary={k:result[k] for k in
                            ('evaluation', 'energy_hartree', 'guess', 'scf_cycles', 'total_seconds')})
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    reply = dict(ok=False, error=error)
                conn.sendall(json.dumps(reply).encode() + b'\n')
                with (root/'server_profile.jsonl').open('a') as f:
                    f.write(json.dumps(dict(operation=req.get('operation'),evaluation=engine.counter,
                        elapsed_seconds=time.perf_counter()-request_start))+'\n')
    Path(args.socket).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--socket', required=True)
    serve(parser.parse_args())

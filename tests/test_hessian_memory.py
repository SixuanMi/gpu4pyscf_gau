"""CPU checks for opt-in isolation, fail-closed compatibility, and budget validation."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from gpu4pyscf_gau import hessian_memory as hm


SOURCE = '''
def get_avail_mem(exclude_memory_pool=True):
    return 80 * 2**30

def _balance_batch_size(n, size, alignment):
    return n

def _jk_energy_per_atom():
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    aux_batch_size = _balance_batch_size(1024,1024,16)
    aux_blksize = min(512,aux_batch_size)
    batch_size = 256
    blksize = min(512,batch_size)
    return mem_avail,aux_batch_size,aux_blksize,blksize

def _get_jk():
    class DF:
        def get_blksize(self,mem_fraction):return 1024
    dfobj=DF()
    mem_avail=get_avail_mem()
    blksize=dfobj.get_blksize(mem_fraction=0.2)
    n_dm=100
    dm_batch_size=min(50,n_dm)
    return mem_avail,blksize,dm_batch_size

def _get_veff():
    mem_avail=get_avail_mem()
    largest_shell_nao=28
    naux=5000
    batch_size=min(2048,max(largest_shell_nao,naux))
    blksize=min(512,batch_size)
    return mem_avail,batch_size,blksize
'''


class MemoryPolicyTests(unittest.TestCase):
    def test_budget_and_tail_limits(self):
        cfg = hm.validate({'policy':'conservative'})
        self.assertEqual(hm.budget(80*2**30,cfg),48*2**30)
        self.assertEqual(hm.budget(10*2**30,cfg),2*2**30)
        with self.assertRaises(MemoryError):hm.budget(7*2**30,cfg)
        self.assertEqual(hm.cap_block(7,128),7)
        with self.assertRaises(MemoryError):hm.cap_block(0,128)

    def test_invalid_configuration_is_rejected(self):
        for value in [{'policy':'lowmem'},{'aux_batch_cap':3},{'reserve_gib':float('nan')},
                      {'integral_block_cap':True},{'memory_fraction':1},{'typo':1},[]]:
            with self.subTest(value=value), self.assertRaises(ValueError):hm.validate(value)

    def test_off_never_imports_gpu_libraries(self):
        with patch.object(hm.importlib,'import_module',side_effect=AssertionError('GPU import')):
            report = {}
            with hm.hessian_memory_scope(None,report):pass
            self.assertFalse(report['applied'])

    def test_scoped_patch_restores_after_failure_and_rejects_unknown_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = []
            for name in ['rhf','uhf']:
                filename = Path(tmp)/(name+'.py');filename.write_text(SOURCE)
                spec = importlib.util.spec_from_file_location(name,filename)
                module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                modules.append(module)
            digest = hashlib.sha256(SOURCE.encode()).hexdigest()
            gpu = types.ModuleType('gpu4pyscf');lib = types.ModuleType('gpu4pyscf.lib')
            lib.multi_gpu = types.SimpleNamespace(num_devices=1)
            imports = {'gpu4pyscf':gpu,'gpu4pyscf.lib':lib,'rhf':modules[0],'uhf':modules[1]}
            originals = [m._jk_energy_per_atom for m in modules]
            original_response = modules[0]._get_jk
            original_h1 = [m._get_veff for m in modules]
            with patch.dict(sys.modules,imports), patch.object(hm.importlib,'import_module',
                    side_effect=lambda name:modules[0] if name.endswith('.rhf') else modules[1]), \
                    patch.dict(hm.SOURCE_HASHES,{'rhf':digest,'uhf':digest}):
                report = {}
                with self.assertRaisesRegex(RuntimeError,'calculation failed'):
                    with hm.hessian_memory_scope({'policy':'conservative'},report):
                        self.assertEqual(modules[0]._jk_energy_per_atom(),(48*2**30,256,128,128))
                        self.assertEqual(originals[0](),(80*2**30,1024,512,256))
                        self.assertEqual(modules[0]._get_jk(),(48*2**30,256,8))
                        self.assertEqual(modules[0]._get_veff(),(48*2**30,256,128))
                        with self.assertRaisesRegex(RuntimeError,'Concurrent'):
                            with hm.hessian_memory_scope({'policy':'conservative'},{}):pass
                        raise RuntimeError('calculation failed')
                self.assertTrue(report['applied'])
                self.assertEqual([m._jk_energy_per_atom for m in modules], originals)
                self.assertIs(modules[0]._get_jk,original_response)
                self.assertEqual([m._get_veff for m in modules],original_h1)
                with hm.hessian_memory_scope({'policy':'conservative','h1_aux_cap':8},{}):
                    with self.assertRaisesRegex(ValueError,'basis shell'):modules[0]._get_veff()
                with patch.dict(hm.SOURCE_HASHES,{'uhf':'not-supported'}):
                    with self.assertRaisesRegex(RuntimeError,'Unsupported'):
                        with hm.hessian_memory_scope({'policy':'conservative'},{}):pass
                    self.assertEqual([m._jk_energy_per_atom for m in modules], originals)


if __name__ == '__main__':
    unittest.main()

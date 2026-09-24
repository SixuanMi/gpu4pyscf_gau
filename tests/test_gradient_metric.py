"""Scope isolation and compatibility checks; numerical validation runs on the GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from gpu4pyscf_gau import gradient_metric as gm
from gpu4pyscf_gau.config import load_config


class GradientMetricTests(unittest.TestCase):
    def test_accuracy_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{}')
            config = load_config(path)['gpu']
            self.assertEqual(config['df_gradient_metric'], 'original')
            self.assertEqual(config['conv_tol_cpscf'], 1e-10)
            self.assertEqual(config['cphf_grid'], 'scf')
            valid = dict(df_gradient_metric='solve', conv_tol_cpscf=1e-10,
                         cphf_grid='scf')
            path.write_text(json.dumps({'gpu': valid}))
            config = load_config(path)['gpu']
            for key, value in valid.items():
                self.assertEqual(config[key], value)
            invalid = [dict(conv_tol_cpscf=x) for x in
                       [0, -1, True, '1e-10', float('inf'), float('nan')]]
            invalid += [dict(cphf_grid='unknown'), dict(cphf_grid='scf', method='hf'),
                        dict(df_gradient_metric='solve', density_fit=False)]
            for values in invalid:
                with self.subTest(values=values):
                    path.write_text(json.dumps({'gpu': values}))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_original_does_not_import_gpu(self):
        with patch.object(gm.importlib, 'import_module', side_effect=AssertionError('GPU import')):
            report = {}
            with gm.gradient_metric_scope('original', report):
                self.assertFalse(report['applied'])
        for value in [None, True, 'lowmem', {}]:
            with self.assertRaises(ValueError):gm.validate(value)

    def test_restore_and_source_guard(self):
        source = 'def _jk_energy_per_atom():\n' + gm._ORIGINAL + '    return dm_oo\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rhf.py';path.write_text(source)
            spec = importlib.util.spec_from_file_location('gradient_test_rhf', path)
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            original = module._jk_energy_per_atom
            gpu = types.ModuleType('gpu4pyscf');lib = types.ModuleType('gpu4pyscf.lib')
            lib.multi_gpu = types.SimpleNamespace(num_devices=1)
            with patch.dict(sys.modules, {'gpu4pyscf':gpu, 'gpu4pyscf.lib':lib}), \
                    patch.object(gm.importlib, 'import_module', return_value=module), \
                    patch.object(gm, 'SOURCE_HASH', hashlib.sha256(source.encode()).hexdigest()):
                report = {}
                with self.assertRaisesRegex(RuntimeError, 'gradient failed'):
                    with gm.gradient_metric_scope('solve', report):
                        self.assertIsNot(module._jk_energy_per_atom, original)
                        self.assertTrue(report['applied'])
                        with self.assertRaisesRegex(RuntimeError, 'Concurrent'):
                            with gm.gradient_metric_scope('solve', {}):pass
                        raise RuntimeError('gradient failed')
                self.assertIs(module._jk_energy_per_atom, original)
                with patch.object(gm, 'SOURCE_HASH', 'unsupported'):
                    with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                        with gm.gradient_metric_scope('solve', {}):pass
                    self.assertIs(module._jk_energy_per_atom, original)


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from gpu4pyscf_gau.config import load_config
from gpu4pyscf_gau.cli import main
from contextlib import redirect_stdout, redirect_stderr
import io


class YamlConfigTests(unittest.TestCase):
    def test_yaml_scientific_values_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('gpu:\n  conv_tol: 1e-11 # scientific without decimal\n'
                            '  conv_tol_grad: 1e-9\n  conv_tol_cpscf: 1e-10\n'
                            '  cphf_grid: scf\n')
            cfg = load_config(path)['gpu']
            self.assertEqual(cfg['conv_tol'], 1e-11)
            self.assertEqual(cfg['conv_tol_grad'], 1e-9)
            self.assertIsInstance(cfg['conv_tol_cpscf'], float)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(['check-config', '--config', str(path)]), 0)
            self.assertEqual(json.loads(out.getvalue())['gpu']['cphf_grid'], 'scf')

    def test_defaults_and_legacy_overrides_in_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ['yaml', 'json']:
                path = Path(tmp) / ('config.' + suffix)
                path.write_text('{}')
                cfg = load_config(path)['gpu']
                self.assertEqual((cfg['conv_tol_cpscf'], cfg['cphf_grid']), (1e-10, 'scf'))
                for tol in [None, 1e-6]:
                    path.write_text(json.dumps({'gpu': {'conv_tol_cpscf': tol, 'cphf_grid': 'default'}}))
                    cfg = load_config(path)['gpu']
                    self.assertEqual((cfg['conv_tol_cpscf'], cfg['cphf_grid']), (tol, 'default'))
                path.write_text('{"gpu":{"method":"hf"}}')
                self.assertEqual(load_config(path)['gpu']['cphf_grid'], 'default')

    def test_invalid_and_unsafe_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.yml'
            for content in ['gpu: [', '[]', 'gpu:\n  unknown_key: 1\n',
                            'gpu:\n  cphf_grid: bad\n', 'gpu:\n  conv_tol_cpscf: .nan\n',
                            '!!python/object/apply:builtins.print [unsafe]']:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError): load_config(path)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['check-config','--config',str(path)]), 2)

    def test_example_configuration_agrees_with_json(self):
        root = Path(__file__).resolve().parents[1]
        yaml = load_config(root/'examples/config.yaml')
        json_config = load_config(root/'examples/config.json')
        self.assertEqual(yaml, json_config)


if __name__ == '__main__':
    unittest.main()

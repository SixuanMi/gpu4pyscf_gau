import tempfile
from pathlib import Path
import unittest
from gpu4pyscf_gau.protocol import read_input, format_output


class ProtocolTests(unittest.TestCase):
    def test_gaussian_fixed_width_and_gradient_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'input'
            p.write_text('         1         1         0         2\n'
                         '         1      1.000000000000     -2.000000000000      0.000000000000      0.000000000000\n')
            parsed = read_input(p)
            self.assertEqual(parsed['coords_bohr'], [[1., -2., 0.]])
            text = format_output(dict(energy_hartree=-0.5, dipole_au=[0.,0.,0.],
                                      gradient_hartree_bohr=[[0.123,-0.456,0.789]]), 1)
            rows = text.splitlines()
            self.assertEqual([len(r) for r in rows], [80, 60])
            self.assertEqual(float(rows[1][:20].replace('D','E')), 0.123)

    def test_fail_closed_for_unsupported_or_corrupt_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'input'
            for data in ['1 3 0 2\n1 0 0 0 0\n', '2 1 0 1\n1 0 0 0 0\n',
                         '1 1 0 2\n1 nan 0 0 0\n', '1 1 0 2\n1 0 0 0 1\n']:
                p.write_text(data)
                with self.assertRaises(ValueError):
                    read_input(p)

    def test_hessian_packing_and_response_offsets(self):
        result = dict(energy_hartree=-1., dipole_au=[1.,2.,3.],
                      gradient_hartree_bohr=[[4.,5.,6.]],
                      hessian_hartree_bohr2=[[11.,12.,13.],[12.,22.,23.],[13.,23.,33.]])
        lines = format_output(result, 2).splitlines()
        flat = [float(line[i:i+20].replace('D','E')) for line in lines for i in range(0,len(line),20)]
        self.assertEqual(flat[:7], [-1.,1.,2.,3.,4.,5.,6.])
        self.assertEqual(flat[7:22], [0.]*15)
        self.assertEqual(flat[22:], [11.,12.,22.,13.,23.,33.])


if __name__ == '__main__':
    unittest.main()

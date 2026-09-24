import tempfile
from pathlib import Path
import unittest
from gpu4pyscf_gau.fchk import read_fchk


class CheckpointTests(unittest.TestCase):
    def test_scalar_and_wrapped_cartesian_fields(self):
        text='title\nmethod\n'
        text+=f'{"Total Energy":43s}R     -1.234567890123D+02\n'
        text+=f'{"Atomic numbers":43s}I   N=           2\n           6           1\n'
        text+=f'{"Current cartesian coordinates":43s}R   N=           6\n 1.0 2.0 3.0\n 4.0 5.0 6.0\n'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.fchk';p.write_text(text);d=read_fchk(p)
        self.assertEqual(d['Total Energy'],-123.4567890123)
        self.assertEqual(d['Atomic numbers'],[6,1])
        self.assertEqual(d['Current cartesian coordinates'],[1.,2.,3.,4.,5.,6.])


if __name__=='__main__':unittest.main()

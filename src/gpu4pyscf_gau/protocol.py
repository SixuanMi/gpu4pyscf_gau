"""Gaussian 16 External, isolated molecule, derivatives 0/1/2, atomic units."""
import math
from pathlib import Path


def read_input(path):
    lines = Path(path).read_text().splitlines()
    n, deriv, charge, mult = map(int, lines[0].split())
    if n < 1 or mult < 1 or deriv not in (0, 1, 2):
        raise ValueError('Only isolated-molecule derivative requests 0/1/2 are supported')
    if len(lines) < n + 1:
        raise ValueError('Truncated External geometry')
    numbers, coords = [], []
    for line in lines[1:n+1]:
        fields = line.replace('D', 'E').replace('d', 'e').split()
        z = int(fields[0]); xyz = list(map(float, fields[1:4]))
        if not 1 <= z <= 118 or len(xyz) != 3 or not all(map(math.isfinite, xyz)):
            raise ValueError('Invalid atom or coordinates')
        if len(fields) > 4 and abs(float(fields[4])) > 1e-12:
            raise ValueError('MM charges/ONIOM are not supported')
        numbers.append(z); coords.append(xyz)
    return dict(numbers=numbers, coords_bohr=coords, charge=charge, multiplicity=mult, deriv=deriv)


def format_output(result, deriv):
    def row(values):
        fields = []
        for value in values:
            if not math.isfinite(value):
                raise ValueError('Non-finite External result')
            field = f'{value:20.12E}'.replace('E', 'D')
            if len(field) != 20:
                raise ValueError('Value exceeds Gaussian field width')
            fields.append(field)
        return ''.join(fields) + '\n'
    text = row([result['energy_hartree'], *result['dipole_au']])
    if deriv:
        text += ''.join(row(g) for g in result['gradient_hartree_bohr'])
    if deriv == 2:
        n = len(result['gradient_hartree_bohr'])
        h = result['hessian_hartree_bohr2']
        if len(h) != 3*n or any(len(r) != 3*n for r in h):
            raise ValueError('Invalid Cartesian Hessian dimensions')
        # Gaussian permits unavailable electrical response data to be zero.
        # They are placeholders: do not interpret resulting IR/Raman intensities.
        values = [0.] * (6 + 9*n) + [h[i][j] for i in range(3*n) for j in range(i+1)]
        text += ''.join(row(values[i:i+3]) for i in range(0, len(values), 3))
    return text

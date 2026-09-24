"""Small typed reader for scalar/array fields in Gaussian formatted checkpoints."""
import re
from pathlib import Path


def read_fchk(path):
    lines = Path(path).read_text(errors='replace').splitlines()
    data = {}
    i = 2
    while i < len(lines):
        line = lines[i]; i += 1
        m = re.match(r'^(.{40,43}?)\s+([IRCLH])\s+(.*)$', line)
        if not m:
            continue
        name, kind, value = m.group(1).strip(), m.group(2), m.group(3).strip()
        if kind not in ('I','R'):
            continue
        convert = int if kind == 'I' else lambda x:float(x.replace('D','E'))
        if value.startswith('N='):
            count = int(value[2:].strip()); values = []
            while len(values) < count and i < len(lines):
                values.extend(convert(x) for x in lines[i].split()); i += 1
            if len(values) != count:
                raise ValueError(f'Truncated fchk field {name}')
            data[name] = values
        else:
            data[name] = convert(value)
    return data

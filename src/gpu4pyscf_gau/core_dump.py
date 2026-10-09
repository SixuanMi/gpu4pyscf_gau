"""Prevent and remove crash dumps without touching checkpoint evidence."""
from pathlib import Path
import re

def disable_core_dumps():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def cleanup_core_dumps(root):
    """Remove only regular core/core.<pid> dumps inside this reaction tree."""
    root = Path(root).resolve()
    removed = []; total = 0
    for path in root.rglob('core*'):
        if not re.fullmatch(r'core(?:[._-][0-9]+(?:[._-][A-Za-z0-9_-]+)*)?', path.name):
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path.relative_to(root))); total += size
    return dict(files=removed, bytes=total)


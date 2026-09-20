# SPDX-License-Identifier: Apache-2.0
"""Apply the canonical c_ops package binding before serving process imports."""

# Standard
import argparse
import hashlib
from pathlib import Path
import shutil


def canonicalize_backend_alias(source: str) -> str:
    """Bind the package attribute to the same merged module as sys.modules.

    Args:
        source: LMCache package initializer text.
    Returns:
        The narrowly patched initializer; repeated installation is idempotent.
    Raises:
        ValueError: Backend-selection anchor is missing or ambiguous.
    """
    anchor = '    sys.modules["lmcache.c_ops"] = _ops\n'
    if source.count(anchor) != 1:
        raise ValueError('Unexpected c_ops backend binding')
    if '\n    c_ops = _ops\n' in source:
        return source
    return source.replace(anchor, anchor +
        '    # Keep package imports and importlib late operator registration identical.\n'
        '    c_ops = _ops\n', 1)


def main() -> None:
    """Preserve the deployed initializer and patch only its c_ops alias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / '__init__.py'
    original = target.read_text()
    patched = canonicalize_backend_alias(original)
    compile(patched, str(target), 'exec')
    backup = target.with_suffix('.py.before_backend_alias_v995')
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(patched)
    print('BACKEND_ALIAS_V995_FIXED', hashlib.sha256(original.encode()).hexdigest(),
          hashlib.sha256(patched.encode()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()

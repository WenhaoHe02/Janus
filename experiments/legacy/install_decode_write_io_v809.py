# SPDX-License-Identifier: Apache-2.0
"""Install only the bounded-I/O admission fix before worker import."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil

# Local
from install_glm_stable_startup_v764 import definition, replace_function


def main() -> None:
    """Preserve unrelated overlays and backups; reject unknown patch anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.lmcache_root
    changes = {}
    path = root/'v1/write_planner.py'
    content = path.read_text()
    if 'active_io_blocks' not in content:
        for old, new in (
            ('    decode_tokens_mib: float = 0.0\n', '    decode_tokens_mib: float = 0.0\n    active_io_blocks: int = 0\n'),
            ('        self._active_waves = 0\n', '        self._active_waves = 0\n        self._active_io_blocks = 0\n'),
            ('                active_waves=self._active_waves,\n', '                active_waves=self._active_waves,\n                active_io_blocks=self._active_io_blocks,\n')):
            if content.count(old) != 1:
                raise ValueError('write planner anchor changed')
            content = content.replace(old, new)
        reference = Path('/tmp/write_planner_v809.py').read_text()
        for name in ('io_block_started', 'io_block_finished'):
            content = replace_function(content, definition(reference, name, 'TuttiWritePlanManager'), 'TuttiWritePlanManager')
    changes[path] = content
    path = root/'v1/storage_backend/cpu_raw_writer.py'
    content = path.read_text()
    if 'WriteIOScope =' not in content:
        content = content.replace('from typing import TypeVar', 'from contextlib import nullcontext\nfrom typing import ContextManager, TypeVar')
        content = content.replace('WriteAdmissionWaiter = Callable[[int, float], None]',
            'WriteAdmissionWaiter = Callable[[int, float], None]\nWriteIOScope = Callable[[int, float], ContextManager[None]]')
    reference = Path('/tmp/cpu_raw_writer_v809.py').read_text()
    for name in ('__init__', 'write'):
        content = replace_function(content, definition(reference, name, 'CPURawBlockWriter'), 'CPURawBlockWriter')
    changes[path] = content
    path = root/'v1/gpu_connector/tutti_direct_loader.py'
    content = path.read_text()
    if 'from contextlib import contextmanager' not in content:
        content = content.replace('import bisect', 'from contextlib import contextmanager\nfrom collections.abc import Iterator\nimport bisect', 1)
    reference = Path('/tmp/tutti_direct_loader_v809.py').read_text()
    for name in ('set_decode_write_state', 'background_write_io_scope', '_ensure_cpu_raw_writer'):
        content = replace_function(content, definition(reference, name, 'TuttiDirectLoader'), 'TuttiDirectLoader')
    changes[path] = content
    for path, content in changes.items():
        ast.parse(content)
    for path, content in changes.items():
        if path.read_text() == content:
            continue
        backup = path.with_name(path.name+'.before_io_v809')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(content)
    print('DECODE_PHYSICAL_IO_ADMISSION_V809_INSTALLED', flush=True)


if __name__ == '__main__':
    main()

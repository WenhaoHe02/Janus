# SPDX-License-Identifier: Apache-2.0
"""Apply only the mapped-control allocation change to the deployed loader."""

# Standard
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil


def replace_once(source: str, old: str, new: str) -> str:
    """Reject ambiguous deployment anchors before changing any source file."""
    if source.count(old) != 1:
        raise ValueError(f'Unexpected anchor count: {old[:80]!r}')
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    """Preserve the normal path and pair mapped allocations with host frees."""
    source = replace_once(source, 'from lmcache.logging import init_logger\n',
        'from lmcache.logging import init_logger\n'
        'from lmcache.v1.gpu_connector.mapped_control_memory import allocate_mapped_control, free_mapped_control\n')
    source = replace_once(source, 'def _cuda_memcpy_async_h2d(',
        'def _cuda_malloc_control(size: int, mapped: bool) -> int:\n'
        '    return allocate_mapped_control(size, _get_cudart()) if mapped else _cuda_malloc_managed(size)\n\n\n'
        'def _cuda_free_control(ptr: int, mapped: bool) -> None:\n'
        '    if mapped:\n'
        '        free_mapped_control(ptr, _get_cudart())\n'
        '    else:\n'
        '        _cuda_free(ptr)\n\n\n'
        'def _cuda_memcpy_async_h2d(')
    source = replace_once(source, '        pci_bdf: str = "",\n    ) -> None:\n        self._session = session',
        '        pci_bdf: str = "",\n        control_memory_mapped: bool = False,\n'
        '    ) -> None:\n        self._control_memory_mapped = control_memory_mapped\n        self._session = session')
    source = replace_once(source, '        timed_out_ptr = 0\n',
        '        timed_out_ptr = 0\n        control_memory_mapped = os.getenv("LMCACHE_TUTTI_MAPPED_CONTROL", "0") == "1"\n')
    for name, dtype in [('sq_tail','c_uint16'),('cq_head','c_uint16'),('cq_phase','c_uint8'),('timed_out','c_int32')]:
        source = replace_once(source, f'{name}_ptr = _cuda_malloc_managed(ctypes.sizeof(ctypes.{dtype}))',
            f'{name}_ptr = _cuda_malloc_control(ctypes.sizeof(ctypes.{dtype}), control_memory_mapped)')
        source = replace_once(source, f'_cuda_free(self._{name}_ptr)',
            f'_cuda_free_control(self._{name}_ptr, self._control_memory_mapped)')
    source = replace_once(source, '                debug_expected_checksums=debug_expected_checksums,\n                pci_bdf=pci_bdf,\n',
        '                debug_expected_checksums=debug_expected_checksums,\n                pci_bdf=pci_bdf,\n                control_memory_mapped=control_memory_mapped,\n')
    source = replace_once(source,
        '            for ptr in (sq_tail_ptr, cq_head_ptr, cq_phase_ptr, timed_out_ptr):\n                _cuda_free(ptr)',
        '            for ptr in (sq_tail_ptr, cq_head_ptr, cq_phase_ptr, timed_out_ptr):\n                _cuda_free_control(ptr, control_memory_mapped)')
    compile(source, '<mapped-control-loader>', 'exec')
    return source


def main() -> None:
    """Back up the exact deployment and support a read-only dry run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    root = Path(importlib.util.find_spec('lmcache').origin).parent/'v1/gpu_connector'
    path = root/'tutti_direct_loader.py'
    original = path.read_text()
    updated = transform(original)
    if not args.dry_run:
        with Path(str(path)+'.before_v867').open('x') as backup:
            backup.write(original)
        shutil.copyfile('/tmp/mapped_control_memory_v866.py', root/'mapped_control_memory.py')
        path.write_text(updated)
    print('MAPPED_CONTROL_V867',dict(dry_run=args.dry_run,
        before=hashlib.sha256(original.encode()).hexdigest(),after=hashlib.sha256(updated.encode()).hexdigest()))


if __name__ == '__main__':
    main()

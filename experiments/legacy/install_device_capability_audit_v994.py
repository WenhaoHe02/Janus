# SPDX-License-Identifier: Apache-2.0
"""Log metadata-only reasons for counted-read eligibility, preserving behavior."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil

# Local
from install_glm_stable_startup_v764 import definition, replace_function


def build_source(source: str) -> str:
    """Preserve the capability predicate and append bounded diagnostic logging.

    Args:
        source: Deployed loader source before model processes are started.
    Returns:
        Source whose public capability query returns exactly the original value.
    Raises:
        ValueError: The predicate structure differs from the reviewed contract.
    """
    owner = 'TuttiDirectLoader'
    method = definition(source, 'supports_device_counted_read', owner)
    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(method.body[-1], ast.Return):
        raise ValueError('Unexpected counted-read capability predicate')
    expression = method.body.pop().value
    method.body.append(ast.Assign(targets=[ast.Name(id='audit_supported', ctx=ast.Store())], value=expression))
    method.body.extend(ast.parse('''
if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_CAPABILITY_AUDIT') == '1':
    import json
    report = {
        'supported': audit_supported,
        'closed': bool(self._closed),
        'ops_module': getattr(_c_ops, '__name__', None),
        'has_counted_op': callable(getattr(_c_ops, 'tutti_read_indexed_device_count', None)),
        'has_segmented_op': callable(getattr(_c_ops, 'tutti_read_indexed_segments_device_count', None)),
        'io_stream_ready': self._io_stream is not None,
        'iova_table_ready': self._staging_page_iovas_gpu is not None,
        'capacity': int(capacity),
        'queue_depth': int(self._q_depth()),
        'io_nbytes': int(io_nbytes),
        'gpu_page_bytes': int(_GPU_PAGE_SIZE),
        'nvme_lbs': int(_NVME_LBS),
        'device_max_data_size': int(self._session.info.max_data_size),
        'staging_bytes': int(self._staging_capacity_bytes()),
        'segmented': bool(segmented),
    }
    seen = getattr(self, '_counted_capability_audit_seen', None)
    if seen is None:
        seen = self._counted_capability_audit_seen = set()
    key = tuple(report.items())
    if key not in seen and len(seen) < 32:
        seen.add(key)
        logger.info('TUTTI_COUNTED_CAPABILITY_AUDIT %s', json.dumps(report, sort_keys=True))
return audit_supported
''').body)
    return replace_function(source, method, owner)


def main() -> None:
    """Install the explicit diagnostic after preserving the exact loader source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/gpu_connector/tutti_direct_loader.py'
    original = target.read_text()
    patched = build_source(original)
    compile(patched, str(target), 'exec')
    backup = target.with_suffix('.py.before_capability_audit_v994')
    if backup.exists():
        raise RuntimeError('Capability diagnostic backup already exists')
    shutil.copy2(target, backup)
    target.write_text(patched)
    print('COUNTED_CAPABILITY_AUDIT_V994_INSTALLED_DIAGNOSTIC_ONLY', flush=True)


if __name__ == '__main__':
    main()

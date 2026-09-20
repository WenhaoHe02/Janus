# SPDX-License-Identifier: Apache-2.0
"""Respect published logical prefix lengths when composing suffix generations."""

# Standard
import argparse
from pathlib import Path


def corrected_source(source: str) -> str:
    """Replace only the verified physical-base lookup, preserving all other code."""
    if 'LOGICAL_BASE_PREFIX_V916' in source:
        return source
    old = '''                object_id = self._resolve_streaming_object_id(
                    base_prefix_key,
                    object_role,
                    int(layer_id),
                )
                record = (
                    self.kv_object_metadata_store.get(object_id)
                    if object_id is not None
                    else None
                )'''
    new = '''                # LOGICAL_BASE_PREFIX_V916: use the logical prefix alias,
                # not the full immutable physical sidecar it references.
                records = self.get_kv_object_records(
                    [base_prefix_key], roles=[object_role], layer_ids=[int(layer_id)]
                )
                record = records[0] if records else None'''
    if source.count(old) != 1:
        raise ValueError('Logical-base deployment anchor changed')
    result = source.replace(old, new, 1)
    compile(result, '<logical-base-v916>', 'exec')
    return result


def install(root: Path) -> None:
    """Install before workers import the package, retaining the prior file."""
    path = root / 'v1/storage_backend/local_disk_backend.py'
    source = path.read_text()
    result = corrected_source(source)
    if source != result:
        backup = path.with_suffix(path.suffix + '.before_v916')
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(result)
    print('LOGICAL_BASE_PREFIX_V916_INSTALLED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    install(parser.parse_args().lmcache_root)

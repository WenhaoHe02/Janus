# SPDX-License-Identifier: Apache-2.0
"""Opt in to per-command indexed NVMe submission before workers import loader."""
# Standard
import argparse
import hashlib
import json
from pathlib import Path


def transform(source: str) -> str:
    """Install isolated native symbols and one-time actual-call evidence.

    Default behavior is unchanged. No CPU/GPU write packing, selection, read
    byte range, completion polling or residency code is modified.
    Raises ValueError if loader anchors differ or the patch is already present.
    """
    anchor = '\n\nclass TuttiDirectLoader:'
    if source.count(anchor) != 1 or 'UNPACKED_SUBMIT_V1139' in source:
        raise ValueError('Unsupported loader source')
    hook = '''
# UNPACKED_SUBMIT_V1139: only replace fixed/segmented counted-read SQ submission.
if os.getenv("LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139", "0") == "1":
    import importlib.util as _unpacked_util
    import sys as _unpacked_sys
    if os.getenv("LMCACHE_TUTTI_COUNTED_HOLD_GIL", "0") == "1":
        raise RuntimeError("Unpacked ablation requires the baseline GIL policy")
    _unpacked_name = "tutti_unpacked_submit_v1139"
    _unpacked_spec = _unpacked_util.spec_from_file_location(
        _unpacked_name, "/tmp/unpacked_submit_v1139/build/tutti_unpacked_submit_v1139.so")
    _unpacked_module = _unpacked_util.module_from_spec(_unpacked_spec)
    _unpacked_sys.modules[_unpacked_name] = _unpacked_module
    _unpacked_spec.loader.exec_module(_unpacked_module)
    _c_ops.tutti_read_indexed_device_count = _unpacked_module.tutti_read_indexed_device_count
    _c_ops.tutti_read_indexed_segments_device_count = _unpacked_module.tutti_read_indexed_segments_device_count
'''
    result = source.replace(anchor, '\n' + hook + anchor)
    before = '''                    read_op = (
                        _c_ops.tutti_read_indexed_device_count
                        if segment_plan is None
                        else _c_ops.tutti_read_indexed_segments_device_count
                    )'''
    after = before + '''
                    if (os.getenv("LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139", "0") == "1"
                            and not getattr(self, "_unpacked_submit_reported_v1139", False)):
                        if getattr(read_op, "__module__", None) != "tutti_unpacked_submit_v1139":
                            raise RuntimeError("Unpacked binding was overwritten after startup")
                        logger.info("UNPACKED_SUBMIT_V1139_ACTIVE device=%s op=%s capacity=%d",
                                    self._cuda_device, read_op.__name__, selected_ids.numel())
                        self._unpacked_submit_reported_v1139 = True
'''
    if result.count(before) != 1:
        raise ValueError('Counted read lookup changed')
    result = result.replace(before, after)
    compile(result, 'loader_unpacked', 'exec')
    return result


def main() -> None:
    """Back up the deployed loader and apply the strict experiment opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    path = args.lmcache_root / 'v1/gpu_connector/tutti_direct_loader.py'
    source = path.read_text()
    target = transform(source)
    backup = path.with_suffix('.py.before_unpacked_v1139')
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_bytes(path.read_bytes())
    path.write_text(target)
    print(json.dumps(dict(patch='UNPACKED_SUBMIT_V1139',
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        installed_sha256=hashlib.sha256(target.encode()).hexdigest())))


if __name__ == '__main__':
    main()

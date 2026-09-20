# SPDX-License-Identifier: Apache-2.0
"""Apply the logical-prefix composition fix, then retain v911 serving controls."""

# Standard
import importlib.util
import os
from pathlib import Path
import runpy

# Local
from install_logical_base_v916 import install


def main() -> None:
    """Install before imports; no algorithm or profiling configuration change."""
    spec = importlib.util.find_spec('lmcache')
    if spec is None or spec.origin is None:
        raise RuntimeError('LMCache is unavailable')
    install(Path(spec.origin).parent)
    if os.getenv("LMCACHE_STORAGE_MODE") == "ssd":
        from install_file_tail_admission_v958 import install as install_file_tail

        install_file_tail(Path(spec.origin).parent, Path("/tmp/cache_engine_source_v958.py"),
                          Path("/tmp/file_prefix_layout.py"))
    runpy.run_path('/tmp/exec_ds_serving_v911.py', run_name='__main__')


if __name__ == '__main__':
    main()

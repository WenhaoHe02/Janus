# SPDX-License-Identifier: Apache-2.0
"""Apply validated serving admission fixes without changing K-shard semantics."""

# Standard
import importlib.util
from pathlib import Path
import runpy

# Local
from install_serving_admission_v911 import install


def main() -> None:
    """Install after deployed overlays, then retain the current serving setup."""
    spec = importlib.util.find_spec('lmcache')
    if spec is None or spec.origin is None:
        raise RuntimeError('LMCache is unavailable')
    install(Path(spec.origin).parent, Path('/tmp/cache_engine_source_v911.py'))
    runpy.run_path('/tmp/exec_ds_serving_v890.py', run_name='__main__')


if __name__ == '__main__':
    main()

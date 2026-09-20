# SPDX-License-Identifier: Apache-2.0
"""Install serving code once after legacy overlays and before server import."""

# Standard
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

# Local
from serving_glm_configuration_v766 import environment


def main() -> None:
    """Preserve server argv, apply explicit flags, and fail before model loading."""
    if os.getenv('LMCACHE_SERVING_EXPERIMENT') != '1':
        raise RuntimeError('explicit serving startup not selected')
    flags = environment(os.environ['LMCACHE_SERVING_ARM'])
    os.environ.update(flags)
    spec = importlib.util.find_spec('lmcache')
    if spec is None or spec.origin is None or len(sys.argv) < 2:
        raise RuntimeError('missing package or original server command')
    root = str(Path(spec.origin).parent)
    for script, arguments in (
        ('install_glm_stable_startup_v764.py', ['--originals', '/tmp', '--helper', '/tmp/serving_counted_ops_v764.py']),
        ('install_decode_write_io_v809.py', []),
    ):
        subprocess.run([sys.executable, '/tmp/' + script, '--lmcache-root', root, *arguments], check=True)
    print('GLM_SERVING_V766_FINAL_FLAGS', flags, flush=True)
    diagnostic_session = os.getenv('LMCACHE_SERVING_DIAGNOSTIC_NSYS')
    if diagnostic_session:
        profiler = '/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
        print('DIAGNOSTIC_ONLY_NSYS_SESSION', diagnostic_session, flush=True)
        os.execvp(profiler, [profiler, 'launch', '--session-new', diagnostic_session,
                  '--trace=cuda,nvtx,osrt', *sys.argv[1:]])
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
"""Require all TP storage replicas to participate in scheduler cache lookup."""

# Standard
import os
from pathlib import Path
import runpy

# Third Party
import yaml


def main() -> None:
    """Use existing minimum-hit lookup semantics instead of rank-zero readiness.

    Only cache admission changes: no waits are added to the client replay,
    and unavailable tail tokens are left for normal model recomputation.
    """
    path = Path(os.environ['LMCACHE_CONFIG_FILE'])
    config = yaml.safe_load(path.read_text())
    config['lookup_server_worker_ids'] = list(range(8))
    path.write_text(yaml.safe_dump(config))
    os.environ['LMCACHE_LOOKUP_SERVER_WORKER_IDS'] = '0,1,2,3,4,5,6,7'
    print('LOOKUP_V894_ALL_RANKS_MIN_READY_PREFIX', config['lookup_server_worker_ids'], flush=True)
    runpy.run_path('/tmp/exec_glm_serving_v867.py', run_name='__main__')


if __name__ == '__main__':
    main()

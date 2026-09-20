# SPDX-License-Identifier: Apache-2.0
"""Backend-aware handoff: DRAM has synchronous puts, not a Tutti write queue."""

# Standard
import json
import subprocess


def needs_tutti_drain(arms: list[str], config: dict) -> bool:
    """Validate eight consistent ranks and distinguish verified CPU-only storage.

    Return False only for a proven DRAM arm; unknown/mixed states raise.
    """
    if len(arms) != 8 or len(set(arms)) != 1:
        raise ValueError('Expected eight consistent serving arms')
    if arms[0] == 'dram':
        if not config.get('local_cpu') or config.get('local_disk') or config.get('store_location') != 'LocalCPUBackend':
            raise ValueError('DRAM handoff requires actual CPU-only configuration')
        return False
    if arms[0] not in ('k_local', 'no_prefetch'):
        raise ValueError('Unsupported handoff backend; verify its publication separately')
    return True


def drain_current_backend() -> list[dict]:
    """Drain Tutti, or verify synchronous CPU-only storage with no queue to drain."""
    from run_original_serving_v957 import get_json
    from run_write_cap_cell_v1143 import drain

    arms = [get_json(f'http://127.0.0.1:{7000+r}/env').get('LMCACHE_SERVING_ARM', '') for r in range(8)]
    command = ['docker', 'exec', 'dsv4-csa-cp8-off', 'python3', '-c',
               "import json,yaml; print(json.dumps(yaml.safe_load(open('/tmp/lmcache_ssd_tutti_kvobj.yaml'))))"]
    config = json.loads(subprocess.check_output(command))
    if needs_tutti_drain(arms, config):
        return drain()
    return [dict(backend='LocalCPUBackend', rank=r, no_tutti_writer=True) for r in range(8)]

# SPDX-License-Identifier: Apache-2.0
"""Run both real NP writer modes with decode/write overlap explicitly allowed."""

# Standard
import json
import os
from pathlib import Path
import subprocess
import sys

# Local
import run_np_write_modes_v1221 as modes


def verify_ready() -> None:
    """Require released host guards and no conflicting model on first launch."""
    if Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '20':
        raise RuntimeError('Previous host guard has not restored 20')
    if modes.PREVIOUS.name == 'gpu_nvme_write':
        raise RuntimeError('Unexpected predecessor')
    if modes.PREVIOUS.parent == modes.ROOT:
        if not (modes.PREVIOUS / 'COMPLETE.json').exists():
            raise RuntimeError('Previous mode incomplete')
        if not json.loads((modes.PREVIOUS / 'watchdog.json').read_text())['disarmed']:
            raise RuntimeError('Previous mode guard still active')
        modes.assert_idle()
        return
    prior = Path(os.getenv('NP_FAILED_PREDECESSOR', '/tmp/flash_np_write_modes_v1221/gpu_pack_cpu_write'))
    if not json.loads((prior / 'watchdog.json').read_text())['disarmed']:
        raise RuntimeError('Failed startup guard still active')
    running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True)
    if any(name.startswith('dsv4-csa-cp8-') for name in running.splitlines()):
        raise RuntimeError('Another model container is active')


def main() -> None:
    """Launch isolated results; keep old startup failures and measurements intact."""
    modes.ROOT = Path(os.getenv('NP_CONCURRENT_RESULT_ROOT', '/tmp/flash_np_concurrent_writes_v1223'))
    modes.verify_ready = verify_ready
    if '--worker' in sys.argv:
        try:
            modes.worker()
        except BaseException as error:
            (modes.ROOT / 'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
            raise
        return
    verify_ready()
    modes.ROOT.mkdir(exist_ok=False)
    for mode in modes.MODES:
        modes.prepare_mode(mode)
    (modes.ROOT / 'plan.json').write_text(json.dumps(dict(
        modes=modes.MODES, datasets=['internal9', 'terminalbench12'], trials=1,
        priority='TPOT', output='original full O', arrival='serial',
        decode_write_pause=False, unlimited_decode_writes=True,
        wait_for_ssd_before_decode=False, drain_between_requests=True,
        synthetic_write_workload=False, actual8rank_execution_required=True,
        old_paused_write_baselines_not_matched=True), indent=2))
    with (modes.ROOT / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, root=str(modes.ROOT))
    (modes.ROOT / 'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

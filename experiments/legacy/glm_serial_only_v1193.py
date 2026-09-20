# SPDX-License-Identifier: Apache-2.0
"""Cancel unstarted periodic cells and safely finish the remaining serial cell."""

# Standard
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT = Path('/tmp/glm_layerwise_fixed_v1191')
POLICY = ROOT / 'SERIAL_ONLY_V1193.json'


def cancel_periodic(arguments: list[str]) -> None:
    """Reject only this queue's periodic GLM clients before preparation or I/O.

    Args:
        arguments: Client command-line arguments without the executable.
    Raises:
        SystemExit: Code75 for an explicitly cancelled periodic cell.
        ValueError: A cancellation attempt is not at the expected first boundary.
    """
    if not POLICY.exists() or '--serial' in arguments:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path)
    args, _ = parser.parse_known_args(arguments)
    if args.model != 'glm' or args.output is None or not args.output.is_relative_to(ROOT):
        return
    if args.output != ROOT/'terminalbench_2.1/0p01/cell/formal':
        raise ValueError('Unexpected periodic boundary; inspect before proceeding')
    (ROOT/'PERIODIC_CANCELLED_V1193.json').write_text(json.dumps(dict(
        output=str(args.output), reason='User requested serial only',
        before_prepare=True, before_requests=True)))
    raise SystemExit(75)


def worker() -> None:
    """Wait for current serial work, validate guard release, then run SWE serial."""
    # Local
    from run_layerwise_public_by_model_v1174 import reuse_container
    from run_layerwise_three_rates_v1180 import check_complete

    process = Path('/proc/43261/cmdline')
    if process.exists() and process.read_bytes():
        if b'run_glm_layerwise_fixed_v1191.py' not in process.read_bytes():
            raise RuntimeError('Predecessor PID changed')
        try:
            descriptor = os.pidfd_open(43261)
        except ProcessLookupError:
            pass
        else:
            try:
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    if not (ROOT/'PERIODIC_CANCELLED_V1193.json').exists():
        raise RuntimeError('Predecessor stopped before authorized cancellation')
    for dataset, count in [('internal', 13), ('terminalbench_2.1', 12)]:
        check_complete(ROOT/dataset/'serial', count, None)
    cancelled = ROOT/'terminalbench_2.1/0p01'
    if not json.loads((cancelled/'watchdog.json').read_text()).get('disarmed'):
        raise RuntimeError('Cancelled scope guard remains active')
    if Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '20':
        raise RuntimeError('Cancelled scope not restored')
    (ROOT/'FAILED.json').rename(ROOT/'SUPERSEDED_PERIODIC_V1193.json')
    old_plan = ROOT/'plan.json'
    old_plan.rename(ROOT/'plan_before_serial_only_v1193.json')
    plan = json.loads((ROOT/'plan_before_serial_only_v1193.json').read_text())
    for dataset in plan['datasets']:
        dataset['rates'] = [None]
    plan['user_scope'] = 'serial only; periodic cells cancelled before measurement'
    old_plan.write_text(json.dumps(plan, indent=2))
    dataset = dict(name='swe_bench_pro', count=7,
                   inputs='/tmp/layerwise_public_by_model_v1174/native/glm/swe_bench_pro')
    (ROOT/'status.json').write_text(json.dumps(dict(dataset=dataset['name'], rate=None,
        phase='running', coordinator='glm_serial_only_v1193', serial_only=True)))
    stage = ROOT/'swe_bench_pro/serial'
    reuse_container('glm', dataset, stage, None, None,
                    client_script='/tmp/run_glm_layerwise_client_v1191.py')
    check_complete(stage, 7, None)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(cells=3, formal_requests=32,
        model='glm', rates=[None], periodic_cancelled=True)))
    (ROOT/'status.json').write_text(json.dumps(dict(phase='complete', serial_only=True)))


def main() -> None:
    """Arm one boundary handoff without stopping the active measurement client."""
    if '--worker' in sys.argv:
        try:
            worker()
        except BaseException as error:
            (ROOT/'SERIAL_SUCCESSOR_FAILED_V1193.json').write_text(json.dumps(dict(error=repr(error))))
            raise
        return
    if POLICY.exists() or (ROOT/'terminalbench_2.1/0p01').exists():
        raise RuntimeError('Already armed or periodic cell already started')
    if b'run_glm_layerwise_fixed_v1191.py' not in Path('/proc/43261/cmdline').read_bytes():
        raise RuntimeError('Expected predecessor is not running')
    with POLICY.open('x') as stream:
        json.dump(dict(rates=[None], requested_by_user=True), stream)
    with (ROOT/'serial_successor_v1193.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, waits_for=43261, rates=[None], formal_requests=32)
    (ROOT/'serial_successor_v1193.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

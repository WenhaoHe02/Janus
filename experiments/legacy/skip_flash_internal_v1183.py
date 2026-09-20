# SPDX-License-Identifier: Apache-2.0
"""Skip user-cancelled Flash internal cells at a safe model-launch boundary."""

# Standard
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT = Path('/tmp/layerwise_three_rates_v1180')
CONTROL = ROOT / 'skip_flash_internal_v1183.json'
TARGET = ROOT / 'flash/internal/0p01'


def cancel_requested_cell(model: str, output: Path) -> None:
    """Exit only for the exact cancelled cell before any runtime mutation."""
    if model != 'flash' or output != TARGET or not CONTROL.exists():
        return
    policy = json.loads(CONTROL.read_text())
    if policy != {'skip_model': 'flash', 'skip_dataset': 'internal'}:
        raise ValueError('Invalid cancellation policy')
    if output.exists():
        raise RuntimeError('Cancelled cell already has output; inspect before continuing')
    (ROOT / 'FLASH_INTERNAL_CANCELLED_V1183.json').write_text(json.dumps({
        'output': str(output), 'reason': 'User says already measured; do not repeat',
        'before_container_launch': True, 'before_host_guard': True,
    }))
    raise SystemExit(75)


def continue_queue() -> None:
    """Wait for the original coordinator, then run only remaining authorized cells."""
    # Local
    import run_layerwise_public_by_model_v1174 as runner
    from run_layerwise_three_rates_v1180 import check_complete, PREPARED

    process = Path('/proc/56411/cmdline')
    if process.exists() and process.read_bytes():
        if b'run_layerwise_three_rates_v1180.py' not in process.read_bytes():
            raise RuntimeError('Predecessor PID changed')
        try:
            descriptor = os.pidfd_open(56411)
        except ProcessLookupError:
            pass
        else:
            try:
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    if not (ROOT / 'FLASH_INTERNAL_CANCELLED_V1183.json').exists():
        raise RuntimeError('Original queue stopped before the authorized skip boundary')
    for dataset, count in [('internal', 13), ('terminalbench_2.1', 12), ('swe_bench_pro', 7)]:
        for slot, rate in [('serial', None), ('0p01', .01), ('0p015', .015)]:
            if dataset == 'internal' and slot == 'serial':
                continue
            check_complete(ROOT / 'pro' / dataset / slot, count, rate)
    if Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '20':
        raise RuntimeError('Prior host guard not restored')
    failure = ROOT / 'FAILED.json'
    failure.rename(ROOT / 'SUPERSEDED_AT_USER_SKIP_V1183.json')
    runner.ROOT = ROOT
    plan = json.loads(PREPARED.read_text())
    for group in plan:
        model = group['model']
        if model == 'pro':
            continue
        datasets = group['datasets']
        if model == 'glm':
            prior = Path('/tmp/layerwise_internal13_v1161/glm')
            check_complete(prior, 13, None)
            (ROOT / 'glm_internal_serial_reused_v1184.json').write_text(json.dumps({
                'source': str(prior), 'rerun': False, 'other_rates_cancelled': [.01, .015],
            }))
        new_model = True
        for dataset in datasets:
            name = dataset['name']
            rates = (.01, .015) if name == 'internal' else (None, .01, .015)
            warm_source = None
            for rate in rates:
                slot = 'serial' if rate is None else str(rate).replace('.', 'p')
                stage = ROOT / model / name / slot
                (ROOT / 'status.json').write_text(json.dumps(dict(
                    model=model, dataset=name, rate=rate, phase='running',
                    arrival_mode='serial' if rate is None else 'periodic',
                    coordinator='skip_flash_internal_v1183', flash_internal_skipped=True)))
                if new_model:
                    command = [sys.executable, '-u', '/tmp/launch_corrective_only_v1144.py',
                               '--model', model, '--inputs', dataset['inputs'],
                               '--output', str(stage), '--rate', str(rate or .005), '--execute']
                    if rate is None:
                        command.append('--serial')
                    if model == 'glm':
                        command += ['--kv-cache-gib', '20']
                    stage.parent.mkdir(parents=True, exist_ok=True)
                    with (stage.parent / 'model_launch_v1183.log').open('x') as log:
                        subprocess.run(command, stdout=log, stderr=log, check=True)
                    new_model = False
                else:
                    runner.reuse_container(model, dataset, stage, rate, warm_source)
                check_complete(stage, dataset['count'], rate)
                warm_source = stage / 'cell/formal/prepared.json'
    (ROOT / 'COMPLETE.json').write_text(json.dumps(dict(
        new_cells=20, new_request_records=197, flash_internal_skipped=True,
        glm_internal_only_existing_serial=True,
        internal_serial_reused=True, rates=[None, .01, .015])))


def main() -> None:
    """Arm one boundary cancellation and one waiting successor; never duplicate."""
    if '--worker' in sys.argv:
        try:
            continue_queue()
        except BaseException as error:
            (ROOT / 'SUCCESSOR_FAILED_V1183.json').write_text(json.dumps({'error': repr(error)}))
            raise
        return
    if CONTROL.exists() or TARGET.exists():
        raise RuntimeError('Policy already installed or cancelled cell started')
    if b'run_layerwise_three_rates_v1180.py' not in Path('/proc/56411/cmdline').read_bytes():
        raise RuntimeError('Original coordinator not running')
    with CONTROL.open('x') as stream:
        json.dump({'skip_model': 'flash', 'skip_dataset': 'internal'}, stream)
    with (ROOT / 'successor_v1183.log').open('x') as log:
        worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                  start_new_session=True)
    receipt = dict(pid=worker.pid, waits_for=56411, skipped='flash/internal',
                   remaining_order=['Pro finishes unchanged', 'Flash public only', 'GLM unchanged'])
    (ROOT / 'successor_v1183.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

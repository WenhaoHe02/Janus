# SPDX-License-Identifier: Apache-2.0
"""Complete serial/0.01/0.015 per dataset before switching models."""

# Standard
import json
from pathlib import Path
import subprocess
import sys

# Local
import run_layerwise_public_by_model_v1174 as runner


ROOT = Path('/tmp/layerwise_three_rates_v1180')
PREPARED = Path('/tmp/layerwise_public_by_model_v1174/prepared_plan.json')
SERIAL_INTERNAL = {
    'pro': Path('/tmp/layerwise_capacity12_v1173/pro'),
    'flash': Path('/tmp/layerwise_remaining_v1169/flash'),
    'glm': Path('/tmp/layerwise_internal13_v1161/glm'),
}


def check_complete(stage: Path, count: int, rate: float | None) -> None:
    """Require all requests and correct arrival metadata plus released host scope."""
    result = json.loads((stage/'COMPLETE.json').read_text())
    if result['successes'] != count or result['failures'] or result['actual_arrival_rate'] != rate:
        raise RuntimeError(f'Incomplete or mismatched cell: {stage}')
    if not json.loads((stage/'watchdog.json').read_text()).get('disarmed'):
        raise RuntimeError(f'Host guard remains armed: {stage}')


def worker() -> None:
    """Reuse completed internal serial cells; warm each new container once per input."""
    try:
        runner.ROOT = ROOT
        runner.wait_for_pro_internal()
        plan = json.loads(PREPARED.read_text())
        for group in plan:
            model = group['model']
            internal = dict(name='internal', inputs=f'/tmp/internal_union13_v1152/{model}', count=13)
            check_complete(SERIAL_INTERNAL[model], 13, None)
            (ROOT/f'{model}_serial_reused.json').write_text(json.dumps({
                'source': str(SERIAL_INTERNAL[model]), 'rerun': False,
            }, indent=2))
            for dataset in [internal] + group['datasets']:
                name = dataset['name']
                rates = (.01, .015) if name=='internal' else (None, .01, .015)
                warm_source = (SERIAL_INTERNAL[model]/'cell/formal/prepared.json'
                               if model=='pro' and name=='internal' else None)
                for rate in rates:
                    slot = 'serial' if rate is None else str(rate).replace('.', 'p')
                    stage = ROOT/model/name/slot
                    (ROOT/'status.json').write_text(json.dumps(dict(
                        model=model, dataset=name, rate=rate,
                        arrival_mode='serial' if rate is None else 'periodic', phase='running')))
                    new_model = model != 'pro' and name=='internal' and rate==.01
                    if new_model:
                        command = [sys.executable, '-u', '/tmp/launch_corrective_only_v1144.py',
                                   '--model', model, '--inputs', dataset['inputs'], '--output', str(stage),
                                   '--rate', str(rate), '--execute']
                        if model=='glm':
                            command += ['--kv-cache-gib', '20']
                        stage.parent.mkdir(parents=True, exist_ok=True)
                        with (stage.parent/'model_launch.log').open('x') as log:
                            subprocess.run(command, stdout=log, stderr=log, check=True)
                    else:
                        runner.reuse_container(model, dataset, stage, rate, warm_source)
                    check_complete(stage, dataset['count'], rate)
                    warm_source = stage/'cell/formal/prepared.json'
        (ROOT/'COMPLETE.json').write_text(json.dumps(dict(
            models=3, datasets_per_model=3, rates=[None, .01, .015],
            internal_serial_reused=True, new_cells=24, new_request_records=249)))
    except BaseException as error:
        (ROOT/'FAILED.json').write_text(json.dumps({'error': repr(error)}))
        raise


def main() -> None:
    """Launch one revised queue after the obsolete idle waiter was cancelled."""
    if '--worker' in sys.argv:
        worker()
        return
    old = Path('/tmp/layerwise_public_by_model_v1174')
    if not (old/'SUPERSEDED.json').exists():
        raise RuntimeError('Previous serial-only queue has not been cancelled')
    proc = Path('/proc/49614/cmdline')
    if proc.exists() and proc.read_bytes():
        raise RuntimeError('Old successor still live')
    ROOT.mkdir(exist_ok=False)
    (ROOT/'prepared_plan.json').write_bytes(PREPARED.read_bytes())
    with (ROOT/'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, root=str(ROOT))
    (ROOT/'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

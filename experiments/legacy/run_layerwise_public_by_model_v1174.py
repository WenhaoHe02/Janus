# SPDX-License-Identifier: Apache-2.0
"""Finish both public datasets per model, reusing the current Pro container."""

# Standard
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

# Local
from compaction_ab_v1030 import restoration_watchdog, restore_setting
from drain_serving_backend_v1157 import drain_current_backend
from diagnose_serving_decode_v973 import assert_idle
from prepare_internal_union_v1152 import select_manifest


ROOT = Path('/tmp/layerwise_public_by_model_v1174')
PLAN = Path('/tmp/layerwise_public_plan_v1174.json')
PRIOR = Path('/tmp/layerwise_capacity12_v1173/pro')


def prepare_inputs() -> list[dict]:
    """Freeze existing model-native public cohorts; reject hash/identity changes."""
    plan = json.loads(PLAN.read_text())
    for group in plan:
        for cell in group['datasets']:
            source = Path(cell['source'])
            manifest = json.loads((source/'manifest.json').read_text())
            identity = str(manifest.get('model_kind', '')) + str(manifest.get('model_display_name', ''))
            if group['model'] not in identity.lower():
                raise ValueError(f'Wrong native model: {source}')
            with (source/'tokens.i32le').open('rb') as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != manifest['token_file_sha256']:
                raise ValueError('Native token checksum mismatch')
            selected = select_manifest(manifest, cell['ids'])
            target = ROOT/'native'/group['model']/cell['name']
            target.mkdir(parents=True, exist_ok=False)
            (target/'manifest.json').write_text(json.dumps(selected, indent=2))
            shutil.copyfile(source/'tokens.i32le', target/'tokens.i32le')
            cell.update(inputs=str(target), count=len(cell['ids']), token_sha256=digest)
    (ROOT/'prepared_plan.json').write_text(json.dumps(plan, indent=2))
    return plan


def wait_for_pro_internal() -> None:
    """Wait for the exact current worker and require complete restored results."""
    proc = Path('/proc/21694/cmdline')
    if proc.exists() and proc.read_bytes():
        if b'launch_corrective_only_v1144.py' not in proc.read_bytes():
            raise RuntimeError('Predecessor PID identity changed')
        try:
            descriptor = os.pidfd_open(21694)
        except ProcessLookupError:
            pass
        else:
            try:
                (ROOT/'status.json').write_text(json.dumps({'phase': 'waiting_for_pro_internal_21694'}))
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    if not (PRIOR/'COMPLETE.json').exists() or (PRIOR/'FAILED.json').exists():
        raise RuntimeError('Pro internal did not complete; no public launch')
    if not json.loads((PRIOR/'watchdog.json').read_text()).get('disarmed'):
        raise RuntimeError('Previous host guard not released')


def reuse_container(model: str, cell: dict, stage: Path,
                    rate: float | None = None, reuse_shapes: Path | None = None,
                    client_script: str = '/tmp/run_corrective_only_cell_v1144.py') -> None:
    """Measure one dataset in the existing verified model container with a host guard."""
    assert_idle()
    drain_current_backend()
    setting = Path('/proc/sys/vm/compaction_proactiveness')
    if setting.read_text().strip() != '20':
        raise RuntimeError('Previous host scope not restored')
    stage.mkdir(parents=True, exist_ok=False)
    before = json.loads(subprocess.check_output(['docker', 'inspect', 'dsv4-csa-cp8-off']))[0]
    (stage/'container.before.json').write_text(json.dumps(before, indent=2))
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    guard = context.Process(target=restoration_watchdog,
        args=(child, str(setting), '20', str(stage/'watchdog.json'), 14400))
    guard.start()
    child.close()
    if not parent.poll(10) or parent.recv() != 'armed':
        raise RuntimeError('Host guard unavailable')
    try:
        setting.write_text('0\n')
        output=stage/'cell' if reuse_shapes is None else stage/'cell/formal'
        if reuse_shapes is not None:
            output.parent.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', client_script,
                   '--model', model, '--inputs', cell['inputs'], '--output', str(output),
                   '--rate', str(rate if rate is not None else .005)]
        if rate is None:
            command.append('--serial')
        if reuse_shapes is None:
            command.append('--condition-full-output')
        else:
            command += ['--reuse-shapes', str(reuse_shapes)]
            (stage/'reused_warmup.json').write_text(json.dumps({
                'prepared_source': str(reuse_shapes),
                'prepared_sha256': hashlib.sha256(reuse_shapes.read_bytes()).hexdigest(),
                'same_container_required_by_prepare_run': True,
            }, indent=2))
        with (stage/'client.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        result = json.loads((output/'COMPLETE.json').read_text())
        if result['successes'] != cell['count'] or result['failures'] or result['actual_arrival_rate'] != rate:
            raise RuntimeError('Incomplete cohort or incorrect arrival mode/rate')
        if reuse_shapes is not None:
            (stage/'cell/COMPLETE.json').write_text(json.dumps(result, indent=2))
        after = json.loads(subprocess.check_output(['docker', 'inspect', 'dsv4-csa-cp8-off']))[0]
        if before['Id'] != after['Id']:
            raise RuntimeError('Container changed during dataset')
        drain_current_backend()
        (stage/'COMPLETE.json').write_text(json.dumps(result, indent=2))
    finally:
        (stage/'restored_host.json').write_text(json.dumps(restore_setting(setting, '20')))
        parent.send('restored_and_verified')
        parent.close()
        guard.join(timeout=5)


def worker() -> None:
    """Pro public two → Flash public two → GLM public two; no internal repeats."""
    try:
        wait_for_pro_internal()
        plan = json.loads((ROOT/'prepared_plan.json').read_text())
        for group in plan:
            model = group['model']
            for index, cell in enumerate(group['datasets']):
                stage = ROOT/model/cell['name']
                (ROOT/'status.json').write_text(json.dumps(dict(model=model, dataset=cell['name'], phase='running')))
                if model != 'pro' and index == 0:
                    command = [sys.executable, '-u', '/tmp/launch_corrective_only_v1144.py',
                               '--model', model, '--inputs', cell['inputs'], '--output', str(stage),
                               '--rate', '.005', '--serial', '--execute']
                    if model == 'glm':
                        command += ['--kv-cache-gib', '20']
                    stage.parent.mkdir(parents=True, exist_ok=True)
                    with (stage.parent/'launch.log').open('x') as log:
                        subprocess.run(command, stdout=log, stderr=log, check=True)
                else:
                    reuse_container(model, cell, stage)
                complete = json.loads((stage/'COMPLETE.json').read_text())
                if complete['successes'] != cell['count'] or complete['failures']:
                    raise RuntimeError('Incomplete model/dataset')
                if not json.loads((stage/'watchdog.json').read_text()).get('disarmed'):
                    raise RuntimeError('Host guard still armed')
        (ROOT/'COMPLETE.json').write_text(json.dumps({'models': 3, 'datasets_per_model': 2,
            'new_requests': 57, 'internal_reused': True, 'arrival_mode': 'serial', 'rate': None}))
    except BaseException as error:
        (ROOT/'FAILED.json').write_text(json.dumps({'error': repr(error)}))
        raise


def main() -> None:
    """Prepare once and start one background queue; refuse duplicate launches."""
    if '--worker' in sys.argv:
        worker()
        return
    ROOT.mkdir(exist_ok=False)
    prepare_inputs()
    with (ROOT/'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    record = dict(pid=process.pid, root=str(ROOT))
    (ROOT/'launch.json').write_text(json.dumps(record))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

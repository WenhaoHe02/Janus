# SPDX-License-Identifier: Apache-2.0
"""Measure NP GPU-write-packing TPOT on internal9 and TB12, one round each."""

# Standard
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

# Local
from compaction_ab_v1030 import restoration_watchdog, restore_setting
from drain_serving_backend_v1157 import drain_current_backend
from prepare_flash_ttft_ablation_v1185 import build_sources
from run_ds_selected_v1057 import snapshot_native_inputs

ROOT = Path('/tmp/flash_np_gpu_write_pack_v1218')
STARTUP = ROOT / 'startup'
SETTING = Path('/proc/sys/vm/compaction_proactiveness')
DATASETS = (
    ('internal', Path('/tmp/ds_ablation_native9_v1128/flash'), 9),
    ('terminalbench_2.1', Path('/tmp/flash_packing_native12_v1133'), 12),
)


def replace_once(source: str, before: str, after: str) -> str:
    """Replace only an exact unique reviewed source anchor."""
    if source.count(before) != 1:
        raise ValueError(f'Unexpected startup/client anchor: {before}')
    return source.replace(before, after)


def verify_ready() -> None:
    """Require the rejected K experiment stopped, host restored and fixed inputs."""
    cancelled = json.loads(Path('/tmp/flash_serving_merge_v1215/USER_CANCELLED.json').read_text())
    if cancelled['host_restored'] != 20 or SETTING.read_text().strip() != '20':
        raise RuntimeError('Old host scope still active')
    for root in ('/tmp/flash_np_internal_separate_v1214', '/tmp/flash_serving_merge_v1215'):
        if not json.loads((Path(root) / 'watchdog.json').read_text())['disarmed']:
            raise RuntimeError('An old guard remains active')
    prior = json.loads(subprocess.check_output(['docker', 'inspect', cancelled['container_stopped']]))[0]
    if prior['State']['Running']:
        raise RuntimeError('Cancelled K container is still running')
    running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
    if any(name.startswith('dsv4-csa-cp8-') for name in running):
        raise RuntimeError('Another DS experiment container is active')
    for _, inputs, count in DATASETS:
        manifest = json.loads((inputs / 'manifest.json').read_text())
        if manifest['selected_count'] != count or manifest['model_kind'] != 'flash':
            raise ValueError('Frozen NP cohort changed')


def prepare() -> None:
    """Build isolated NP startup with the existing tested GPU packer installer."""
    names = ('startup_ds_serving_v916.sh', 'exec_ds_serving_v916.py',
             'exec_ds_serving_v911.py', 'exec_ds_serving_v890.py')
    original = {name: (Path('/tmp') / name).read_text() for name in names}
    old_root = '/tmp/flash_ttft_ablation_startup_v1185'
    generated = build_sources(original, old_root)
    sources = {name: generated[name].replace(old_root, str(STARTUP)) for name in names}
    name = 'exec_ds_serving_v890.py'
    sources[name] = replace_once(sources[name], 'from flash_ttft_ablation_v1185 import environment',
                                 'from np_gpu_write_pack_v1218 import environment')
    sources[name] = replace_once(sources[name],
        "if arm != 'k_local':\n        raise ValueError('Prediction stays enabled')",
        "if arm != 'no_prefetch':\n        raise ValueError('GPU write experiment requires NP')")
    sources['gpu_write.sh'] = (
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'export PYTHONPATH="/tmp:${PYTHONPATH:-}"\n'
        'export LMCACHE_TTFT_ABLATION_ARM_V1185=baseline\n'
        f'exec bash {STARTUP}/startup_ds_serving_v916.sh\n')
    client = Path('/tmp/flash_np_read_separate_v1212/startup/client.py').read_text()
    sources['client.py'] = replace_once(client,
        'from np_read_separate_configuration_v1212 import environment',
        'from np_gpu_write_pack_v1218 import environment')
    STARTUP.mkdir()
    for name, source in sources.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
        (STARTUP / name).write_text(source)
    (ROOT / 'sources.json').write_text(json.dumps({
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}, indent=2))


def worker() -> None:
    """Run both datasets in one NP container, retaining full decode and all rows."""
    verify_ready()
    for name, inputs, _ in DATASETS:
        destination = ROOT / 'inputs' / name
        destination.mkdir(parents=True)
        proof = snapshot_native_inputs(inputs, destination)
        (destination / 'identity.json').write_text(json.dumps(proof, indent=2))
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    guard = context.Process(target=restoration_watchdog, args=(
        child, str(SETTING), '20', str(ROOT / 'watchdog.json'), 14400))
    guard.start()
    child.close()
    if not parent.poll(10) or parent.recv() != 'armed':
        parent.close()
        raise RuntimeError('Host restore guard failed')
    try:
        SETTING.write_text('0\n')
        (ROOT / 'status.json').write_text(json.dumps(dict(phase='relaunch_np_gpu_pack')))
        env = dict(os.environ, PYTHONPATH='/tmp',
                   LMCACHE_EXPERIMENT_STARTUP_SCRIPT=str(STARTUP / 'gpu_write.sh'),
                   LMCACHE_EXPERIMENT_FORCE_RELAUNCH='1', LMCACHE_ABLATION_MAX_MODEL_LEN='524288',
                   LMCACHE_SSD_LOCAL_CPU_GB='32', LMCACHE_CPU_PER_RANK_GB='128',
                   LMCACHE_CPU_FIRST_RANK_GB='128')
        with (ROOT / 'relaunch.log').open('x') as log:
            subprocess.run(['bash', '/tmp/relaunch_ds_serving_v892.sh', 'flash', 'no_prefetch'],
                           env=env, stdout=log, stderr=log, check=True)
        (ROOT / 'container.json').write_bytes(subprocess.check_output(
            ['docker', 'inspect', 'dsv4-csa-cp8-off']))
        cells = []
        for dataset, inputs, count in DATASETS:
            stage = ROOT / dataset
            stage.mkdir()
            (ROOT / 'status.json').write_text(json.dumps(dict(
                phase='warm_and_measure', dataset=dataset, trials=1)))
            with (stage / 'client.log').open('x') as log:
                subprocess.run([sys.executable, '-u', str(STARTUP / 'client.py'),
                                '--variant', 'baseline', '--inputs', str(inputs),
                                '--output', str(stage / 'cell'), '--trials', '1', '--full-output'],
                               env=env, stdout=log, stderr=log, check=True)
            result = json.loads((stage / 'cell/COMPLETE.json').read_text())
            if (len(result['trials']) != 1 or result['metric'] != 'TTFT+TPOT'
                    or result['trials'][0]['result']['successes'] != count
                    or result['trials'][0]['result']['failures']):
                raise RuntimeError('Incomplete GPU-write-packing trial')
            drain_current_backend()
            cell = dict(dataset=dataset, count=count, result=result)
            (stage / 'COMPLETE.json').write_text(json.dumps(cell, indent=2))
            cells.append(cell)
        with (ROOT / 'server_after.log').open('x') as log:
            subprocess.run(['docker', 'logs', 'dsv4-csa-cp8-off'], stdout=log, stderr=log, check=True)
        (ROOT / 'COMPLETE.json').write_text(json.dumps(dict(cells=cells, formal_requests=21), indent=2))
    finally:
        (ROOT / 'restored_host.json').write_text(json.dumps(restore_setting(SETTING, '20')))
        parent.send('restored_and_verified')
        parent.close()
        guard.join(timeout=5)


def main() -> None:
    """Start the two newly requested GPU packing cells once, no baseline repeat."""
    if '--worker' in sys.argv:
        try:
            worker()
        except BaseException as error:
            (ROOT / 'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
            raise
        return
    verify_ready()
    ROOT.mkdir(exist_ok=False)
    prepare()
    (ROOT / 'plan.json').write_text(json.dumps(dict(
        model='flash', arm='no_prefetch', datasets=[dict(name=n, count=c, inputs=str(p))
                                                  for n, p, c in DATASETS],
        trials=1, output='original full O', arrival='serial',
        read_pack='native serving batched submission; no new LBA coalescing patch',
        write_pack='GPU packs mapped pinned CPU buffer; unchanged CPU raw SSD writer',
        decode_write_pause=True, added_write_workload=False,
        metrics=['TTFT', 'TPOT', 'prefill'], baseline_repeated=False)))
    with (ROOT / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, root=str(ROOT))
    (ROOT / 'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

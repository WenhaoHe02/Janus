# SPDX-License-Identifier: Apache-2.0
"""Add only the four newly authorized public GLM fixed-layerwise periodic cells."""

# Standard
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/tmp/glm_layerwise_periodic_v1199')
PREVIOUS = Path('/tmp/flash_packing_ttft_v1197')
STARTUP = Path('/tmp/glm_sparse_layout_v1190/startup/startup.sh')
DATASETS = (
    dict(name='terminalbench_2.1', count=12,
         inputs='/tmp/layerwise_public_by_model_v1174/native/glm/terminalbench_2.1'),
    dict(name='swe_bench_pro', count=7,
         inputs='/tmp/layerwise_public_by_model_v1174/native/glm/swe_bench_pro'),
)


def verify_predecessor() -> None:
    """Require packing completion and released host guard before switching model.

    Raises RuntimeError if prior experiments or restoration are incomplete.
    """
    complete = json.loads((PREVIOUS / 'COMPLETE.json').read_text())
    guard = json.loads((PREVIOUS / 'watchdog.json').read_text())
    if complete.get('formal_observations') != 126 or not guard.get('disarmed'):
        raise RuntimeError('Packing queue has not completed/released host settings')
    if (PREVIOUS / 'FAILED.json').exists():
        raise RuntimeError('Packing queue failed; preserve and inspect before launch')
    if Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '20':
        raise RuntimeError('Previous host settings not restored')
    if not STARTUP.is_file():
        raise RuntimeError('Missing previously verified fixed GLM startup')


def launch_glm() -> None:
    """Launch the same fixed GLM configuration under its own restoration scope.

    This scope ends before periodic cells acquire their separate host guards.
    Raises RuntimeError or CalledProcessError on unsafe state or launch failure.
    """
    # Local
    from compaction_ab_v1030 import restoration_watchdog, restore_setting
    from diagnose_serving_decode_v973 import assert_idle
    from drain_serving_backend_v1157 import drain_current_backend

    assert_idle()
    drain_current_backend()
    stage = ROOT / 'deployment'
    stage.mkdir()
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    setting = Path('/proc/sys/vm/compaction_proactiveness')
    guard = context.Process(target=restoration_watchdog, args=(
        child, str(setting), '20', str(stage / 'watchdog.json'), 18000))
    guard.start()
    child.close()
    if not parent.poll(10) or parent.recv() != 'armed':
        parent.close()
        raise RuntimeError('Deployment restoration guard failed')
    try:
        setting.write_text('0\n')
        (ROOT / 'status.json').write_text(json.dumps(dict(phase='relaunch', model='glm')))
        env = dict(os.environ, LMCACHE_EXPERIMENT_STARTUP_SCRIPT=str(STARTUP),
                   LMCACHE_EXPERIMENT_FORCE_RELAUNCH='1', LMCACHE_ABLATION_MAX_MODEL_LEN='524288',
                   LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES=str(20 * 1024**3),
                   LMCACHE_SSD_LOCAL_CPU_GB='32', LMCACHE_CPU_PER_RANK_GB='128',
                   LMCACHE_CPU_FIRST_RANK_GB='128')
        with (stage / 'relaunch.log').open('x') as log:
            subprocess.run(['bash', '/tmp/relaunch_glm_serving_v957.sh', 'no_prefetch', 'ttft'],
                           env=env, stdout=log, stderr=log, check=True)
        info = subprocess.check_output(['docker', 'inspect', 'dsv4-csa-cp8-off'])
        (stage / 'container.json').write_bytes(info)
    finally:
        (stage / 'restored_host.json').write_text(json.dumps(restore_setting(setting, '20')))
        parent.send('restored_and_verified')
        parent.close()
        guard.join(timeout=5)


def worker() -> None:
    """Warm each public cohort once, then measure only .01 and .015 with full O.

    Raises on any partial cell; completed and failed records are never replaced.
    """
    # Local
    from run_layerwise_public_by_model_v1174 import reuse_container
    from run_layerwise_three_rates_v1180 import check_complete
    from run_ds_selected_v1057 import snapshot_native_inputs

    verify_predecessor()
    for dataset in DATASETS:
        path = ROOT / 'inputs' / dataset['name']
        path.mkdir(parents=True)
        identity = snapshot_native_inputs(Path(dataset['inputs']), path)
        (path / 'identity.json').write_text(json.dumps(identity, indent=2))
    launch_glm()
    for dataset in DATASETS:
        warm = None
        for rate in (.01, .015):
            stage = ROOT / dataset['name'] / str(rate).replace('.', 'p')
            (ROOT / 'status.json').write_text(json.dumps(dict(
                phase='running', model='glm', dataset=dataset['name'], rate=rate)))
            reuse_container('glm', dataset, stage, rate, warm,
                            client_script='/tmp/run_glm_layerwise_client_v1191.py')
            check_complete(stage, dataset['count'], rate)
            warm = stage / 'cell/formal/prepared.json'
    (ROOT / 'COMPLETE.json').write_text(json.dumps(dict(
        model='glm', cells=4, formal_requests=38, rates=[.01, .015], serial_rerun=False)))
    (ROOT / 'status.json').write_text(json.dumps(dict(phase='complete')))


def main() -> None:
    """Start one detached queue after native input and startup checks.

    Raises FileExistsError for a previous attempt, avoiding duplicate experiments.
    """
    if '--worker' in sys.argv:
        try:
            worker()
        except BaseException as error:
            (ROOT / 'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
            raise
        return
    verify_predecessor()
    for dataset in DATASETS:
        manifest = json.loads((Path(dataset['inputs']) / 'manifest.json').read_text())
        if manifest.get('selected_count') != dataset['count']:
            raise ValueError('Frozen public cohort count differs')
    ROOT.mkdir(exist_ok=False)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in STARTUP.parent.iterdir() if path.is_file()}
    (ROOT / 'plan.json').write_text(json.dumps(dict(
        model='glm', datasets=DATASETS, rates=[.01, .015], full_original_output=True,
        serial_rerun=False, source_fix='compact_layout_v1191', startup=str(STARTUP),
        startup_sha256=hashes, previous_serial_root='/tmp/glm_layerwise_fixed_v1191'), indent=2))
    with (ROOT / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, root=str(ROOT), cells=4, serial_rerun=False)
    (ROOT / 'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

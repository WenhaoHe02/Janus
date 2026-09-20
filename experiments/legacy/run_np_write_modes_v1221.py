# SPDX-License-Identifier: Apache-2.0
"""Run both requested real GPU write mechanisms using one round per dataset."""

# Standard
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Local
from diagnose_serving_decode_v973 import assert_idle
import run_np_gpu_write_pack_v1218 as runner

ROOT = Path('/tmp/flash_np_write_modes_v1221')
MODES = ('gpu_pack_cpu_write', 'gpu_nvme_write')
PREVIOUS = Path('/tmp/flash_np_gpu_write_pack_v1218')


def verify_ready() -> None:
    """Require predecessor guard released, host20 and an idle serving container."""
    if (not json.loads((PREVIOUS / 'watchdog.json').read_text())['disarmed']
            or Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '20'):
        raise RuntimeError('Predecessor host scope has not ended')
    if PREVIOUS.name != 'flash_np_gpu_write_pack_v1218' and not (PREVIOUS / 'COMPLETE.json').exists():
        raise RuntimeError('Previous real-write mode did not complete')
    assert_idle()


def prepare_mode(mode: str) -> None:
    """Reuse normal NP startup, replacing the unused hook with the actual writer hook."""
    runner.ROOT = ROOT / mode
    runner.STARTUP = runner.ROOT / 'startup'
    runner.ROOT.mkdir()
    runner.prepare()
    startup = runner.STARTUP / 'gpu_write.sh'
    startup.write_text(runner.replace_once(startup.read_text(),
        'export LMCACHE_TTFT_ABLATION_ARM_V1185=baseline\n',
        'export LMCACHE_TTFT_ABLATION_ARM_V1185=baseline\n'
        f'export LMCACHE_NP_WRITE_MODE_V1220={mode}\n'))
    executor = runner.STARTUP / 'exec_ds_serving_v890.py'
    source = executor.read_text()
    source = runner.replace_once(source, 'from np_gpu_write_pack_v1218 import environment',
                                 'from np_write_modes_configuration_v1221 import environment')
    anchor = "    print('DS_SERVING_V890_FINAL_FLAGS', flags, flush=True)"
    source = runner.replace_once(source, anchor,
        "    subprocess.run([sys.executable, '/tmp/install_np_write_modes_v1220.py',\n"
        "                    '--lmcache-root', str(root)], check=True)\n" + anchor)
    compile(source, str(executor), 'exec')
    executor.write_text(source)
    client = runner.STARTUP / 'client.py'
    source = runner.replace_once(client.read_text(),
        'from np_gpu_write_pack_v1218 import environment',
        'from np_write_modes_configuration_v1221 import environment')
    source = runner.replace_once(source, "if __name__ == '__main__':", 
        "from np_write_gate_v1221 import require_gpu_write\n"
        "before_measured_phase = require_gpu_write\n\nif __name__ == '__main__':")
    source = runner.replace_once(source,
        '        asyncio.run(replay(args, manifest, tokens, proof))',
        "        anchors = [dict(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())]\n"
        "        asyncio.run(replay(args, manifest, tokens, proof))\n"
        "        anchors.append(dict(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()))\n"
        "        (args.output / 'write_decode_clock_anchors.json').write_text(json.dumps(anchors))")
    compile(source, str(client), 'exec')
    client.write_text(source)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in runner.STARTUP.iterdir() if p.is_file()}
    (runner.ROOT / 'sources.json').write_text(json.dumps(sources, indent=2))


def worker() -> None:
    """Measure both real mechanisms sequentially; never overlap model/host scopes."""
    global PREVIOUS
    runner.verify_ready = verify_ready
    completed = []
    for mode in MODES:
        os.environ['LMCACHE_NP_WRITE_MODE_V1220'] = mode
        runner.ROOT = ROOT / mode
        runner.STARTUP = runner.ROOT / 'startup'
        (ROOT / 'status.json').write_text(json.dumps(dict(mode=mode, phase='running')))
        runner.worker()
        result = json.loads((runner.ROOT / 'COMPLETE.json').read_text())
        if result['formal_requests'] != 21:
            raise RuntimeError('Missing requested full-output measurements')
        completed.append(dict(mode=mode, result=result))
        PREVIOUS = runner.ROOT
    (ROOT / 'COMPLETE.json').write_text(json.dumps(dict(modes=completed), indent=2))


def main() -> None:
    """Launch four single-round cells; verify actual execution before any formal timing."""
    if '--worker' in sys.argv:
        try:
            worker()
        except BaseException as error:
            (ROOT / 'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
            raise
        return
    verify_ready()
    ROOT.mkdir(exist_ok=False)
    for mode in MODES:
        prepare_mode(mode)
    (ROOT / 'plan.json').write_text(json.dumps(dict(
        modes=MODES, datasets=['internal9', 'terminalbench12'], trials=1,
        metric_priority='TPOT', output='original full O', arrival='serial',
        cpu_default_unchanged=True, replaces_actual_alignment_copy=True,
        nvme_write_uses_original_extent_addresses=True,
        decode_write_pause_unchanged=True, actual8rank_execution_required=True)))
    with (ROOT / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    receipt = dict(pid=process.pid, root=str(ROOT))
    (ROOT / 'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

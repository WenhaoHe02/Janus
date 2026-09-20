# SPDX-License-Identifier: Apache-2.0
"""Replace only the seven GLM cells affected by missing compact-layout objects."""

# Standard
import json
from pathlib import Path
import subprocess
import sys

# Local
from run_layerwise_public_by_model_v1174 import reuse_container
from run_layerwise_three_rates_v1180 import check_complete

ROOT = Path('/tmp/glm_layerwise_fixed_v1191')
EXPECTED_CONTAINER = '67d328ba4bb140add24b585d32a7c2dbb5048e44b989fc2254a4cee22c6942cf'


def worker() -> None:
    """Reuse the validated fixed container; warm each complete dataset separately."""
    try:
        current = subprocess.check_output(['docker', 'inspect', '--format={{.Id}}', 'dsv4-csa-cp8-off'], text=True).strip()
        if current != EXPECTED_CONTAINER:
            raise RuntimeError('Validated fixed container changed; do not launch another')
        prior = Path('/tmp/glm_sparse_layout_v1190')
        if not (prior/'COMPLETE.json').exists() or not json.loads((prior/'watchdog.json').read_text()).get('disarmed'):
            raise RuntimeError('Diagnostic has not released its host scope')
        datasets = [
            dict(name='internal', count=13, inputs='/tmp/internal_union13_v1152/glm', rates=[None]),
            dict(name='terminalbench_2.1', count=12,
                 inputs='/tmp/layerwise_public_by_model_v1174/native/glm/terminalbench_2.1', rates=[None, .01, .015]),
            dict(name='swe_bench_pro', count=7,
                 inputs='/tmp/layerwise_public_by_model_v1174/native/glm/swe_bench_pro', rates=[None, .01, .015]),
        ]
        (ROOT/'plan.json').write_text(json.dumps(dict(datasets=datasets, container=current,
            reason='Replace GLM measurements with missing compact-layout objects',
            unchanged_models=['flash', 'pro'], original_data_preserved=True), indent=2))
        for dataset in datasets:
            warm = None
            for rate in dataset['rates']:
                slot = 'serial' if rate is None else str(rate).replace('.', 'p')
                stage = ROOT / dataset['name'] / slot
                (ROOT/'status.json').write_text(json.dumps(dict(dataset=dataset['name'], rate=rate, phase='running')))
                reuse_container('glm', dataset, stage, rate, warm,
                                client_script='/tmp/run_glm_layerwise_client_v1191.py')
                check_complete(stage, dataset['count'], rate)
                warm = stage/'cell/formal/prepared.json'
        (ROOT/'COMPLETE.json').write_text(json.dumps(dict(cells=7, formal_requests=70,
            model='glm', internal_rates=[None], public_rates=[None, .01, .015])))
    except BaseException as error:
        (ROOT/'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
        raise


def main() -> None:
    """Launch once; never recreate completed outputs or restart a running queue."""
    if '--worker' in sys.argv:
        worker()
        return
    ROOT.mkdir(exist_ok=False)
    with (ROOT/'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    receipt = dict(pid=process.pid, root=str(ROOT), reused_container=EXPECTED_CONTAINER)
    (ROOT/'launch.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

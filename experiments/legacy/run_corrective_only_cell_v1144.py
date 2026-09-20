# SPDX-License-Identifier: Apache-2.0
"""Measure an already launched corrective-only container using the existing runner."""
# Standard
import argparse
from array import array
import asyncio
import importlib
import json
import os
from pathlib import Path

# Local
from corrective_only_configuration_v1144 import corrective_only_flags


def main() -> None:
    """Prepare cold store/warmup then measure; never relaunch or replace a cohort."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('glm','flash','pro'),required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rate',type=float,default=.015)
    parser.add_argument('--serial',action='store_true')
    parser.add_argument('--reuse-shapes',type=Path)
    parser.add_argument('--condition-full-output',action='store_true')
    args=parser.parse_args()
    if args.rate<=0: raise ValueError('Arrival rate must be positive')
    args.arm='no_prefetch'; args.prefill_only=False; args.device_owner=False
    args.isolate_request_cache=True
    args.configuration_module=('serving_glm_configuration_v766' if args.model=='glm'
                               else 'serving_ds_configuration_v890')
    configuration=importlib.import_module(args.configuration_module)
    baseline_environment=configuration.environment

    def environment(arm: str) -> dict[str,str]:
        """Supply matched flags to the unchanged public runtime verifier."""
        return corrective_only_flags(args.model,baseline_environment(arm))

    configuration.environment=environment
    # This override is confined to this client process, not the running workers
    # or an HTTP import allowlist. Both prepare/replay still verify all8 workers.
    from run_original_serving_v957 import prepare_run,replay
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    if args.model=='glm' and 'glm' not in (manifest.get('model_display_name','')+' '+manifest['model']).lower():
        raise ValueError('GLM requires its own native input manifest')
    tokens=array('i'); path=args.inputs/'tokens.i32le'
    with path.open('rb') as stream: tokens.fromfile(stream,path.stat().st_size//4)
    if args.serial and args.model=='glm':
        os.environ['LMCACHE_GLM_SERIAL_WRITE_BARRIER_V1123']='1'
    root=args.output
    requested_serial=args.serial
    phases=('conditioning_excluded','formal') if args.condition_full_output else ('formal',)
    if args.condition_full_output: root.mkdir(exist_ok=False)
    for phase in phases:
        # Excluded conditioning is serial; only formal traffic uses arrival rate.
        args.serial=True if phase=='conditioning_excluded' else requested_serial
        if args.condition_full_output:
            args.output=root/phase
            args.reuse_shapes=root/'conditioning_excluded/prepared.json' if phase=='formal' else None
        proof=prepare_run(args,manifest,tokens)
        proof.update(comparison_arm='corrective_only',model=args.model,
            definition='no prediction; true Indexer then missing sparse KV reads',
            full_history_layer_restore=False, conditioning_excluded=phase=='conditioning_excluded',
            equivalent_to_existing_ds_np=args.model in ('flash','pro'))
        asyncio.run(replay(args,manifest,tokens,proof))
        result=json.loads((args.output/'COMPLETE.json').read_text())
        if result['successes']!=manifest['selected_count'] or result['failures']:
            raise RuntimeError('Conditioning/formal request failed')
    if args.condition_full_output:
        (root/'COMPLETE.json').write_text((root/'formal/COMPLETE.json').read_text())


if __name__=='__main__':
    main()

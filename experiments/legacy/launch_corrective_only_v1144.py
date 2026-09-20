# SPDX-License-Identifier: Apache-2.0
"""Plan or explicitly execute one three-model corrective-only serving cell."""
# Standard
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys


def main() -> None:
    """Default to plan-only; execution requires the current cap queue to finish."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('glm','flash','pro'),required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rate',type=float,default=.015)
    parser.add_argument('--serial',action='store_true')
    parser.add_argument('--kv-cache-gib', type=int,
                        help='Explicit per-GPU KV capacity; record before launch')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--after',type=Path,default=Path('/tmp/flash_write_caps_v1143'))
    args=parser.parse_args()
    # A fresh launcher reads cancellation before creating a host guard/container.
    if args.execute:
        from skip_flash_internal_v1183 import cancel_requested_cell
        cancel_requested_cell(args.model, args.output)
    if args.kv_cache_gib is not None and args.kv_cache_gib <= 0:
        raise ValueError('KV capacity must be positive')
    startup=Path('/tmp/corrective_only_v1144')/args.model/'startup.sh'
    launcher=('/tmp/relaunch_glm_serving_v957.sh' if args.model=='glm' else '/tmp/relaunch_ds_serving_v892.sh')
    command=(['bash',launcher,'no_prefetch','ttft'] if args.model=='glm'
             else ['bash',launcher,args.model,'no_prefetch'])
    cell=['python3','-u','/tmp/run_corrective_only_cell_v1144.py','--model',args.model,
          '--inputs',args.inputs.as_posix(),'--output',(args.output/'cell').as_posix(),'--rate',str(args.rate),
          '--condition-full-output']
    if args.serial: cell.append('--serial')
    plan=dict(model=args.model,startup=startup.as_posix(),launch=command,measure=cell,
        execute=args.execute,actual_rate=None if args.serial else args.rate,
        arrival_mode='serial' if args.serial else 'periodic',
        scope='true Indexer followed by sparse miss loading; no full-history layer restore',
        existing_ds_np_semantically_equivalent=args.model in ('flash','pro'),
        kv_cache_gib=args.kv_cache_gib)
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute: return
    if os.name!='posix': raise RuntimeError('Execute only on the GPU server; local plan is read-only')
    # Imports that depend on the remote runtime occur only on explicit execution.
    from compaction_ab_v1030 import restoration_watchdog,restore_setting
    from diagnose_serving_decode_v973 import assert_idle
    from run_ds_selected_v1057 import snapshot_native_inputs
    from drain_serving_backend_v1157 import drain_current_backend as drain

    if not (args.after/'ALL_CAPS_COMPLETE.json').exists() or not (args.after/'SERVING_RESTORED.json').exists():
        raise RuntimeError('Current write-cap experiment has not completed/restored; no launch')
    if (args.after/'FAILED.json').exists() or not json.loads((args.after/'watchdog.json').read_text()).get('disarmed'):
        raise RuntimeError('Previous run failed or still owns the host scope')
    if not startup.exists(): raise FileNotFoundError('Prepare this model startup first')
    if args.output.exists(): raise FileExistsError(args.output)
    assert_idle(); drain()
    setting=Path('/proc/sys/vm/compaction_proactiveness')
    if setting.read_text().strip()!='20': raise RuntimeError('Host setting not restored')
    args.output.mkdir(parents=True)
    identity=snapshot_native_inputs(args.inputs,args.output)
    (args.output/'plan.json').write_text(json.dumps(dict(**plan,identity=identity),indent=2))
    context=multiprocessing.get_context('spawn'); parent,child=context.Pipe()
    guard=context.Process(target=restoration_watchdog,args=(child,str(setting),'20',str(args.output/'watchdog.json'),18000))
    guard.start(); child.close()
    if not parent.poll(10) or parent.recv()!='armed': raise RuntimeError('Host guard failed')
    try:
        setting.write_text('0\n')
        env=dict(os.environ,LMCACHE_EXPERIMENT_STARTUP_SCRIPT=str(startup),
            LMCACHE_EXPERIMENT_FORCE_RELAUNCH='1',LMCACHE_ABLATION_MAX_MODEL_LEN='524288',
            LMCACHE_SSD_LOCAL_CPU_GB='32',LMCACHE_CPU_PER_RANK_GB='128',LMCACHE_CPU_FIRST_RANK_GB='128')
        if args.kv_cache_gib is not None:
            env['LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES'] = str(args.kv_cache_gib * 1024**3)
        with (args.output/'relaunch.log').open('x') as log:
            subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
        if args.kv_cache_gib is not None and args.model in ('flash', 'pro'):
            manifest=json.loads((args.inputs/'manifest.json').read_text())
            required=max(r['prompt_tokens'] + r.get('output_budget', 0)
                         for chain in manifest['chains'] for r in chain['rounds'])
            logs=subprocess.check_output(['docker', 'logs', 'dsv4-csa-cp8-off'],
                                         stderr=subprocess.STDOUT, text=True)
            capacities=[int(value.replace(',', '')) for value in
                        re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', logs)]
            if not capacities or min(capacities) < required:
                raise RuntimeError(f'KV capacity {capacities} cannot cover required {required} tokens')
            (args.output/'capacity_preflight.json').write_text(json.dumps(
                dict(kv_cache_gib=args.kv_cache_gib, reported_token_capacities=capacities,
                     required_prompt_plus_output=required), indent=2))
        with (args.output/'client.log').open('x') as log:
            subprocess.run(cell,stdout=log,stderr=log,check=True)
        result=json.loads((args.output/'cell/COMPLETE.json').read_text())
        if result['failures'] or result['unavailable'] or result['successes']!=result['selected']:
            raise RuntimeError('Corrective-only cell contains failed or missing requests')
        drain()
        (args.output/'COMPLETE.json').write_text((args.output/'cell/COMPLETE.json').read_text())
    except BaseException as error:
        (args.output/'FAILED.json').write_text(json.dumps(dict(error=repr(error))))
        raise
    finally:
        (args.output/'restored_host.json').write_text(json.dumps(restore_setting(setting,'20')))
        parent.send('restored_and_verified'); parent.close(); guard.join(timeout=5)


if __name__=='__main__':
    main()

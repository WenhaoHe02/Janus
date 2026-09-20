# SPDX-License-Identifier: Apache-2.0
"""Prepare with uncapped writes, then measure exactly one configured write cap."""
# Standard
import argparse
from array import array
import asyncio
import json
from pathlib import Path
import time
import urllib.request

# Local
from run_original_serving_v957 import get_json, prepare_run, replay


def configure(rate: float) -> list[dict]:
    """Set the existing public background limiter and verify all eight workers."""
    states=[]
    for rank in range(8):
        request=urllib.request.Request(f'http://127.0.0.1:{7000+rank}/write_slack/background_limit',
            data=json.dumps(dict(rate_mib_s=rate,burst_mib=8)).encode(),
            headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=20) as response:
            state=json.load(response)
        if (state.get('status')!='success' or state.get('background_rate_mib_s')!=rate
                or state.get('zero_cap_unlimited_overlap') is not True):
            raise RuntimeError(f'Rank{rank} did not activate the requested write policy')
        states.append(state)
    return states


def drain() -> list[dict]:
    """Wait for real outstanding writes without clearing them or changing policy."""
    deadline=time.monotonic()+900
    while True:
        states=[get_json(f'http://127.0.0.1:{7000+r}/write_slack/status') for r in range(8)]
        if not any(s.get(k,0) for s in states for k in ('queued_waves','active_waves','active_io_blocks')):
            return states
        if time.monotonic()>deadline:
            raise TimeoutError('Write queues did not drain')
        time.sleep(2)


def main() -> None:
    """Keep original full requests/periodic runner, with cap outside warmup timing."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cap',type=float,choices=(0,1024,512,256,128,64),required=True)
    parser.add_argument('--reuse-shapes',type=Path)
    args=parser.parse_args()
    args.arm='k_local'; args.configuration_module='serving_ds_configuration_v890'
    args.rate=.015; args.serial=False; args.prefill_only=False
    args.device_owner=False; args.isolate_request_cache=True
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    tokens=array('i'); token_path=args.inputs/'tokens.i32le'
    with token_path.open('rb') as stream:
        tokens.fromfile(stream,token_path.stat().st_size//4)
    prepared_policy=configure(0)
    proof=prepare_run(args,manifest,tokens)
    ready=drain()
    active=configure(args.cap)
    proof['write_cap']=dict(cap_mib_s=args.cap,no_cap=args.cap==0,
        phase='background/read overlap; decode paused; explicit tool slack bypasses bucket',
        unit='MiB/s per rank SSD',burst_mib=8,preparation_cap='unlimited_excluded',
        workers=active,old_ablation_flags_off=True)
    (args.output/'cap_policy.json').write_text(json.dumps(dict(
        preparation=prepared_policy,drained_before_formal=ready,formal=active),indent=2))
    try:
        asyncio.run(replay(args,manifest,tokens,proof))
        after=[get_json(f'http://127.0.0.1:{7000+r}/write_slack/status') for r in range(8)]
        if any(s.get('background_rate_mib_s')!=args.cap for s in after):
            raise RuntimeError('Write cap changed during measurement')
        (args.output/'cap_after_formal.json').write_text(json.dumps(after,indent=2))
    finally:
        reset=configure(0)
        (args.output/'cap_reset_after_formal.json').write_text(json.dumps(reset,indent=2))
    (args.output/'WRITE_CAP_VERIFIED.json').write_text(json.dumps(dict(rate=.015,cap_mib_s=args.cap,
        records_preserved=True,actual_cap_checked_before_and_after=True)))


if __name__=='__main__':
    main()

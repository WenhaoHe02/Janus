# SPDX-License-Identifier: Apache-2.0
"""Replay frozen native-token sessions at open-loop session arrival rates."""

# Standard
import argparse
from array import array
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

# Local
from serving_observation_v762 import StreamObservation, next_turn_due, session_arrivals
from serving_cache_tag_v1033 import cache_parameters


def complete(endpoint: str, model: str, prompt: list[int], output: int,
             *, cache_tag: str | None = None) -> dict[str, Any]:
    """Stream original tokens/output budget, optionally isolating their cache.

    Return timings or an explicit failure record. Invalid cache tags raise
    ValueError before a request is sent; the tag never changes prompt content.
    """
    body = dict(model=model, prompt=prompt, max_tokens=output, ignore_eos=True,
        temperature=0, stream=True, stream_options={'include_usage': True},
        kv_transfer_params=cache_parameters(False, cache_tag))
    request = urllib.request.Request(endpoint + '/v1/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    observation = StreamObservation(time.monotonic())
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for raw in response:
                received = time.monotonic()
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                value = line[5:].strip()
                if value == '[DONE]':
                    observation.done = True
                    break
                observation.observe(json.loads(value), received)
        result = observation.result(time.monotonic(), output)
        if result['usage'].get('prompt_tokens') != len(prompt):
            raise ValueError('server input length differs from frozen source')
        return dict(status='success', **result)
    except Exception as error:
        return dict(status='failed', error_type=type(error).__name__, error=str(error),
            request_id=observation.request_id, usage=observation.usage,
            send_s=observation.sent, end_s=time.monotonic(),
            first_visible_s=observation.first, last_visible_s=observation.last)


async def replay(inputs: Path, output: Path, endpoint: str, rate: float,
                 seed: int, tool_policy: Path, prepared_state: Path,
                 arrival_mode: str = 'poisson') -> None:
    """Replay dependent turns without resetting cache or reinserting generated output.

    Setup and runtime proof must be supplied before this timed-only runner.
    Failures terminate their dependent session; other sessions continue. All
    unsent descendants are recorded, not silently removed from the denominator.
    """
    manifest = json.loads((inputs / 'manifest.json').read_text())
    policy = json.loads(tool_policy.read_text())
    setup = json.loads(prepared_state.read_text())
    if setup.get('decode_mode') != 'normal_demand_decode':
        raise ValueError('missing normal-decode deployment proof')
    info = json.loads(subprocess.check_output(['docker', 'inspect', setup['container']]))[0]
    if not info['State']['Running'] or info['Id'] != setup['container_id']:
        raise ValueError('prepared model/container no longer matches')
    required = setup['required_worker_env']
    mandatory = {'LMCACHE_SERVING_EXPERIMENT': '1',
        'LMCACHE_INDEXER_PROFILE_ACCURACY': '0',
        'LMCACHE_CSA_ATTENTION_KV_TIMING': '0',
        'LMCACHE_TUTTI_DEVICE_TIMESTAMPS': '0',
        'LMCACHE_GLM_IO_VOLUME_PROFILE': '0',
        'LMCACHE_TUTTI_COUNTED_VOLUME_PROFILE': '0',
        'LMCACHE_GLM_VOLUME_V750': '0', 'LMCACHE_DS_COMPLETE_IO_V510': '0'}
    if any(required.get(key) != value for key, value in mandatory.items()):
        raise ValueError('prepared state lacks required profile/decode controls')
    envs = []
    for rank in range(8):
        with urllib.request.urlopen(f'http://127.0.0.1:{7000+rank}/env', timeout=10) as response:
            env = json.load(response)
        if any(env.get(key) != value for key, value in required.items()):
            raise ValueError(f'worker{rank} differs from prepared configuration')
        envs.append(dict(rank=rank, values={key: env[key] for key in required}))
    chains = manifest['chains']
    bootstrap_ids = {chain['rounds'][0]['source_request_id'] for chain in chains}
    if bootstrap_ids != set(setup['published_setup_source_ids']):
        raise ValueError('setup source cohort differs')
    if not policy.get('provenance'):
        raise ValueError('tool delays require explicit measured/simulated provenance')
    for chain in chains:
        for row in chain['rounds'][2:]:
            next_turn_due(0, float(policy['seconds_by_source_id'][row['source_request_id']]))
    tokens = array('i')
    packed = inputs / 'tokens.i32le'
    with packed.open('rb') as stream:
        tokens.fromfile(stream, packed.stat().st_size // 4)
    if sys.byteorder != 'little':
        tokens.byteswap()
    if arrival_mode == 'periodic':
        if rate <= 0:
            raise ValueError('Arrival rate must be positive')
        arrivals = [index / rate for index in range(len(chains))]
    elif arrival_mode == 'poisson':
        arrivals = session_arrivals(len(chains), rate, seed)
    else:
        raise ValueError('Unsupported arrival mode')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runtime.json').write_text(json.dumps(dict(setup=setup, actual_env=envs,
        inputs=str(inputs), tool_policy=policy, arrival_offsets_s=arrivals,
        offered_sessions_s=rate, seed=seed, arrival_mode=arrival_mode, profile_measurement=False,
        generated_output_reinserted=False, source_O_preserved=True), indent=2))
    # No GPU flush, setup request, or warmup occurs inside the measured interval.
    start = time.monotonic()
    completed = []
    sessions = []
    with (output / 'requests.jsonl').open('x') as log:
        async def session(chain: dict, arrival: float) -> None:
            due = start + arrival
            await asyncio.sleep(max(0, due - time.monotonic()))
            rows = chain['rounds'][1:]
            failed = False
            for index, row in enumerate(rows):
                identity = dict(chain=chain['chain'], source_round=row['round'],
                    source_request_id=row['source_request_id'], source_output=row['output_budget'],
                    source_prompt_tokens=row['prompt_tokens'], scheduled_session_s=start+arrival)
                if failed:
                    record = dict(identity, status='not_sent_after_session_failure')
                else:
                    delay = 0 if index == 0 else float(policy['seconds_by_source_id'][row['source_request_id']])
                    due = next_turn_due(due, delay)
                    await asyncio.sleep(max(0, due - time.monotonic()))
                    offset, count = row['token_offset'], row['prompt_tokens']
                    prompt = list(tokens[offset:offset+count])
                    if len(prompt) != count:
                        raise ValueError('frozen token corpus truncated')
                    result = await asyncio.to_thread(complete, endpoint, manifest['model'], prompt, row['output_budget'])
                    record = dict(identity, tool_delay_s=delay, eligible_s=due, **result)
                    due = result['end_s']
                    failed = result['status'] != 'success'
                completed.append(record)
                log.write(json.dumps(record) + '\n')
                log.flush()
            sessions.append(dict(chain=chain['chain'], failed=failed,
                scheduled_arrival_s=start+arrival, end_s=time.monotonic(),
                e2e_ms=1000*(time.monotonic()-start-arrival)))
        await asyncio.gather(*(session(chain, arrival) for chain, arrival in zip(chains, arrivals, strict=True)))
    end = time.monotonic()
    (output / 'session_results.json').write_text(json.dumps(sessions, indent=2))
    (output / 'COMPLETE.json').write_text(json.dumps(dict(request_records=len(completed),
        successes=sum(row['status']=='success' for row in completed),
        failures=sum(row['status']=='failed' for row in completed),
        unsent=sum(row['status'].startswith('not_sent') for row in completed),
        interval_start_s=start, interval_end_s=end,
        actual_prefill_throughput='pending request-specific engine/cache log join; reference A is not used'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tool-policy', type=Path, required=True)
    parser.add_argument('--prepared-state', type=Path, required=True)
    parser.add_argument('--arrival-mode', choices=('poisson', 'periodic'), default='poisson')
    args = parser.parse_args()
    asyncio.run(replay(args.inputs, args.output, args.endpoint, args.rate,
                       args.seed, args.tool_policy, args.prepared_state, args.arrival_mode))

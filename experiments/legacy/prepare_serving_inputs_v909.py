# SPDX-License-Identifier: Apache-2.0
"""Warm separate synthetic cache keys, then seed only original predecessors."""

# Standard
import argparse
from array import array
from dataclasses import asdict
import json
import importlib
from pathlib import Path
import re
import subprocess
import time
import urllib.request

# Local
from run_glm52_recovery_smoke import post_rank, token_sequence, wait_reuse_ready
from strict_stream_completion_v909 import stream_completion
from prewarm_glm_proxy_buckets_v438 import hit_tokens
from serving_preparation_guard_v920 import (
    publication_decision, validate_predecessor_evidence, validate_warmup_records,
)


def wait_seed_publication(
    container: str, request_id: str, prompt_tokens: int, started: float,
    *, chunk_size: int,
) -> dict:
    """Wait for new writes, or verify the same request already hit full chunks.

    Args:
        container: Current serving container, not a different arm's container.
        request_id: Identity of the successfully completed seed request.
        prompt_tokens: Original unmodified prompt length.
        started: Request start time used to limit the exact-ID log lookup.
        chunk_size: Deployed storage chunk size; this runner uses 256-token KV.
    Returns:
        Publication readiness evidence or exact cached full-chunk evidence.
    Raises:
        ValueError: Rank evidence reports a failed or insufficient publication.
        RuntimeError: Exact request-specific hit evidence is absent/ambiguous.
        TimeoutError: A genuinely new publication fails to become ready.
    """
    restored = hit_tokens(container, request_id, started)
    states = [post_rank(7000 + rank, request_id) for rank in range(8)]
    decision = publication_decision(states, prompt_tokens=prompt_tokens,
                                    chunk_size=chunk_size, restored_tokens=restored)
    if decision != 'pending':
        return dict(wait_s=0.0, basis=decision, restored_tokens=restored,
                    statuses=states)
    ready = wait_reuse_ready(request_id)
    ready['basis'] = 'published'
    return ready


def main() -> None:
    """Prepare the frozen pilot without generating or storing formal suffixes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('k_local', 'no_prefetch', 'ssd_only', 'dram'), default='k_local')
    parser.add_argument('--configuration-module', default='serving_glm_configuration_v766')
    parser.add_argument('--resume-ssd-seed-id', default=None)
    parser.add_argument('--reuse-shapes-from', type=Path)
    parser.add_argument('--reuse-warmup-records', type=Path)
    parser.add_argument('--reuse-warmup-env', type=Path)
    parser.add_argument('--warmup-repeats', type=int, default=1)
    args = parser.parse_args()
    if args.warmup_repeats < 1:
        raise ValueError('warmup-repeats must be positive')
    if args.configuration_module not in ('serving_glm_configuration_v766',
                                         'serving_ds_configuration_v890'):
        raise ValueError('Unsupported serving configuration module')
    environment = importlib.import_module(args.configuration_module).environment
    if args.resume_ssd_seed_id and args.arm != 'ssd_only':
        raise ValueError('seed continuation is restricted to ordinary SSD storage')
    args.output.mkdir(exist_ok=bool(args.resume_ssd_seed_id))
    container = 'dsv4-csa-cp8-off'
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    deadline = time.monotonic() + 1800
    while True:
        info = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
        if not info['State']['Running']:
            raise RuntimeError('model container stopped during preparation')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('model startup did not become healthy')
        time.sleep(5)
    required = environment(args.arm)
    config = subprocess.check_output(
        ['docker', 'exec', container, 'cat', '/tmp/lmcache_ssd_tutti_kvobj.yaml'],
        text=True)
    chunk_match = re.search(r'^chunk_size:\s*(\d+)\s*$', config, re.MULTILINE)
    if chunk_match is None or int(chunk_match[1]) != 256:
        raise ValueError('this preparation runner requires verified chunk_size=256')
    if args.reuse_shapes_from:
        previous = json.loads(args.reuse_shapes_from.read_text())
        if previous['container_id'] != info['Id'] or previous['required_worker_env'] != required:
            raise ValueError('compiled-shape reuse requires the same container and flags')
    envs = []
    for rank in range(8):
        with urllib.request.urlopen(f'http://127.0.0.1:{7000+rank}/env', timeout=10) as response:
            env = json.load(response)
        different = {k: env.get(k) for k, v in required.items() if env.get(k) != v}
        if different:
            raise ValueError(f'rank {rank} configuration differs: {different}')
        envs.append({'rank': rank, 'env': {k: env[k] for k in required}})
    (args.output / 'worker_env.json').write_text(json.dumps(envs, indent=2))
    seed_value = 788031
    shapes = sorted({(r['reusable_input_prefix_tokens'] // 256 * 256,
                      r['prompt_tokens'] - r['reusable_input_prefix_tokens'] // 256 * 256)
                     for c in manifest['chains'] for r in c['rounds'][1:]})
    if args.reuse_warmup_records:
        if not args.reuse_warmup_env or args.reuse_shapes_from:
            raise ValueError('warmup-record reuse needs its worker env and no other reuse mode')
        previous_envs = json.loads(args.reuse_warmup_env.read_text())
        if previous_envs != envs:
            raise ValueError('warmup records have different worker configuration')
        current_log = subprocess.check_output(
            ['docker', 'logs', container], stderr=subprocess.STDOUT, text=True)
        current_ids = set(re.findall(r'Reqid: (cmpl-[^- ,]+)', current_log))
        validate_warmup_records(
            [json.loads(line) for line in args.reuse_warmup_records.read_text().splitlines()],
            shapes=shapes, repeats=args.warmup_repeats,
            current_container_request_ids=current_ids)
    prefix = token_sequence(max(h for h, a in shapes) + 256, seed_value)
    exact_history_seed = args.configuration_module == 'serving_ds_configuration_v890'
    with (args.output / 'preparation.jsonl').open('a' if args.resume_ssd_seed_id else 'x') as log:
        def record(row: dict) -> None:
            """Persist each completed preparation stage immediately."""
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
        if args.reuse_warmup_records:
            record({'phase': 'reuse_existing_compiled_shapes',
                    'container_id': info['Id'],
                    'evidence': str(args.reuse_warmup_records),
                    'verified_shape_count': len(shapes),
                    'cache_publication_reused': False})
        elif args.reuse_shapes_from:
            record({'phase': 'reuse_existing_compiled_shapes', 'proof': str(args.reuse_shapes_from)})
        elif args.resume_ssd_seed_id:
            record({'phase': 'existing_synthetic_seed_excluded',
                    'request_id': args.resume_ssd_seed_id,
                    'publication': 'ordinary SSD: raw-generation barrier not applicable; subsequent exact-hit checks required'})
        elif not exact_history_seed:
            seed_started = time.time()
            seed = stream_completion(prefix, skip_save=False)
            if seed.request_id is None:
                raise RuntimeError('warmup seed has no request ID')
            ready = (dict(wait_s=None) if args.arm in ('ssd_only', 'dram')
                     else wait_seed_publication(container, seed.request_id,
                                                len(prefix), seed_started,
                                                chunk_size=256))
            record({'phase': 'synthetic_seed_excluded', 'completion': asdict(seed),
                    'publication_wait_s': ready['wait_s']})
        for index, (history, append) in enumerate(
                [] if args.reuse_shapes_from or args.reuse_warmup_records else shapes):
            if exact_history_seed:
                # All DS storage paths omit intermediate SWA/compressor state.
                # A shorter prefix inside one long seed is not a saved tail.
                # Give each shape a disjoint key and the exact saved boundary.
                prefix = token_sequence(history, seed_value + 1000 + index)
                seed_started = time.time()
                seed = stream_completion(prefix, skip_save=False)
                if seed.request_id is None:
                    raise RuntimeError('exact-history warmup seed has no request ID')
                ready = (dict(wait_s=None) if args.arm in ('ssd_only', 'dram')
                         else wait_seed_publication(container, seed.request_id,
                                                    len(prefix), seed_started,
                                                    chunk_size=256))
                record({'phase': 'exact_history_seed_excluded', 'H': history,
                        'completion': asdict(seed), 'publication_wait_s': ready['wait_s']})
            suffix = token_sequence(append, seed_value + 100 + index)
            if history < len(prefix) and suffix[0] == prefix[history]:
                suffix[0] = 1000 + ((suffix[0] - 999) % 28000)
            for repeat in range(args.warmup_repeats):
                started = time.time()
                result = stream_completion(prefix[:history] + suffix, skip_save=True)
                if result.request_id is None:
                    raise RuntimeError('warmup returned no identity')
                restored = hit_tokens(container, result.request_id, started)
                record({'phase': 'shape_warmup_excluded', 'H': history, 'A': append,
                        'repeat': repeat + 1, 'actual_hit': restored,
                        'completion': asdict(result)})
                if restored != history:
                    raise ValueError('synthetic warmup did not exercise requested cache hit')
        packed = args.inputs / 'tokens.i32le'
        tokens = array('i')
        with packed.open('rb') as stream:
            tokens.fromfile(stream, packed.stat().st_size // 4)
        published = []
        predecessor_evidence = []
        for chain in manifest['chains']:
            row = chain['rounds'][0]
            offset = row['token_offset']
            started = time.time()
            result = stream_completion(list(tokens[offset:offset+row['prompt_tokens']]), skip_save=False)
            if result.request_id is None:
                raise RuntimeError('source setup returned no request ID')
            record({'phase': 'original_predecessor_completion_observed',
                    'container_id': info['Id'],
                    'source_request_id': row['source_request_id'],
                    'completion': asdict(result)})
            ready = (dict(wait_s=None) if args.arm in ('ssd_only', 'dram')
                     else wait_seed_publication(container, result.request_id,
                                                row['prompt_tokens'], started,
                                                chunk_size=256))
            published.append(row['source_request_id'])
            predecessor_evidence.append({
                'source_request_id': row['source_request_id'],
                'container_id': info['Id'], 'required_worker_env': required,
                'request_id': result.request_id,
                'completion_tokens': result.completion_tokens,
                'readiness': ready.get('basis'),
            })
            record({'phase': 'original_predecessor_setup_excluded',
                    'source_request_id': row['source_request_id'], 'completion': asdict(result),
                    'container_id': info['Id'], 'readiness_evidence': ready,
                    'publication_wait_s': ready['wait_s'], 'wall_s': time.time()-started})
    current = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    if current['Id'] != info['Id'] or not current['State']['Running']:
        raise ValueError('container changed during preparation')
    if args.arm not in ('ssd_only', 'dram'):
        published = validate_predecessor_evidence(
            predecessor_evidence,
            expected_source_ids=[c['rounds'][0]['source_request_id']
                                 for c in manifest['chains']],
            container_id=info['Id'], required_env=required)
    proof = dict(container=container, container_id=info['Id'],
                 decode_mode='normal_demand_decode', required_worker_env=required,
                 published_setup_source_ids=published, preparation_dir=str(args.output),
                 predecessor_evidence=predecessor_evidence,
                 synthetic_warmup=True, formal_suffixes_saved_during_preparation=False,
                 exact_history_warmup=exact_history_seed,
                 warmup_repeats=args.warmup_repeats,
                 publication_basis=('ordinary SSD: subsequent actual-hit log verification required'
                                    if args.arm in ('ssd_only', 'dram') else 'eight-rank reuse_ready barrier'))
    (args.output / 'prepared_state.json').write_text(json.dumps(proof, indent=2))
    print('PREPARED_FOR_TIMED_PILOT', flush=True)


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
"""Prepare only agreed source IDs with verified predecessors for Pro serving."""

# Standard
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

# Local
from build_serving_tool_policy_v907 import interturn_gap
from prepare_native_chain_replay_v587 import prepare


def read_original(location: dict, identity: str) -> dict:
    """Read the exact recorded request and reject missing or failed originals."""
    with Path(location['source_file']).open(encoding='utf-8') as stream:
        line = next(line for number, line in enumerate(stream, 1)
                    if number == location['source_line'])
    row = json.loads(line)
    if row['request_id'] != identity or not row.get('is_complete'):
        raise ValueError('Original request identity/completion differs')
    if row.get('response_status') != 200 or not row.get('request_body', {}).get('messages'):
        raise ValueError('No successful original request body')
    return row


def build_source(selection_path: Path, graph_path: Path, output: Path) -> dict:
    """Freeze verified original pairs; explicitly retain every excluded candidate."""
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    edges = {edge['child']: edge for edge in graph['edges']}
    chains, requests, membership, exclusions, waits = [], {}, [], [], {}
    seen = set()
    for row in selection['requests']:
        identity = row['id']
        if identity in seen:
            raise ValueError('Duplicate selected source ID')
        seen.add(identity)
        try:
            original = read_original(row, identity)
            if identity not in edges:
                raise ValueError('no_verified_original_predecessor')
            parent_id = edges[identity]['parent']
            parent_location = graph['requests'][parent_id]
            parent = read_original(parent_location, parent_id)
            gap = interturn_gap(parent, original)
        except (ValueError, KeyError, FileNotFoundError) as error:
            exclusions.append(dict(source_request_id=identity, cluster=row['cluster'],
                                   reason=str(error)))
            continue
        for rid, location, record in [(identity, row, original),
                                      (parent_id, parent_location, parent)]:
            requests[rid] = dict(source_file=location['source_file'],
                                 source_line=location['source_line'],
                                 reported_history=record.get('cached_tokens'),
                                 reported_prompt=record.get('prompt_tokens'))
        chains.append([parent_id, identity])
        membership.append(dict(source_request_id=identity, cluster=row['cluster'],
                               source_H=row['history'], source_A=row['append'],
                               source_O=row['output']))
        waits[identity] = gap
    source = dict(dataset='internal', unbranched_chains=chains, requests=requests,
                  membership=membership, measured_request_count=len(chains),
                  cluster_counts=dict(Counter(r['cluster'] for r in membership)),
                  selected_count=len(seen), excluded_source_candidates=exclusions,
                  selection_file=str(selection_path),
                  selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
    output.mkdir(exist_ok=False)
    (output/'source.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    (output/'tool_policy.json').write_text(json.dumps(dict(
        provenance='Original next request start minus previous response end; tool/client/network idle proxy',
        seconds_by_source_id=waits, first_formal_wait='arrival scheduled independently'),
        indent=2), encoding='utf-8')
    print(json.dumps(dict(phase='original_pairs', selected=len(seen),
                          verified=len(chains), exclusions=exclusions)), flush=True)
    return source


def main() -> None:
    """Use the frozen DS encoder, without executing GPU inference or new selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = build_source(args.selection, args.graph, args.output)
    if not source['unbranched_chains']:
        raise ValueError('No original predecessor pairs; no replay started')
    command = ['docker', 'exec', '-i', 'dsv4-csa-cp8-off', 'python3',
               '/tmp/prepare_ds_serving_expanded_v904.py', '--worker',
               '--source', str(args.output/'source.json'), '--output', str(args.output/'inputs')]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True) as worker:
        assert worker.stdin is not None and worker.stdout is not None
        evidence = json.loads(worker.stdout.readline())
        if not evidence.get('ready'):
            raise ValueError('Native encoder did not become ready')

        def tokenize(body: dict) -> list[int]:
            """Encode exact original messages and tools using the preserved renderer."""
            worker.stdin.write(json.dumps(body)+'\n')
            worker.stdin.flush()
            response = json.loads(worker.stdout.readline())
            return response['tokens']

        try:
            prepare(args.output/'source.json', args.output/'inputs',
                    'cpu-frozen-ds-renderer', 'deepseek-v4-pro', 278528, 65536,
                    tokenize_body=tokenize)
        finally:
            worker.stdin.close()
            worker.wait(timeout=30)
    path = args.output/'inputs/manifest.json'
    manifest = json.loads(path.read_text())
    labels = {r['source_request_id']: r['cluster'] for r in source['membership']}
    for chain in manifest['chains']:
        for row in chain['rounds'][1:]:
            row['cluster'] = labels[row['source_request_id']]
    manifest.update(native_assets=evidence['assets'],
                    selection_file=str(args.selection),
                    source_candidate_exclusions=source['excluded_source_candidates'])
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(phase='native_inputs_ready',
                          formal_requests=sum(len(c['rounds'])-1 for c in manifest['chains']),
                          native_exclusions=manifest['excluded'],
                          inputs=str(path.parent))), flush=True)


if __name__ == '__main__':
    main()

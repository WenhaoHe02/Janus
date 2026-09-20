# SPDX-License-Identifier: Apache-2.0
"""Prepare the GLM pilot's exact source sessions with DS native encoding."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def render_worker() -> None:
    """Render original text and tool schemas with the frozen DS assets.

    Reads JSON request bodies on stdin and emits native token IDs on stdout.
    Raises ValueError for multimodal content or ambiguous thinking flags.
    """
    from transformers import AutoTokenizer

    assets = Path('/tmp/dsv4_native_assets_v826')
    sys.path.insert(0, str(assets))
    from encoding_dsv4 import encode_messages

    evidence = json.loads((assets / 'manifest.json').read_text())
    if evidence['flash']['hashes'] != evidence['pro']['hashes']:
        raise ValueError('Flash and Pro tokenizers differ')
    tokenizer = AutoTokenizer.from_pretrained(str(assets / 'flash'), local_files_only=True)
    print(json.dumps({'ready': True, 'assets': evidence}), flush=True)
    for line in sys.stdin:
        body = json.loads(line)
        messages = json.loads(json.dumps(body['messages']))
        for message in messages:
            content = message.get('content')
            if content is None:
                message['content'] = ''
            elif isinstance(content, list):
                if any(part.get('type') != 'text' for part in content):
                    raise ValueError('Cannot omit multimodal content')
                message['content'] = '\n'.join(part['text'] for part in content)
        if body.get('tools'):
            system = next((m for m in messages if m['role'] == 'system'), None)
            if system is None:
                system = {'role': 'system', 'content': ''}
                messages.insert(0, system)
            if system.get('tools') is not None and system['tools'] != body['tools']:
                raise ValueError('Conflicting tool schemas')
            system['tools'] = body['tools']
        thinking = (body.get('chat_template_kwargs') or {}).get('thinking', True)
        if not isinstance(thinking, bool):
            raise ValueError('Ambiguous thinking mode')
        prompt = encode_messages(messages, thinking_mode='thinking' if thinking else 'chat')
        print(json.dumps({'tokens': tokenizer.encode(prompt, add_special_tokens=False)}), flush=True)


def main() -> None:
    """Create a new replay corpus, preserving source identities and budgets.

    Preparation uses CPU only and must run outside formal timing intervals.
    Server-tokenizer verification remains required after switching to DS.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        render_worker()
        return
    from prepare_native_chain_replay_v587 import prepare

    source = args.source
    source_manifest = json.loads(source.read_text())
    with subprocess.Popen(['docker', 'exec', '-i', 'dsv4-csa-cp8-off',
                           'python3', '/tmp/prepare_ds_serving_expanded_v904.py', '--worker',
                           '--source', str(source), '--output', str(args.output)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
        assert worker.stdin is not None and worker.stdout is not None
        evidence = json.loads(worker.stdout.readline())
        if not evidence.get('ready'):
            raise ValueError('DS renderer failed')

        def tokenize(body: dict) -> list[int]:
            """Return original-body native DS IDs from the CPU renderer."""
            worker.stdin.write(json.dumps(body) + '\n')
            worker.stdin.flush()
            return json.loads(worker.stdout.readline())['tokens']

        try:
            prepare(source, args.output, 'not-yet-server-verified', 'deepseek-v4-pro',
                    278528, 65536, tokenize_body=tokenize)
            path = args.output / 'manifest.json'
            manifest = json.loads(path.read_text())
            expected = source_manifest['unbranched_chains']
            observed = [[r['source_request_id'] for r in c['rounds']] for c in manifest['chains']]
            if observed != expected or manifest.get('excluded'):
                raise ValueError('Native preparation excluded or changed selected sessions; inspect excluded list')
            if sum(len(c) - 1 for c in observed) != source_manifest['measured_request_count']:
                raise ValueError('Formal request count differs from frozen selection')
            expected_budget = {r['source_request_id']: r['source_O'] for r in source_manifest['membership']}
            membership = {r['source_request_id']: r['cluster'] for r in source_manifest['membership']}
            for chain in manifest['chains']:
                for row in chain['rounds'][1:]:
                    if row['output_budget'] != expected_budget[row['source_request_id']]:
                        raise ValueError('Original output length differs from frozen source')
                    row['cluster'] = membership[row['source_request_id']]
            manifest['cluster_counts'] = source_manifest['cluster_counts']
            manifest['measured_request_count'] = source_manifest['measured_request_count']
            manifest['source_membership'] = source_manifest['membership']
            manifest.update(native_assets=evidence['assets'],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                server_token_ids_verified=False,
                mode_policy='Original thinking boolean, default true; same DS reference encoder as v830')
            path.write_text(json.dumps(manifest, indent=2))
            print('DS_EXPANDED64_PREPARED_CPU_ONLY', flush=True)
        finally:
            worker.stdin.close()
            worker.wait(timeout=30)


if __name__ == '__main__':
    main()

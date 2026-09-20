# SPDX-License-Identifier: Apache-2.0
"""Freeze the user-approved thirteen-source internal union in each native model."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def select_manifest(manifest: dict, ids: list[str]) -> dict:
    """Return a native subset preserving round records and token offsets.

    Raises ValueError on duplicate IDs, absent IDs, or repeated timed sources.
    """
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source IDs')
    chains = [c for c in manifest['chains']
              if c['rounds'][c['first_timed_index']]['source_request_id'] in ids]
    found = [c['rounds'][c['first_timed_index']]['source_request_id'] for c in chains]
    if len(found) != len(ids) or set(found) != set(ids):
        raise ValueError('Missing or repeated timed source')
    output = dict(manifest, chains=chains, selected_count=len(ids),
                  measured_request_count=len(ids), selected_source_ids=found,
                  native_content_warmup=True, unavailable=[])
    output['selection_sha256'] = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    return output


def main() -> None:
    """Copy model-native tokens to a new subset directory after hash validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    token_path = args.source / 'tokens.i32le'
    digest = hashlib.sha256(token_path.read_bytes()).hexdigest()
    if digest != manifest['token_file_sha256']:
        raise ValueError('Original native tokens disagree with manifest')
    ids = json.loads(args.ids.read_text())['ids']
    selected = select_manifest(manifest, ids)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(token_path, args.output / 'tokens.i32le')
    (args.output / 'manifest.json').write_text(json.dumps(selected, indent=2))
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        source=str(args.source), source_manifest=str(manifest_path), original_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        token_sha256=digest, selected_source_ids=ids, retokenized=False), indent=2))
    print(json.dumps(dict(output=str(args.output), n=len(ids), token_sha256=digest)))


if __name__ == '__main__':
    main()

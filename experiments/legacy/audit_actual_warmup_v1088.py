# SPDX-License-Identifier: Apache-2.0
"""Validate native-content warmup evidence without claiming zero compilation."""

# Standard
import json
from pathlib import Path


def validate_actual_warmup(manifest: dict, prepared: dict, events: list[dict]) -> dict:
    """Check every formal input was warmed before cleanup and production seeding.

    Args:
        manifest: Actual native-token manifest used by the measured batch.
        prepared: Completed preparation proof from the first measured rate.
        events: Ordered preparation records from that same first-rate batch.
    Returns:
        Count, ordered source identities and scope of the validated evidence.
    Raises:
        ValueError: Content hashes, lengths, completeness or cleanup order differ.
    """
    if not manifest.get('native_content_warmup') or prepared.get('warmup_input_mode') != 'actual_native_content':
        raise ValueError('Actual native-content warmup was not declared')
    rows = [chain['rounds'][chain['first_timed_index']] for chain in manifest['chains']]
    ids = [row['source_request_id'] for row in rows]
    warmed = [(index, event) for index,event in enumerate(events)
              if event['phase']=='actual_content_warmup_excluded']
    cleared = [index for index,event in enumerate(events) if event['phase']=='actual_warmup_cache_cleared']
    seeded = [(index,event) for index,event in enumerate(events)
              if event['phase']=='original_predecessor_setup_excluded']
    if (not rows or len(set(ids)) != len(ids)
            or [event['source_request_id'] for _,event in warmed] != ids
            or len(cleared) != 1 or len(seeded) != len(rows)
            or max(index for index,_ in warmed) >= cleared[0]
            or min(index for index,_ in seeded) <= cleared[0]):
        raise ValueError('Incomplete or out-of-order warmup/clear/seed sequence')
    for row, (_,event) in zip(rows,warmed,strict=True):
        if (event['input_token_sha256'] != row['input_token_sha256']
                or event['completion']['prompt_tokens'] != row['prompt_tokens']
                or event['H'] != row['reusable_input_prefix_tokens']
                or event['A'] != row['new_input_tokens']):
            raise ValueError('Warmup content or H/A differs from formal input')
    return dict(count=len(rows), source_ids=ids, cleanup_before_formal_seed=True,
        actual_content_verified=True,
        limitation='Input/preparation proof only; not proof that every JIT specialization was warm')


def audit_first_rate(folder: Path) -> dict:
    """Read archived first-rate warmup and manifest; return validation evidence.

    Raises OSError or ValueError if archived evidence is missing or inconsistent.
    """
    manifest = json.loads((folder.parent/'manifest.json').read_text(encoding='utf-8'))
    prepared = json.loads((folder/'prepared.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in (folder/'preparation.jsonl').read_text(encoding='utf-8').splitlines()]
    return dict(source=str(folder), **validate_actual_warmup(manifest, prepared, events))

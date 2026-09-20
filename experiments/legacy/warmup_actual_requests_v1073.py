# SPDX-License-Identifier: Apache-2.0
"""Warm exact native requests on isolated cache keys before formal seeding."""

# Standard
from array import array
from dataclasses import asdict
import hashlib
import time
from typing import Callable


def warm_actual_requests(rows: list[dict], tokens: array, model: str, arm: str,
                        record: Callable, complete: Callable,
                        publication: Callable, disk_publication: Callable) -> None:
    """Run every actual prefix and append path once outside formal timing.

    Each input is unchanged native tokens, with an isolated cache tag. Both seed
    and full-input warmup allow writes so production save paths are exercised.
    Original formal O is untouched; this preparation generates only the helper's
    standard four tokens. Caller must clear warmup caches before formal seeding.
    """
    for row in rows:
        start = row['token_offset']
        prompt = list(tokens[start:start+row['prompt_tokens']])
        history = row['reusable_input_prefix_tokens']
        tag = 'actual_warmup_v1073:' + row['source_request_id']
        for phase, request_tokens in [('actual_prefix_seed_excluded', prompt[:history]),
                                      ('actual_content_warmup_excluded', prompt)]:
            if not request_tokens:
                continue
            began = time.time()
            response = complete(request_tokens, skip_save=False, model=model, cache_tag=tag)
            if response.request_id is None or response.prompt_tokens != len(request_tokens):
                raise RuntimeError('Actual-content warmup response does not match input')
            readiness = None
            if arm in ('k_local','no_prefetch'):
                readiness = publication('dsv4-csa-cp8-off', response.request_id,
                                        len(request_tokens), began, chunk_size=256)
            elif arm == 'ssd_only':
                readiness = disk_publication(response.request_id)
            record(dict(phase=phase, source_request_id=row['source_request_id'],
                        cache_tag=tag, H=history, A=len(prompt)-history,
                        input_token_sha256=hashlib.sha256(array('i',request_tokens).tobytes()).hexdigest(),
                        completion=asdict(response), readiness=readiness))

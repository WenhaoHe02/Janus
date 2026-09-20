# SPDX-License-Identifier: Apache-2.0
"""Warmup completions with mandatory server success and token-count evidence."""

# Standard
from dataclasses import dataclass
import json
import time
from typing import Any
import urllib.request

# Local
from serving_cache_tag_v1033 import cache_parameters


@dataclass(frozen=True)
class CompletionTiming:
    """Observed request identity, client timings and authoritative token usage."""

    request_id: str
    ttft_s: float | None
    e2e_s: float
    prompt_tokens: int
    completion_tokens: int


def stream_completion(
    prompt: list[int], *, skip_save: bool,
    endpoint: str = 'http://127.0.0.1:8000',
    model: str = 'deepseek-v4-pro',
    cache_tag: str | None = None,
) -> CompletionTiming:
    """Issue a four-token warmup and reject incomplete or failed SSE responses.

    Args:
        prompt: Native token IDs; never cropped or changed.
        skip_save: Whether to suppress suffix publication during warmup.
        endpoint: The existing experiment server's base URL.
        model: Verified served model alias, unchanged between setup and measurement.
        cache_tag: Optional LMCache tag isolating this request's cache snapshot.
    Returns:
        Client timing and usage only after an explicit successful stream end.
    Raises:
        RuntimeError: Server error, unsuccessful finish, missing DONE/usage or
            mismatch between requested and reported prompt/output token counts.
        URLError: The HTTP transport fails.
    """
    payload: dict[str, Any] = dict(
        model=model, prompt=prompt, max_tokens=4,
        temperature=0, ignore_eos=True, stream=True,
        stream_options={'include_usage': True},
        kv_transfer_params=cache_parameters(skip_save, cache_tag),
    )
    request = urllib.request.Request(
        endpoint + '/v1/completions', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    started = time.perf_counter()
    first = None
    identity = None
    usage = None
    finished = False
    saw_done = False
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                saw_done = True
                break
            chunk = json.loads(data)
            if chunk.get('error') is not None:
                raise RuntimeError(f'Warmup server error: {chunk["error"]}')
            if chunk.get('id'):
                identity = identity or chunk['id']
                if identity != chunk['id']:
                    raise RuntimeError('Warmup response request_id changed')
            if chunk.get('usage') is not None:
                usage = chunk['usage']
            for choice in chunk.get('choices') or []:
                reason = choice.get('finish_reason')
                if reason is not None:
                    if reason not in ('length', 'stop'):
                        raise RuntimeError(f'Warmup {identity} failed: finish_reason={reason}')
                    finished = True
                if choice.get('text') and first is None:
                    first = time.perf_counter()
    ended = time.perf_counter()
    if not saw_done or not finished or not identity or not isinstance(usage, dict):
        raise RuntimeError(f'Incomplete warmup {identity}: missing terminal/usage evidence')
    if usage.get('prompt_tokens') != len(prompt) or usage.get('completion_tokens') != 4:
        raise RuntimeError(f'Warmup {identity} token usage mismatch: {usage}')
    return CompletionTiming(identity, None if first is None else first - started,
                            ended - started, usage['prompt_tokens'], usage['completion_tokens'])

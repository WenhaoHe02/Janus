# SPDX-License-Identifier: Apache-2.0
"""Public HTTP/readiness operations for pre-admission Tutti warmup."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
import urllib.error
import urllib.request


def private_endpoint(endpoint: str) -> str:
    """Validate one loopback HTTP endpoint; reject credentials and external hosts."""
    url = urlparse(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Warmup requires a private loopback HTTP endpoint")
    return endpoint.rstrip("/")


def json_request(url: str, payload: dict | None = None, method: str = "GET") -> dict:
    """Read one bounded JSON response, propagating HTTP and decoding failures."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def validate_publication(
    states: list[dict], request_id: str, tokens: int, chunk_size: int = 256
) -> bool:
    """Check eight exact-generation publication replies; fail on corrupt evidence.

    Returns False for legitimate pending/not-found states. Raises ValueError for
    failed, mismatched or insufficient ready replies. Tokens are not padded.
    """
    if len(states) != 8 or not request_id or tokens < 0 or chunk_size <= 0:
        raise ValueError("Invalid publication request or rank coverage")
    expected = tokens // chunk_size * chunk_size
    for state in states:
        if state.get("status") != "success" or state.get("state") == "failed":
            raise ValueError("A rank failed publication")
        if state.get("request_id") != request_id:
            raise ValueError("Publication belongs to another request")
        if state.get("state") == "ready":
            matched = state.get("matched_request_id") or ""
            if matched != request_id and not matched.startswith(request_id + "-"):
                raise ValueError("Publication matched another generation")
            if state.get("token_count", 0) < expected:
                raise ValueError("Published prefix is incomplete")
        elif state.get("state") not in {"not_found", "pending"}:
            raise ValueError("Unknown publication state")
    return all(state["state"] == "ready" for state in states)


def read_hit_tokens(log: str, request_id: str, expected: int) -> int:
    """Require exact-request scheduler evidence that LMCache supplied the prefix."""
    matches = re.findall(
        rf"Reqid:\s*{re.escape(request_id)}(?:-[^,\s]+)?\s*,[^\n]*?"
        r"LMCache hit tokens:\s*(\d+),\s*need to load:\s*(\d+)",
        log,
    )
    pairs = {(int(hit), int(load)) for hit, load in matches}
    if len(pairs) != 1:
        raise ValueError("Missing or ambiguous cache-hit evidence")
    hit, load = pairs.pop()
    if hit < expected or (expected > 0 and load <= 0):
        raise ValueError("Warmup did not execute the required external-cache read")
    marker = re.search(rf"Reqid:\s*{re.escape(request_id)}(?:-[^,\s]+)?\s*,", log)
    interval = log[marker.end() :].split("Reqid:", 1)[0]
    for failure in (
        "proxy submit failed",
        "device owner prediction is unavailable",
        "Tutti direct load failed",
        "CUDA error",
        "Traceback (most recent call last)",
    ):
        if failure in interval:
            raise ValueError(f"Reuse path failed or fell back: {failure}")
    return hit


class StartupClient:
    """Use existing public control APIs; never upload scripts or change worker code."""

    def __init__(
        self,
        endpoint: str,
        rank_endpoints: list[str],
        model: str,
        server_log: Path,
        timeout_s: float = 900,
    ) -> None:
        """Initialize eight unique private rank endpoints and a served model alias."""
        self.endpoint = private_endpoint(endpoint)
        self.rank_endpoints = [private_endpoint(value) for value in rank_endpoints]
        if (
            len(set(self.rank_endpoints)) != 8
            or len({urlparse(value).port or 80 for value in self.rank_endpoints}) != 8
            or not model
            or timeout_s <= 0
        ):
            raise ValueError(
                "Eight unique rank endpoints and positive timeout required"
            )
        self.model = model
        self.server_log = server_log
        self.timeout_s = timeout_s

    def verify_workers(self, expected: dict[str, str]) -> list[dict]:
        """Require all eight workers to match the selected startup flags."""
        states = [json_request(url + "/env") for url in self.rank_endpoints]
        for rank, state in enumerate(states):
            wrong = [key for key, value in expected.items() if state.get(key) != value]
            if wrong:
                raise ValueError(f"Worker {rank} differs on {wrong}")
        return states

    def verify_loaded_native(self, receipt_path: Path) -> None:
        """Require actual native installation to have run in all eight workers."""
        digest = json.loads(receipt_path.read_text())["module_sha256"]
        self.verify_workers({"JANUS_LOADED_COUNTED_SHA256": digest})

    def complete(
        self,
        tokens: list[int],
        tag: str,
        output_tokens: int = 4,
        *,
        skip_save: bool = False,
    ) -> dict:
        """Run an actual seed/hit request and reject incomplete SSE/usage evidence."""
        if (
            not tokens
            or output_tokens < 2
            or re.fullmatch(r"[A-Za-z0-9_.:-]+", tag) is None
        ):
            raise ValueError("Invalid warmup request")
        payload = dict(
            model=self.model,
            prompt=tokens,
            max_tokens=output_tokens,
            temperature=0,
            ignore_eos=True,
            stream=True,
            stream_options={"include_usage": True},
            kv_transfer_params={
                "lmcache.skip_save": skip_save,
                "lmcache.tag.serving_request": tag,
            },
        )
        request = urllib.request.Request(
            self.endpoint + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        identity, usage, done, finished = None, None, False, False
        started = time.perf_counter()
        started_wall = time.time()
        first_token_ms = None
        text_chunks = []
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    done = True
                    break
                item = json.loads(value)
                if item.get("error"):
                    raise RuntimeError("Warmup server returned an error")
                if item.get("id"):
                    if identity and item["id"] != identity:
                        raise RuntimeError("Response identity changed")
                    identity = item["id"]
                if item.get("usage") is not None:
                    usage = item["usage"]
                for choice in item.get("choices") or []:
                    if choice.get("text"):
                        text_chunks.append(choice["text"])
                    if first_token_ms is None and choice.get("text"):
                        first_token_ms = (time.perf_counter() - started) * 1000
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if reason not in {"stop", "length"}:
                            raise RuntimeError("Warmup generation failed")
                        finished = True
        if not done or not finished or not identity or not isinstance(usage, dict):
            raise RuntimeError("Warmup lacks completion evidence")
        if (
            usage.get("prompt_tokens") != len(tokens)
            or usage.get("completion_tokens") != output_tokens
        ):
            raise RuntimeError("Warmup token usage mismatch")
        return dict(
            request_id=identity,
            prompt_tokens=len(tokens),
            output_tokens=output_tokens,
            ttft_ms=first_token_ms,
            duration_ms=(time.perf_counter() - started) * 1000,
            skip_save=skip_save,
            output_sha256=hashlib.sha256("".join(text_chunks).encode()).hexdigest(),
            started_wall=started_wall,
        )

    def wait_publication(
        self, request_id: str, tokens: int, restored_tokens: int = 0
    ) -> list[dict]:
        """Wait for exact publication before inspecting global writer queues."""
        deadline = time.monotonic() + self.timeout_s
        with ThreadPoolExecutor(max_workers=8) as pool:
            while time.monotonic() < deadline:
                states = list(
                    pool.map(
                        lambda url: json_request(
                            url + "/write_slack/reuse_ready",
                            {"request_id": request_id},
                            "POST",
                        ),
                        self.rank_endpoints,
                    )
                )
                if validate_publication(states, request_id, tokens):
                    self.drain()
                    return states
                required = tokens // 256 * 256
                if (
                    required > 0
                    and restored_tokens >= required
                    and all(state["state"] == "not_found" for state in states)
                ):
                    self.drain()
                    return states
                time.sleep(0.1)
        raise TimeoutError("Publication did not finish before readiness")

    def drain(self) -> None:
        """Wait for all writers with zero failures; missing fields fail closed."""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                states = [
                    json_request(url + "/write_slack/status")
                    for url in self.rank_endpoints
                ]
            except urllib.error.HTTPError as error:
                if error.code != 503:
                    raise
                time.sleep(0.1)
                continue
            fields = ("queued_waves", "active_waves", "active_io_blocks")
            if any(
                state.get("status") != "success"
                or state.get("failed_waves") != 0
                or any(key not in state for key in fields)
                for state in states
            ):
                raise RuntimeError("Missing writer state or failed writes")
            if all(state[key] == 0 for state in states for key in fields):
                return
            time.sleep(0.1)
        raise TimeoutError("Writers did not drain")

    def clear_dedicated_cache(self) -> None:
        """Clear only the explicitly dedicated pre-admission disk cache after drain."""
        self.drain()
        for url in self.rank_endpoints:
            state = json_request(
                url + "/cache/clear?locations=LocalDiskBackend", method="DELETE"
            )
            if state.get("status") != "success":
                raise RuntimeError("Dedicated warmup cache could not be cleared")

    def require_hit(self, request_id: str, history: int) -> int:
        """Read the supervisor's own server log; do not infer hits from timings."""
        hit = read_hit_tokens(
            self.server_log.read_text(errors="replace"),
            request_id,
            history // 256 * 256,
        )
        if hit > history:
            raise ValueError("Warmup reused part of the append, changing its shape")
        return hit

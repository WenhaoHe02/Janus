# SPDX-License-Identifier: Apache-2.0
"""Public contracts for native-token warmup and exact publication/read evidence."""

# Standard
from array import array
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Local
from startup_io import (
    StartupClient,
    private_endpoint,
    read_hit_tokens,
    validate_publication,
)
from warm_manifest import load_inputs, warm_cases


class FakeClient:
    """Record warmup actions without a server or GPU."""

    def __init__(self) -> None:
        self.calls = []

    def clear_dedicated_cache(self) -> None:
        self.calls.append(("clear",))

    def complete(self, tokens: list[int], tag: str) -> dict:
        self.calls.append(("complete", list(tokens), tag))
        return dict(
            request_id=f"request-{len(self.calls)}",
            prompt_tokens=len(tokens),
            output_tokens=4,
        )

    def wait_publication(
        self, request_id: str, tokens: int, restored_tokens: int = 0
    ) -> None:
        self.calls.append(("publish", request_id, tokens, restored_tokens))

    def require_hit(self, request_id: str, history: int) -> int:
        self.calls.append(("hit", request_id, history))
        return history // 256 * 256


class StartupIOTests(unittest.TestCase):
    """Never replace actual content with padded/synthetic same-length prompts."""

    def test_cache_hit_with_failed_prediction_is_not_acceptance(self) -> None:
        log = "Reqid: r-0, LMCache hit tokens: 512, need to load: 512\n"
        with self.assertRaises(ValueError):
            read_hit_tokens(log + "CSA attention-KV proxy submit failed", "r", 512)

    def test_reuse_does_not_save_append_and_records_first_text(self) -> None:
        class Response:
            def __enter__(self) -> object:
                """Yield representative OpenAI streaming events."""
                return iter(
                    [
                        b'data: {"id":"r","choices":[{"text":""}]}\n',
                        b'data: {"id":"r","choices":[{"text":"ok",'
                        b'"finish_reason":"length"}]}\n',
                        b'data: {"id":"r","choices":[],"usage":'
                        b'{"prompt_tokens":2,"completion_tokens":4}}\n',
                        b"data: [DONE]\n",
                    ]
                )

            def __exit__(self, *args: object) -> bool:
                """Do not suppress stream reader errors."""
                return False

        client = StartupClient(
            "http://127.0.0.1:8020",
            [f"http://127.0.0.1:{7300 + i}" for i in range(8)],
            "flash",
            Path("server.log"),
        )
        with patch(
            "startup_io.urllib.request.urlopen", return_value=Response()
        ) as send:
            with patch("startup_io.time.perf_counter", side_effect=[1.0, 1.2, 1.5]):
                result = client.complete([1, 2], "reuse", skip_save=True)
        self.assertTrue(
            json.loads(send.call_args.args[0].data)["kv_transfer_params"][
                "lmcache.skip_save"
            ]
        )
        self.assertAlmostEqual(result["ttft_ms"], 200)
        self.assertAlmostEqual(result["duration_ms"], 500)

    def test_publication(self) -> None:
        state = dict(
            status="success",
            state="ready",
            request_id="r",
            matched_request_id="r-0",
            token_count=512,
        )
        self.assertTrue(validate_publication([state] * 8, "r", 513))
        for changes in (
            dict(state="failed"),
            dict(request_id="other"),
            dict(token_count=100),
            dict(matched_request_id="r-otherprefix0x"),
        ):
            # A suffix beginning r- is allowed by the actual OpenAI/vLLM ID contract.
            if "matched_request_id" in changes:
                changes = dict(matched_request_id="unrelated")
            states = [state] * 7 + [dict(state, **changes)]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_publication(states, "r", 513)
        with self.assertRaises(ValueError):
            validate_publication([state] * 7, "r", 513)

    def test_pending_publication(self) -> None:
        pending = dict(status="success", state="not_found", request_id="r")
        self.assertFalse(validate_publication([pending] * 8, "r", 512))

    def test_exact_read_identity(self) -> None:
        log = (
            "Reqid: cmpl-r-0-x, Total tokens 800, "
            "LMCache hit tokens: 512, need to load: 512"
        )
        self.assertEqual(read_hit_tokens(log, "cmpl-r", 512), 512)
        for bad in (
            log.replace("cmpl-r", "cmpl-r2"),
            log.replace("need to load: 512", "need to load: 0"),
            "",
        ):
            with self.assertRaises(ValueError):
                read_hit_tokens(bad, "cmpl-r", 512)

    def test_no_remote_warmup(self) -> None:
        for value in (
            "http://remote/",
            "http://user:secret@localhost/",
            "https://localhost/",
        ):
            with self.assertRaises(ValueError):
                private_endpoint(value)

    def test_exact_tokens_and_order(self) -> None:
        tokens = array("i", range(280))
        rows = [
            dict(
                source_request_id="a",
                token_offset=3,
                prompt_tokens=276,
                reusable_input_prefix_tokens=257,
            )
        ]
        before = copy.deepcopy(rows)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            events = warm_cases(client, tokens, rows, Path(folder) / "warm", "job")
        completes = [call for call in client.calls if call[0] == "complete"]
        self.assertEqual(completes[0][1], list(tokens[3:260]))
        self.assertEqual(completes[1][1], list(tokens[3:279]))
        self.assertEqual(completes[0][2], completes[1][2])
        self.assertEqual(
            [call[0] for call in client.calls],
            ["clear", "complete", "publish", "complete", "hit", "publish", "clear"],
        )
        self.assertEqual(events[0]["append"], 19)
        self.assertEqual(rows, before)

    def test_invalid_context_rejected_before_requests(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tokens.i32le").write_bytes(array("i", range(300)).tobytes())
            row = dict(
                source_request_id="a",
                token_offset=0,
                prompt_tokens=300,
                reusable_input_prefix_tokens=257,
            )
            manifest = dict(
                selected_count=1, chains=[dict(first_timed_index=0, rounds=[row])]
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_inputs(root, 303)
            self.assertEqual(load_inputs(root, 304)[2][0], row)


if __name__ == "__main__":
    unittest.main()

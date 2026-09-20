# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract checks for source extraction and cold-start readiness."""

# Standard
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

# Local
from build_review import source_dependencies
from build_native import source_fingerprint
from cold_start import read_plan, serve, validate_receipt
from release_decode_cleanup import submission_source


def plan_fixture(port: int) -> dict:
    """Return a tiny test server plan; no GPU, SSD, or model is involved."""
    server = (
        "import http.server,threading,time;"
        f"s=http.server.HTTPServer(('127.0.0.1',{port}),http.server.BaseHTTPRequestHandler);"
        "s.RequestHandlerClass.do_GET=lambda self:"
        "(self.send_response(200),self.end_headers());"
        "threading.Thread(target=s.serve_forever,daemon=True).start();time.sleep(2)"
    )
    warmup = (
        "import os,json,pathlib;"
        "p=dict(status='complete',artifact_id=os.environ['JANUS_ARTIFACT_ID'],"
        "warmup_input_mode='actual_native_content',covered_requests=1,"
        "expected_requests=1,manifest_sha256='test-only');"
        "pathlib.Path(os.environ['JANUS_WARMUP_RECEIPT']).write_text(json.dumps(p))"
    )
    return dict(
        identity=dict(
            model_revision="test",
            gpu_arch="test-cpu",
            tp_size=1,
            precision="test",
            shape_policy="one fake request",
        ),
        build=[[sys.executable, "-c", "pass"]],
        server=[sys.executable, "-c", server],
        warmup=[sys.executable, "-c", warmup],
        verify=[sys.executable, "-c", "pass"],
        health_url=f"http://127.0.0.1:{port}/health",
        health_timeout_s=5,
        command_timeout_s=5,
    )


class SubmissionTests(unittest.TestCase):
    """Test public contracts, including preparation failure and cleanup."""

    def test_decode_bypass_removed(self) -> None:
        """Removing the bypass covers multiline and same-line flag assignments."""
        for source in (
            "flags = dict(LMCACHE_SERVING_ASSUMED_HBM_DECODE='1', arm='k')\n",
            "flags = dict(LMCACHE_SERVING_ASSUMED_HBM_DECODE='1',\n arm='k')\n",
        ):
            result = submission_source("experiments/legacy/config.py", source)
            self.assertNotIn("ASSUMED_HBM", result)
            compile(result, "config.py", "exec")
        self.assertIsNone(
            submission_source(
                "experiments/legacy/serving_decode_policy_v763.py", "old code"
            )
        )

    def test_submission_has_no_decode_bypass(self) -> None:
        """The delivered runtime/configuration tree must not contain bypass hooks."""
        root = Path(__file__).resolve().parents[2]
        legacy = root / "experiments/legacy"
        if not legacy.is_dir():
            self.skipTest("Run this check inside the exported review tree")
        for source in legacy.iterdir():
            if source.suffix in (".py", ".sh"):
                text = source.read_text(encoding="utf-8-sig")
                self.assertNotIn("ASSUMED_HBM", text, str(source))
                self.assertNotIn("skip_decode_reads", text, str(source))

    def test_native_sources_required(self) -> None:
        """Native compile inputs cannot silently come from an old shared object."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                source_fingerprint(Path(directory))

    def test_native_sources_hash_content(self) -> None:
        """Source changes invalidate the native compilation identity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "csrc").mkdir()
            for name in (
                "csrc/tutti_kv_ops.cu",
                "csrc/tutti_kv_ops.cuh",
                "device_count_bindings_v137.cpp",
            ):
                (root / name).write_text("test-only")
            before = source_fingerprint(root)
            (root / "csrc/tutti_kv_ops.cu").write_text("changed")
            self.assertNotEqual(before, source_fingerprint(root))

    def test_python_dependencies(self) -> None:
        """Local imports and CUDA source references are retained, not libraries."""
        self.assertEqual(
            source_dependencies(
                "import json\nfrom helper_v12 import main\np='/tmp/native_v2.cu'", ".py"
            ),
            {"helper_v12.py", "native_v2.cu"},
        )

    def test_shell_dependencies(self) -> None:
        """Shell startup references become source dependencies."""
        self.assertEqual(
            source_dependencies("exec bash /tmp/startup_v1.sh", ".sh"),
            {"startup_v1.sh"},
        )

    def test_invalid_python_rejected(self) -> None:
        """Broken dependency source does not silently pass extraction."""
        with self.assertRaises(SyntaxError):
            source_dependencies("def broken(", ".py")

    def test_valid_receipt(self) -> None:
        """All declared actual-content requests must be covered."""
        validate_receipt(
            dict(
                status="complete",
                artifact_id="x",
                warmup_input_mode="actual_native_content",
                covered_requests=12,
                expected_requests=12,
                manifest_sha256="hash",
            ),
            "x",
        )

    def test_mismatched_receipts(self) -> None:
        """Synthetic, partial, empty and old-plan receipts all fail closed."""
        base = dict(
            status="complete",
            artifact_id="x",
            warmup_input_mode="actual_native_content",
            covered_requests=12,
            expected_requests=12,
            manifest_sha256="hash",
        )
        for change in (
            dict(covered_requests=11),
            dict(covered_requests=0),
            dict(artifact_id="previous"),
            dict(warmup_input_mode="synthetic_shape"),
            dict(status="pending"),
            dict(manifest_sha256=""),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_receipt(dict(base, **change), "x")

    def test_plan_validation(self) -> None:
        """Reject public warmup endpoints, shell strings, or missing identity."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            for change in (
                dict(health_url="http://example.com/health"),
                dict(server="python server.py"),
                dict(identity={}),
                dict(command_timeout_s=0),
            ):
                path.write_text(json.dumps(dict(plan_fixture(8000), **change)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_plan(path)

    def test_success_and_verify_failure(self) -> None:
        """A real child process is stopped and readiness removed on each exit."""
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                plan = plan_fixture(port)
                if fail:
                    plan["verify"] = [sys.executable, "-c", "raise SystemExit(7)"]
                path, out = Path(folder) / "plan.json", Path(folder) / "run"
                path.write_text(json.dumps(plan))
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):
                        serve(path, out)
                else:
                    self.assertEqual(serve(path, out), 0)
                self.assertFalse((out / "READY.json").exists())
                self.assertTrue((out / "warmup_receipt.json").exists())
                with self.assertRaises(FileExistsError):
                    serve(path, out)

    def test_build_failure(self) -> None:
        """A failed build never starts the model server."""
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_fixture(8000)
            plan["build"] = [[sys.executable, "-c", "raise SystemExit(9)"]]
            path, out = Path(folder) / "plan.json", Path(folder) / "run"
            path.write_text(json.dumps(plan))
            with self.assertRaises(subprocess.CalledProcessError):
                serve(path, out)
            self.assertFalse((out / "server.log").exists())
            self.assertFalse((out / "READY.json").exists())


if __name__ == "__main__":
    unittest.main()

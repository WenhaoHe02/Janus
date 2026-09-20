# SPDX-License-Identifier: Apache-2.0
"""Check overlay compatibility and normalization using public contracts."""

# Standard
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

# Local
from apply_overlay import apply_overlay
from assemble_runtime import isolate_handoff_commands, normalize_runtime


class OverlayTests(unittest.TestCase):
    """Retain actual demand reads and reject unknown source before modification."""

    def test_host_namespace_commands_do_not_inherit_profiler_preload(self) -> None:
        source = """def _maybe_unmount_for_tutti(path):
    if True:
        if True:
            host_umount = subprocess.run(
                ["nsenter", path], check=False, text=True,
            )
            host_still_mounted = subprocess.run(
                ["nsenter", path], check=False, text=True,
            )
    return None
"""
        patched = isolate_handoff_commands(source)
        self.assertEqual(patched.count("env=handoff_env"), 2)
        self.assertIn('"LD_PRELOAD"', patched)
        self.assertEqual(isolate_handoff_commands(patched), patched)

    def test_backend_alias_survives_windows_source_line_endings(self) -> None:
        source = 'if True:\r\n    sys.modules["lmcache.c_ops"] = _ops\r\n'
        result = normalize_runtime("lmcache/__init__.py", source, "")
        self.assertIn("    c_ops = _ops\n", result)
        self.assertEqual(normalize_runtime("lmcache/__init__.py", result, ""), result)

    def test_dsv4_bypass_removal_preserves_phase(self) -> None:
        source = """def phase(manager, request, phases):
    # Serving v889: explicit idealized decode, never infer from Q length.
    from lmcache.integration.vllm.serving_decode_policy_v763 import skip_decode_reads
    if skip_decode_reads(phases[request] == "decode"):
        manager.set_kv_forward_phase("inactive")
        return False
    manager.set_kv_forward_phase(phases[request])
    return True
"""
        cleaned = normalize_runtime("lmcache/request_forward_phase.py", source, "")
        namespace = {}
        exec(cleaned, namespace)

        class Manager:
            """Observe only the public phase-setting contract."""

            def set_kv_forward_phase(self, phase: str) -> None:
                self.phase = phase

        manager = Manager()
        self.assertTrue(namespace["phase"](manager, "r", {"r": "decode"}))
        self.assertEqual(manager.phase, "decode")

    def test_glm_demand_read_not_skipped(self) -> None:
        source = (
            "from lmcache.integration.vllm.serving_decode_policy_v763 "
            "import skip_decode_reads\n"
            """def should_read():
    if skip_decode_reads(glm_forward_phase.is_decode()):
        return False
    return True
"""
        )
        cleaned = normalize_runtime("lmcache/adapter.py", source, "")
        namespace = {}
        exec(cleaned, namespace)
        self.assertTrue(namespace["should_read"]())

    def test_unknown_bypass_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_runtime("lmcache/a.py", "x = skip_decode_reads(True)", "")

    def test_overlay_is_checked_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "source/overlays/dsv4"
            (overlay / "vllm").mkdir(parents=True)
            target = root / "installed_vllm"
            target.mkdir()
            (target / "a.py").write_bytes(b"old")
            (overlay / "vllm/a.py").write_bytes(b"new")
            manifest = dict(
                files=[
                    dict(
                        path="vllm/a.py",
                        before_sha256=hashlib.sha256(b"old").hexdigest(),
                        after_sha256=hashlib.sha256(b"new").hexdigest(),
                    )
                ]
            )
            (overlay / "manifest.json").write_text(json.dumps(manifest))
            args = (root / "source", root / "installed_lmcache", target, "flash")
            self.assertEqual(apply_overlay(*args, check_only=True)["changed"], 1)
            self.assertEqual((target / "a.py").read_bytes(), b"old")
            self.assertEqual(apply_overlay(*args)["changed"], 1)
            self.assertEqual(apply_overlay(*args)["current"], 1)
            (target / "a.py").write_bytes(b"unexpected user edit")
            with self.assertRaises(ValueError):
                apply_overlay(*args)
            self.assertEqual((target / "a.py").read_bytes(), b"unexpected user edit")


if __name__ == "__main__":
    unittest.main()

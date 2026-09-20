# SPDX-License-Identifier: Apache-2.0
"""Verify cold-start binary selection without requiring a GPU in unit tests."""

# Standard
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

# Local
from install_native_runtime import transform
from native_runtime import SOURCE_FILES, SYMBOLS, verify_build_receipt


def receipt_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create fake binary bytes for digest validation only, never for loading."""
    source = root / "source"
    hashes = {}
    for name in SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test source")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    module = root / "test.so"
    module.write_bytes(b"not an actual extension")
    receipt = root / "BUILD.json"
    receipt.write_text(
        json.dumps(
            dict(
                module=str(module),
                module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                identity=dict(sources=hashes),
                symbols=list(SYMBOLS),
            )
        )
    )
    return receipt, source, module


class NativeRuntimeTests(unittest.TestCase):
    """Fail closed for stale binary/source identity and old worker hooks."""

    def test_matching_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            receipt, source, module = receipt_fixture(Path(folder))
            self.assertEqual(
                verify_build_receipt(receipt, source)["module"], str(module)
            )

    def test_changed_module(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            receipt, source, module = receipt_fixture(Path(folder))
            module.write_bytes(b"replaced binary")
            with self.assertRaises(ValueError):
                verify_build_receipt(receipt, source)

    def test_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            receipt, source, _ = receipt_fixture(Path(folder))
            (source / SOURCE_FILES[0]).write_bytes(b"new code")
            with self.assertRaises(ValueError):
                verify_build_receipt(receipt, source)

    def test_incomplete_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            receipt, source, _ = receipt_fixture(Path(folder))
            data = json.loads(receipt.read_text())
            data["symbols"] = []
            receipt.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify_build_receipt(receipt, source)

    def test_hook_idempotent(self) -> None:
        original = "import os\nclass TuttiDirectLoader:\n    pass\n"
        patched = transform(original)
        self.assertIn("ensure_counted_ops()", patched)
        self.assertNotIn(".so", patched)
        self.assertEqual(transform(patched), patched)

    def test_old_or_incomplete_hooks_rejected(self) -> None:
        for source in (
            "# DS_STATIC_COUNTED_RUNTIME_V365\nclass TuttiDirectLoader: pass",
            "# JANUS_VERIFIED_COUNTED_RUNTIME\nclass TuttiDirectLoader: pass",
            "class Unrelated: pass",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                transform(source)


if __name__ == "__main__":
    unittest.main()

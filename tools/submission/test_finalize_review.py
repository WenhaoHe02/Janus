# SPDX-License-Identifier: Apache-2.0
"""Verify portable source export without changing runtime semantics."""

# Standard
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

# Local
from finalize_review import normalize_line_endings


class LineEndingTests(unittest.TestCase):
    """Source manifests must match exact delivered LF bytes."""

    def test_normalization_and_idempotence(self) -> None:
        """Only CRLF sequences change; subsequent normalization is a no-op."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"#!/bin/sh\r\nprintf 'ready'\r\n"
            (root / "start.sh").write_bytes(raw)
            records = {"start.sh": dict(sha256=hashlib.sha256(raw).hexdigest())}
            self.assertEqual(normalize_line_endings(root, records), 1)
            normalized = (root / "start.sh").read_bytes()
            self.assertEqual(normalized, raw.replace(b"\r\n", b"\n"))
            self.assertEqual(records["start.sh"]["bytes"], len(normalized))
            self.assertEqual(
                records["start.sh"]["sha256"], hashlib.sha256(normalized).hexdigest()
            )
            self.assertEqual(normalize_line_endings(root, records), 0)

    def test_added_overlay_has_no_before_hash(self) -> None:
        """New GLM modules remain additions, not nonexistent base replacements."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "overlays/glm/lmcache/addition.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"value = 1\r\n")
            manifest = root / "overlays/glm/manifest.json"
            manifest.write_text(
                json.dumps(
                    dict(files=[dict(path="lmcache/addition.py", before_sha256=None)])
                )
            )
            records = {
                path.relative_to(root).as_posix(): {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
                for path in (source, manifest)
            }
            normalize_line_endings(root, records)
            row = json.loads(manifest.read_text())["files"][0]
            self.assertIsNone(row["before_sha256"])
            self.assertEqual(
                row["after_sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()

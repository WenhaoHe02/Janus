# SPDX-License-Identifier: Apache-2.0
"""Verify the shell entry delegates to the same supervisor without starting GPUs."""

# Standard
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def find_bash() -> str | None:
    """Find native Bash, using Git Bash instead of the WSL stub on Windows."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    candidate = Path(git).parent.parent / "bin/bash.exe" if git else None
    return str(candidate) if candidate and candidate.is_file() else None


class ShellLauncherTests(unittest.TestCase):
    """Help and argument errors must work from an unrelated working directory."""

    def test_delegation_and_syntax(self) -> None:
        """Preserve arguments and exit status, without running any model work."""
        bash = find_bash()
        if bash is None:
            self.skipTest("Native Bash is unavailable")
        launcher = Path(__file__).resolve().parents[2] / "start.sh"
        subprocess.run([bash, "-n", str(launcher)], check=True, timeout=15)
        with tempfile.TemporaryDirectory() as directory:
            for arguments, expected in (([], 0), (["--help"], 0), (["--invalid"], 2)):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [bash, str(launcher), *arguments],
                        cwd=directory,
                        env=dict(os.environ, PYTHON=sys.executable),
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertIn("--model", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

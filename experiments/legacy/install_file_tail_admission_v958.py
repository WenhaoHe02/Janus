# SPDX-License-Identifier: Apache-2.0
"""Install the reviewed filesystem lookup guard after legacy overlays."""

# Standard
from pathlib import Path
import shutil


def install(root: Path, engine_source: Path, helper: Path) -> None:
    """Install a single scoped admission block before worker imports.

    Args:
        root: Installed LMCache package root.
        engine_source: Reviewed repository source containing the v958 block.
        helper: Reviewed file-prefix validation module.
    Raises:
        ValueError: Deployment anchors changed or a required marker is absent.
    """
    marker = "        # FILESYSTEM_TAIL_ADMISSION_V958:"
    end = '        if self._tutti_config is None or "LocalDiskBackend" not in block_mapping:\n'
    target = root / "v1/cache_engine.py"
    text = target.read_text()
    if marker not in text:
        reference = engine_source.read_text()
        block = reference[reference.index(marker):reference.index(end, reference.index(marker))]
        if text.count(end) != 1:
            raise ValueError("Filesystem lookup admission anchor changed")
        updated = text.replace(end, block + end, 1)
        compile(updated, str(target), "exec")
        backup = target.with_suffix(".py.before_v958")
        if not backup.exists():
            backup.write_bytes(target.read_bytes())
        target.write_text(updated)
    destination = root / "v1/storage_backend/file_prefix_layout.py"
    compile(helper.read_text(), str(destination), "exec")
    shutil.copyfile(helper, destination)
    print("FILESYSTEM_TAIL_ADMISSION_V958_INSTALLED", flush=True)

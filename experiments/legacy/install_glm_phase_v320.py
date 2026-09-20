# SPDX-License-Identifier: Apache-2.0
"""Install opt-in scheduler phase telemetry without changing inference gates."""

# Standard
from pathlib import Path


def replace_once(text: str, anchor: str, replacement: str) -> str:
    """Replace one known source anchor, rejecting ambiguous deployments."""
    if text.count(anchor) != 1:
        raise RuntimeError(f"phase telemetry anchor mismatch: {anchor[:80]}")
    return text.replace(anchor, replacement, 1)


def install(root: Path) -> None:
    """Add scheduler phase metadata to the explicit measurement package.

    Args:
        root: Installed LMCache package or a disposable deployment check copy.
    Raises:
        RuntimeError: If a source anchor differs from the audited v80 base.
    """
    (root / "integration/vllm/glm_forward_phase.py").write_bytes(
        Path("/tmp/glm_forward_phase_v320.py").read_bytes()
    )
    path = root / "integration/vllm/vllm_v1_adapter.py"
    text = path.read_text()
    if "experiment_forward_phases_v320" in text:
        return
    anchor = "logger = init_logger(__name__)\n"
    text = replace_once(
        text,
        anchor,
        "from lmcache.integration.vllm import glm_forward_phase\n\n" + anchor,
    )
    anchor = (
        "class LMCacheConnectorMetadata(KVConnectorMetadata):\n"
        "    requests: list[ReqMeta] = field(default_factory=list)\n"
    )
    text = replace_once(
        text,
        anchor,
        anchor + "    experiment_forward_phases_v320: dict[str, str] = "
        "field(default_factory=dict)\n",
    )
    anchor = "        meta = LMCacheConnectorMetadata()\n"
    text = replace_once(
        text,
        anchor,
        anchor + "        meta.experiment_forward_phases_v320 = "
        "glm_forward_phase.scheduler_phases(scheduler_output)\n",
    )
    anchor = (
        '            raise ValueError(f"Expected LMCacheConnectorMetadata, '
        'got {type(metadata)}")\n'
    )
    text = replace_once(
        text,
        anchor,
        anchor
        + "        glm_forward_phase.observe("
        "metadata.experiment_forward_phases_v320)\n",
    )
    anchor = (
        '    """Send decode selections to the existing opt-in coverage observer."""\n'
    )
    text = replace_once(
        text,
        anchor,
        anchor + "    if not glm_forward_phase.is_decode():\n        return\n",
    )
    compile(text, str(path), "exec")
    path.write_text(text)
    print("GLM_SCHEDULER_PHASE_V320_INSTALLED", root, flush=True)

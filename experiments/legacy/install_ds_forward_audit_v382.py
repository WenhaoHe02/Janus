# SPDX-License-Identifier: Apache-2.0
"""Install phase-accurate read-only DS telemetry in a new container only."""

# Standard
import argparse
from pathlib import Path
import re


def replace_once(source: str, anchor: str, replacement: str) -> str:
    """Replace one audited source anchor, failing on missing/ambiguous matches."""
    if source.count(anchor) != 1:
        raise RuntimeError(f"source anchor mismatch: {anchor[:100]}")
    return source.replace(anchor, replacement, 1)


def install(root: Path, helper: Path) -> None:
    """Install diagnostic module and hooks without changing inference gates.

    Args:
        root: Installed lmcache package, or disposable source-check copy.
        helper: Phase/residency observer module source.
    Raises:
        RuntimeError: Deployed source differs from audited v381 anchors.
    """
    adapter_path = root / "integration/vllm/vllm_v1_adapter.py"
    manager_path = root / "v1/csa_attention_kv_prefetch_manager.py"
    adapter = adapter_path.read_text(encoding="utf-8")
    manager = manager_path.read_text(encoding="utf-8")
    if "ds_forward_audit_v382" in adapter or "ds_forward_audit_v382" in manager:
        raise RuntimeError("audit installer must run once on a fresh deployment")
    adapter = replace_once(
        adapter,
        "logger = init_logger(__name__)\n",
        "from lmcache.integration.vllm import ds_forward_audit_v382\n\n"
        "logger = init_logger(__name__)\n",
    )
    anchor = (
        "class LMCacheConnectorMetadata(KVConnectorMetadata):\n"
        "    requests: list[ReqMeta] = field(default_factory=list)\n"
    )
    adapter = replace_once(
        adapter,
        anchor,
        anchor + "    ds_forward_audit_v382: dict = field(default_factory=dict)\n",
    )
    anchor = "        meta = LMCacheConnectorMetadata()\n"
    adapter = replace_once(
        adapter,
        anchor,
        anchor + "        meta.ds_forward_audit_v382 = "
        "ds_forward_audit_v382.scheduler_snapshot(scheduler_output)\n",
    )
    anchor = (
        '            raise ValueError(f"Expected LMCacheConnectorMetadata, '
        'got {type(metadata)}")\n'
    )
    adapter = replace_once(
        adapter,
        anchor,
        anchor
        + "        ds_forward_audit_v382.observe(metadata.ds_forward_audit_v382)\n",
    )
    manager = replace_once(
        manager,
        "logger = init_logger(__name__)\n",
        "from lmcache.integration.vllm import ds_forward_audit_v382\n\n"
        "logger = init_logger(__name__)\n",
    )
    begin = manager.index("            def _patched_forward(*args:")
    end = manager.index("            indexer_module.forward =", begin)
    body = manager[begin:end]
    anchor = (
        '                if forward_phase == "inactive":\n'
        "                    return orig_forward(*args, **kwargs)\n"
    )
    body = replace_once(
        body,
        anchor,
        '                if forward_phase == "inactive":\n'
        "                    audit_topk = orig_forward(*args, **kwargs)\n"
        "                    ds_forward_audit_v382.record_topk(\n"
        '                        mgr, layer_id, audit_topk, stage="inactive_bypass"\n'
        "                    )\n"
        "                    return audit_topk\n",
    )
    pattern = r"(?m)^( +)return true_topk$"
    if len(re.findall(pattern, body)) != 2:
        raise RuntimeError("unexpected correction return sites")
    body = re.sub(
        pattern,
        lambda match: (
            match[1]
            + "ds_forward_audit_v382.record_topk(\n"
            + match[1]
            + '    mgr, layer_id, true_topk, stage="after_correction"\n'
            + match[1]
            + ")\n"
            + match.group(0)
        ),
        body,
    )
    anchor = "                    first_drain_ms = 0.0\n"
    body = replace_once(
        body,
        anchor,
        '                    if forward_phase in {"continuation", "decode"}:\n'
        "                        ds_forward_audit_v382.record_topk(\n"
        '                            mgr, layer_id, true_topk, stage="before_correction"\n'
        "                        )\n" + anchor,
    )
    manager = manager[:begin] + body + manager[end:]
    compile(adapter, str(adapter_path), "exec")
    compile(manager, str(manager_path), "exec")
    adapter_path.write_text(adapter, encoding="utf-8")
    manager_path.write_text(manager, encoding="utf-8")
    (root / "integration/vllm/ds_forward_audit_v382.py").write_bytes(
        helper.read_bytes()
    )
    print("DS_FORWARD_AUDIT_V382_INSTALLED", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument(
        "--helper", type=Path, default=Path("/tmp/ds_forward_audit_v382.py")
    )
    args = parser.parse_args()
    install(args.lmcache_root, args.helper)

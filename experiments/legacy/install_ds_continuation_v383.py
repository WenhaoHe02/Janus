# SPDX-License-Identifier: Apache-2.0
"""Apply only the continuation fix, preserving deployed GPU fast-path patches."""

# Standard
import argparse
import ast
from pathlib import Path

# Local
from install_ds_forward_audit_v382 import replace_once


def method_span(source: str, name: str) -> tuple[int, int]:
    """Locate one method by AST line range; reject ambiguous source layouts."""
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"ambiguous method: {name}")
    return nodes[0].lineno - 1, nodes[0].end_lineno


def install(root: Path, source_dir: Path) -> None:
    """Install the locally tested public phase contract in a fresh container.

    Args:
        root: Actual installed LMCache package or disposable source copy.
        source_dir: Saved new manager source and request_forward_phase helper.
    Raises:
        RuntimeError: Previously applied fix or changed source anchors.
    """
    ap = root / "integration/vllm/vllm_v1_adapter.py"
    mp = root / "v1/csa_attention_kv_prefetch_manager.py"
    adapter = ap.read_text(encoding="utf-8")
    manager = mp.read_text(encoding="utf-8")
    if "LMCACHE_DSV4_CONTINUATION_CORRECTION" in adapter:
        raise RuntimeError("continuation fix already installed")
    adapter = replace_once(
        adapter,
        "logger = init_logger(__name__)\n",
        "from lmcache.integration.vllm.request_forward_phase import (\n"
        "    resume_external_kv_request, scheduled_request_phases,\n)\n\n"
        "logger = init_logger(__name__)\n",
    )
    anchor = (
        "class LMCacheConnectorMetadata(KVConnectorMetadata):\n"
        "    requests: list[ReqMeta] = field(default_factory=list)\n"
    )
    adapter = replace_once(
        adapter,
        anchor,
        anchor + "    request_forward_phases: dict[str, str] = "
        "field(default_factory=dict)\n",
    )
    anchor = "        meta = LMCacheConnectorMetadata()\n"
    adapter = replace_once(
        adapter,
        anchor,
        anchor
        + '        if os.getenv("LMCACHE_DSV4_CONTINUATION_CORRECTION", "0") == "1":\n'
        "            meta.request_forward_phases = "
        "scheduled_request_phases(scheduler_output)\n",
    )
    anchor = (
        "        if _CSA_ATTENTION_KV_PREFETCH_MANAGER is None:\n"
        "            _maybe_lazy_attach_csa_prefetch(self.lmcache_engine)\n"
    )
    adapter = replace_once(
        adapter,
        anchor,
        anchor
        + '\n        if os.getenv("LMCACHE_DSV4_CONTINUATION_CORRECTION", "0") '
        '== "1":\n'
        "            resume_external_kv_request(\n"
        "                _CSA_ATTENTION_KV_PREFETCH_MANAGER,\n"
        "                metadata.request_forward_phases,\n"
        "                has_external_load=has_external_kv,\n"
        "            )\n",
    )
    local_manager = (source_dir / "csa_attention_kv_prefetch_manager.py").read_text(
        encoding="utf-8"
    )
    first, end = method_span(local_manager, "set_kv_forward_phase")
    replacement = local_manager.splitlines(keepends=True)[first:end]
    first, end = method_span(manager, "set_kv_forward_phase")
    lines = manager.splitlines(keepends=True)
    manager = "".join(lines[:first] + replacement + lines[end:])
    manager = replace_once(
        manager,
        '                    if getattr(mgr, "_kv_forward_phase", "prefill") '
        '== "decode":\n',
        '                    if getattr(mgr, "_kv_forward_phase", "prefill") not in {\n'
        '                        "prefill", "continuation"\n                    }:\n',
    )
    begin = manager.index("            def _patched_forward(*args:")
    end = manager.index("            indexer_module.forward =", begin)
    body = manager[begin:end]
    anchor = '                decode_forward = forward_phase == "decode"\n'
    body = replace_once(
        body,
        anchor,
        anchor
        + '                demand_only_forward = forward_phase in '
        '{"continuation", "decode"}\n',
    )
    if body.count("not decode_forward") != 6:
        raise RuntimeError("unexpected prediction-only gate count")
    body = body.replace("not decode_forward", "not demand_only_forward")
    # Keep GPU-counted authoritative correction available to later prefill.
    body = replace_once(
        body,
        "                        not demand_only_forward\n"
        "                        and prediction_ready\n",
        "                        not decode_forward\n"
        "                        and prediction_ready\n",
    )
    body = replace_once(
        body,
        "waiter = None if decode_forward else mgr._prediction_waiter",
        "waiter = None if demand_only_forward else mgr._prediction_waiter",
    )
    manager = manager[:begin] + body + manager[end:]
    compile(manager, str(mp), "exec")
    compile(adapter, str(ap), "exec")
    ap.write_text(adapter, encoding="utf-8")
    mp.write_text(manager, encoding="utf-8")
    (root / "integration/vllm/request_forward_phase.py").write_bytes(
        (source_dir / "request_forward_phase.py").read_bytes()
    )
    print("DS_CONTINUATION_V383_INSTALLED", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("/tmp/sources_continuation_v383")
    )
    args = parser.parse_args()
    install(args.lmcache_root, args.source_dir)

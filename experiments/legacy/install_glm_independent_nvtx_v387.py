# SPDX-License-Identifier: Apache-2.0
"""Observe GLM's existing CPU/GPU launch chain without changing I/O switches."""

# Standard
import argparse
import ast
from pathlib import Path

SPECS = {
    "v1/glm_dsa_predictive_prefetch.py": {
        "GLMDSAPhysicalPrefetchSink": {
            "submit",
            "submit_authoritative",
            "wait_for_consumer",
            "_stage_shared_predictions",
            "_submit_after_prediction",
        },
        "GLMDSAPredictivePrefetchManager": {
            "after_source_layer",
            "observe_true_topk",
            "wait_for_prediction",
        },
    },
    "integration/vllm/glm_dsa_vllm_0202.py": {
        "VLLM0202GLMDSAProxy": {"predict", "predict_async"},
    },
    "v1/csa_attention_kv_prefetch_manager.py": {
        "CSAAttentionKVPrefetchManager": {
            "submit_topk_miss_reads",
            "prepare_topk_block_selection",
            "submit_selection_miss_reads",
            "wait_for_layer",
            "drain_for_layer",
            "wait_for_pending_reads",
            "_submit_reads",
            "_issue_reads",
            "_prepare_shard_gather",
            "_finalize_shard_gather",
            "register_request_chunks",
            "_compile_indexed_layer_tables",
            "_compile_layer_major_table",
            "_issue_layer_major_read",
        },
    },
    "v1/gpu_connector/tutti_direct_loader.py": {
        "TuttiDirectLoader": {
            "load_indexed_chunks_to_hbm",
            "load_chunks_to_hbm",
            "_load_indexed_chunks_to_hbm_locked",
            "_load_chunks_to_hbm_locked",
            "_load_batch",
            "_resolve_range_ios",
            "ensure_lba_cache",
            "register_lba_cache",
        },
    },
    "v1/ssd_tp_sharded_prefetch.py": {
        "TorchDistributedShardGather": {
            "gather_owner_rows_into",
            "gather_owner_rows_gpu_into",
            "gather_rows_into",
            "_gather_owner_packet",
        },
    },
    "v1/cache_engine.py": {
        "LMCacheEngine": {"retrieve", "prepare_dsv4_streaming_retrieve"}
    },
}
MODEL_SPECS = {
    name: {"forward"}
    for name in (
        "DeepseekV2DecoderLayer",
        "DeepseekV2MLAAttention",
        "DeepseekV2MoE",
        "DeepseekV2MLP",
        "Indexer",
    )
}


def annotate(path: Path, specs: dict[str, set[str]], prefix: str) -> None:
    """Add decorators to matching class methods without rewriting their bodies.

    Args:
        path: Source in a new experiment container or disposable validation copy.
        specs: Class names mapped to methods to observe.
        prefix: GLM-specific label prefix for profile classification.
    Raises:
        RuntimeError: A requested class has no matching method or was patched twice.
    """
    source = path.read_text(encoding="utf-8")
    if "INDEPENDENT_GLM_NVTX_V387" in source:
        raise RuntimeError("independent GLM scopes already installed")
    tree = ast.parse(source)
    found = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in specs:
            methods = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in specs[node.name]
            ]
            if not methods:
                raise RuntimeError(f"no scope methods in {node.name}")
            found.extend((node.name, method) for method in methods)
    if {name for name, _ in found} != set(specs):
        raise RuntimeError(f"missing class in {path}")
    lines = source.splitlines(keepends=True)
    for name, method in sorted(found, key=lambda item: item[1].lineno, reverse=True):
        label = f"{prefix}.{name}.{method.name}"
        lines.insert(
            method.lineno - 1,
            " " * method.col_offset + f"@glm_experiment_trace_call({label!r})\n",
        )
    insertion = 1
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion = node.end_lineno
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            insertion = max(insertion, node.end_lineno)
        else:
            break
    lines.insert(
        insertion,
        "\n# INDEPENDENT_GLM_NVTX_V387\n"
        "from lmcache.v1.experiment_nvtx_v370 "
        "import trace_call as glm_experiment_trace_call\n",
    )
    patched = "".join(lines)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")
    print(
        "GLM_SCOPES_V387",
        path.name,
        [(name, method.name) for name, method in found],
        flush=True,
    )


def install(root: Path, vllm: Path, helper: Path) -> None:
    """Install GLM and common storage scopes, leaving all timing/accuracy flags off."""
    (root / "v1/experiment_nvtx_v370.py").write_bytes(helper.read_bytes())
    for relative, specs in SPECS.items():
        annotate(root / relative, specs, f"glm.{Path(relative).stem}")
    annotate(vllm / "model_executor/models/deepseek_v2.py", MODEL_SPECS, "glm.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument(
        "--helper", type=Path, default=Path("/tmp/experiment_nvtx_glm_v387.py")
    )
    args = parser.parse_args()
    install(args.lmcache_root, args.vllm_root, args.helper)

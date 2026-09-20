# SPDX-License-Identifier: Apache-2.0
"""Install opt-in native-consumer auditing into a disposable container only."""

# Standard
import argparse
from pathlib import Path

# Local
from install_glm_continuation_v325 import replace_once


def install(root: Path, vllm: Path) -> None:
    """Add native input/output samples, adapter context and public plan snapshots."""
    (root / "integration/vllm/glm_native_audit.py").write_bytes(
        Path("/tmp/glm_native_audit_v361.py").read_bytes()
    )
    manager = root / "v1/csa_attention_kv_prefetch_manager.py"
    source = manager.read_text()
    if "def snapshot_request_read_plan(" not in source:
        anchor = "    def profile_topk_residency(\n"
        method = '''    def snapshot_request_read_plan(self) -> dict[str, Any]:
        """Return CPU mapping/residency snapshots after the caller's layer gates.

        For diagnostics only; request admission must be quiescent while called.
        """
        result = {}
        for layer, state in self._layers.items():
            if not state.chunks:
                continue
            limit = int(state.chunks[-1].end_compressed_block)
            result[str(layer)] = {
                "data_ptr": state.k_cache_tensor.data_ptr(),
                "resident": state.in_pool_bitmap[:limit].cpu().clone(),
                "slbas": (state.indexed_slba_table.cpu().clone()
                          if state.indexed_slba_table is not None else None),
                "dst_rows": (state.indexed_dst_rows_table.cpu().clone()
                             if state.indexed_dst_rows_table is not None else None),
                "chunks": [{"path": c.disk_meta.path,
                            "first": c.first_compressed_block,
                            "count": c.n_compressed_blocks,
                            "physical_rows": list(c.physical_block_ids),
                            "raw_extents": [tuple(e) for e in c.raw_extents]}
                           for c in state.chunks],
            }
        return result

'''
        source = replace_once(source, anchor, method + anchor)
        compile(source, str(manager), "exec")
        manager.write_text(source)
    adapter = root / "integration/vllm/vllm_v1_adapter.py"
    source = adapter.read_text()
    if "glm_native_audit.begin(" not in source:
        anchor = "        glm_forward_phase.observe(metadata.experiment_forward_phases_v320)\n"
        source = replace_once(source, anchor, anchor + '''        if os.getenv("LMCACHE_GLM_NATIVE_AUDIT", "0") == "1":
            from lmcache.integration.vllm import glm_native_audit
            glm_native_audit.begin(metadata.requests)
''')
        anchor = "        # This callback runs after the model forward. Physical prediction,\n"
        source = replace_once(source, anchor, '''        if os.getenv("LMCACHE_GLM_NATIVE_AUDIT", "0") == "1":
            from lmcache.integration.vllm import glm_native_audit
            glm_native_audit.flush()

''' + anchor)
        compile(source, str(adapter), "exec")
        adapter.write_text(source)
    attention = vllm / "v1/attention/backends/mla/flashmla_sparse.py"
    source = attention.read_text()
    if "glm_native_audit.capture(" not in source:
        source = source.replace("import torch\n", "import os\nimport torch\n", 1)
        anchor = "        # Concatenate q if it's a tuple (ql_nope, q_pe)\n"
        source = replace_once(source, anchor, '''        audit_ticket = None
        if os.getenv("LMCACHE_GLM_NATIVE_AUDIT", "0") == "1":
            from lmcache.integration.vllm import glm_native_audit
            audit_ticket = glm_native_audit.capture(
                layer, q, kv_c_and_k_pe_cache, self.topk_indices_buffer, attn_metadata
            )

''' + anchor)
        anchor = "        return attn_out, None\n"
        source = replace_once(source, anchor, '''        if audit_ticket is not None:
            glm_native_audit.finish(audit_ticket, attn_out)
''' + anchor)
        compile(source, str(attention), "exec")
        attention.write_text(source)
    print("GLM_NATIVE_AUDIT_V361_INSTALLED", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.lmcache_root, args.vllm_root)

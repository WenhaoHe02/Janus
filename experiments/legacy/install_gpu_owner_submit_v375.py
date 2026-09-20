# SPDX-License-Identifier: Apache-2.0
"""Make the existing device-owner submission safe for fixed-width padded IDs."""

# Standard
import argparse
from pathlib import Path

# Local
from install_aligned_entry_io_v374 import method_span, replace_once


def install(root: Path) -> None:
    """Compact history IDs before issuing either device reads or KV packets.

    Args:
        root: Installed package in the newly created experiment container.
    Raises:
        RuntimeError: The supported device-owner method anchor is absent.
    """
    path = root / "v1/csa_attention_kv_prefetch_manager.py"
    source = path.read_text()
    if "GPU_OWNER_COMPACT_V375" in source:
        return
    start, end = method_span(
        source, "CSAAttentionKVPrefetchManager", "fire_predicted_device_owner_reads"
    )
    body = source[start:end]
    body = replace_once(
        body,
        "            local_capability = bool(\n",
        """            # GPU_OWNER_COMPACT_V375: do not send a padding-sized payload.
            from lmcache.v1.compact_owner_ids_v375 import compact_owner_history_ids

            if covered_end <= 0:
                raise RuntimeError("device owner has no registered history")
            with torch.cuda.device(state.k_cache_tensor.device), torch.inference_mode():
                selection_stream = torch.cuda.current_stream(
                    state.k_cache_tensor.device
                )
                selection_stream.wait_event(ready_event)
                selected_ids = compact_owner_history_ids(
                    selected_ids, covered_end, owner_block_bound
                )
                ready_event = selection_stream.record_event()
            local_capability = bool(
""",
    )
    body = replace_once(
        body,
        "            partition = BlockPartition(\n",
        """            if not getattr(self, "_gpu_owner_submit_v375_logged", False):
                self._gpu_owner_submit_v375_logged = True
                logger.info(
                    "GPU_OWNER_SUBMIT_V375_ACTIVE rank=%d world=%d "
                    "history_blocks=%d fixed_ids=%d owner_bound=%d io_queued=%s",
                    transport.rank, transport.world_size, covered_end,
                    selected_ids.numel(), owner_block_bound, local_event is not None,
                )
            partition = BlockPartition(
""",
    )
    source = source[:start] + body + source[end:]
    compile(source, str(path), "exec")
    path.write_text(source)
    (root / "v1/compact_owner_ids_v375.py").write_bytes(
        Path("/tmp/compact_owner_ids_v375.py").read_bytes()
    )
    print("GPU_OWNER_COMPACT_V375_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)

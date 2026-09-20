# SPDX-License-Identifier: Apache-2.0
"""Restore GLM chunked-prefill prediction and demand-only decode reads.

Apply to an installed copy after V320 phase telemetry. Frozen overlays are never
edited. Every prefill chunk keeps prediction; decode uses authoritative demand.
"""

# Standard
import argparse
from pathlib import Path


def replace_once(text: str, anchor: str, replacement: str) -> str:
    """Replace exactly one audited source anchor or reject the deployment."""
    if text.count(anchor) != 1:
        raise RuntimeError(f"continuation anchor mismatch: {anchor[:90]}")
    return text.replace(anchor, replacement, 1)


def install(root: Path) -> None:
    """Add optional continuation callbacks using the existing demand-read API.

    Args:
        root: Installed LMCache package or disposable source-check directory.
    Raises:
        RuntimeError: If the source differs from the frozen GLM v80 base.
    """
    sink_path = root / "v1/glm_dsa_predictive_prefetch.py"
    sink = sink_path.read_text()
    if "continuation: bool = False" not in sink:
        anchor = """    def submit_authoritative(
        self,
        target_layer: int,
        true_topk: torch.Tensor,
    ) -> None:
"""
        sink = replace_once(
            sink,
            anchor,
            anchor.replace(
                "    ) -> None:\n",
                "        *,\n        continuation: bool = False,\n    ) -> None:\n",
            ),
        )
        sink = replace_once(
            sink,
            '        """Demand-load early Full groups, or every group when prediction is off.\n',
            '        """Demand-load bootstrap groups, prediction-off or continuation forwards.\n',
        )
        anchor = """            true_topk: Authoritative token indices for the group.
        """
        sink = replace_once(
            sink,
            anchor,
            anchor
            + """    continuation: Demand-only later prefill/decode; bypass the initial
                prediction-owner exclusion without scheduling new predictions.
        """,
        )
        sink = replace_once(
            sink,
            "        if target in self._scheduled_targets and self._enable_prediction:\n",
            "        if (not continuation and target in self._scheduled_targets\n"
            "                and self._enable_prediction):\n",
        )
        anchor = "    def submit_authoritative(\n"
        helper = '''    def has_active_request(self, request_id: str) -> bool:
        """Return whether an exact scheduler request owns the live physical plan."""
        return bool(request_id) and self._attention_kv_manager.active_request_id == request_id

'''
        sink = replace_once(sink, anchor, helper + anchor)
        compile(sink, str(sink_path), "exec")
        sink_path.write_text(sink)

    hooks_path = root / "integration/vllm/glm_dsa_vllm_0202.py"
    hooks = hooks_path.read_text()
    if "continuation_observer" not in hooks:
        anchor = "        forward_enabled: Callable[[], bool] | None = None,\n"
        hooks = replace_once(
            hooks,
            anchor,
            anchor
            + """        continuation_enabled: Callable[[], bool] | None = None,
        continuation_observer: Callable[[int, torch.Tensor], None] | None = None,
""",
        )
        anchor = "        self._forward_enabled = forward_enabled\n"
        hooks = replace_once(
            hooks,
            anchor,
            anchor
            + """        self._continuation_enabled = continuation_enabled
        self._continuation_observer = continuation_observer
""",
        )
        anchor = """            if (
                enabled
                and not full_indexer_layer
                and self._consumer_waiter is not None
"""
        hooks = replace_once(
            hooks,
            anchor,
            """            if (
                (enabled or (self._continuation_enabled is not None
                             and self._continuation_enabled()))
                and not full_indexer_layer
                and self._consumer_waiter is not None
""",
        )
        anchor = """                    if self._disabled_authoritative_observer is not None:
                        self._disabled_authoritative_observer(target_layer, true_topk)
                    return result
"""
        hooks = replace_once(
            hooks,
            anchor,
            """                    if self._disabled_authoritative_observer is not None:
                        self._disabled_authoritative_observer(target_layer, true_topk)
                    if (self._continuation_enabled is not None
                            and self._continuation_enabled()
                            and self._continuation_observer is not None):
                        self._continuation_observer(target_layer, true_topk)
                    return result
""",
        )
        anchor = "        consumer_starter: Optional launch-only next-layer prefetch callback.\n"
        hooks = replace_once(
            hooks,
            anchor,
            """        continuation_enabled: Whether this single scheduler request owns a live
            physical plan needing demand-only continuation reads.
        continuation_observer: Demand-read callback, distinct from the profile-only
            disabled observer; observes official Full top-K and gates its KV.
"""
            + anchor,
        )
        compile(hooks, str(hooks_path), "exec")
        hooks_path.write_text(hooks)

    adapter_path = root / "integration/vllm/vllm_v1_adapter.py"
    adapter = adapter_path.read_text()
    if "glm_continuation_enabled_v325" not in adapter:
        anchor = "def _glm_dsa_wait_for_consumer("
        helper = '''def glm_continuation_enabled_v325() -> bool:
    """Require the exact one-request scheduler identity, not kernel shape."""
    sink = _GLM_DSA_PHYSICAL_PREFETCH_SINK
    request_id = glm_forward_phase.single_request_id()
    return sink is not None and sink.has_active_request(request_id)


def glm_continuation_authoritative_v325(
    target_layer: int, true_topk: torch.Tensor
) -> None:
    """Demand-read decode history misses, without decode proxy predictions."""
    sink = _GLM_DSA_PHYSICAL_PREFETCH_SINK
    if sink is not None and glm_continuation_enabled_v325():
        sink.submit_authoritative(target_layer, true_topk, continuation=True)


'''
        adapter = replace_once(adapter, anchor, helper + anchor)
        anchor = "            forward_enabled=_glm_dsa_forward_enabled,\n"
        adapter = replace_once(
            adapter,
            anchor,
            anchor
            + """            continuation_enabled=glm_continuation_enabled_v325,
            continuation_observer=glm_continuation_authoritative_v325,
""",
        )
        anchor = """                _set_glm_dsa_prefill_forward_active(True)
            else:
                glm_manager.cancel_active_request()
"""
        adapter = replace_once(
            adapter,
            anchor,
            """                _set_glm_dsa_prefill_forward_active(True)
            elif (glm_forward_phase.is_prefill()
                  and glm_continuation_enabled_v325()):
                # Later chunks retain the same physical plan/resident KV but
                # need fresh predictors and correction for their new queries.
                glm_manager.begin_request(glm_forward_phase.single_request_id())
                _set_glm_dsa_prefill_forward_active(True)
            else:
                glm_manager.cancel_active_request()
""",
        )
        compile(adapter, str(adapter_path), "exec")
        adapter_path.write_text(adapter)
    print("GLM_CONTINUATION_DEMAND_V325_INSTALLED", root, flush=True)


def main() -> None:
    """Install into an explicit package path only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.lmcache_root)


if __name__ == "__main__":
    main()

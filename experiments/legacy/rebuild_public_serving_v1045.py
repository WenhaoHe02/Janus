# SPDX-License-Identifier: Apache-2.0
"""User-authorized, explicitly labeled ATIF/Terminus request reconstruction."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    """Return a stable digest of reconstructed JSON content."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def assistant_text(step: dict) -> tuple[str, str]:
    """Reconstruct Terminus JSON from its saved normalized response.

    Args:
        step: One prior ATIF agent step, not the selected step's answer.
    Returns:
        Text and provenance label. Non-normalized saved text is retained verbatim.
    Raises:
        ValueError: Unsupported command schema or non-text response.
    Notes:
        ATIF bash_command entries are parsed text commands, not native API tools.
        JSON whitespace and omitted original fields cannot be recovered exactly.
    """
    message = step.get("message", "")
    if not isinstance(message, str):
        raise ValueError("Only saved text responses are supported")
    if not message.startswith(("Analysis: ", "Plan: ")):
        return message, "saved_verbatim_text"
    analysis, plan = "", ""
    if message.startswith("Analysis: "):
        analysis, separator, plan = message[10:].partition("\nPlan: ")
        if not separator:
            plan = ""
    else:
        plan = message[6:]
    commands = []
    complete = False
    for call in step.get("tool_calls") or []:
        if call["function_name"] == "bash_command":
            args = call["arguments"]
            commands.append(
                dict(keystrokes=args["keystrokes"], duration=args["duration"])
            )
        elif call["function_name"] == "mark_task_complete":
            complete = True
        else:
            raise ValueError(f"Unsupported parsed command: {call['function_name']}")
    result = dict(analysis=analysis, plan=plan, commands=commands)
    if complete:
        result["task_complete"] = True
    return json.dumps(result, ensure_ascii=False, indent=2), "normalized_to_json"


def reconstruct_request(
    trajectory: dict, step_id: int, *, interleaved_thinking: bool = False
) -> tuple[dict, dict]:
    """Rebuild the input before a selected agent answer without leaking that answer.

    Args:
        trajectory: Saved ATIF trajectory with its original rendered initial prompt.
        step_id: Target API-response step whose input is reconstructed.
        interleaved_thinking: Original Harbor setting controlling historical reasoning.
    Returns:
        OpenAI-style messages and a record of contributing source steps.
    Raises:
        ValueError: Missing target, unsupported parser or incomplete history.
    """
    if trajectory["agent"].get("extra", {}).get("parser") != "json":
        raise ValueError("Only verified Terminus JSON trajectories are supported")
    steps = trajectory["steps"]
    positions = [i for i, step in enumerate(steps) if step["step_id"] == step_id]
    if len(positions) != 1 or steps[positions[0]]["source"] != "agent":
        raise ValueError("Target must be one unique agent-response step")
    messages, used = [], []
    for step in steps[: positions[0]]:
        if (step.get("extra") or {}).get("context_management"):
            raise ValueError(
                "Compacted histories need separate recorded handoff reconstruction"
            )
        source = step["source"]
        if source in ("user", "system"):
            if not isinstance(step.get("message"), str):
                raise ValueError("Non-text incoming message")
            messages.append(dict(role=source, content=step["message"]))
            used.append(dict(step_id=step["step_id"], treatment="saved_verbatim_text"))
        elif source == "agent":
            text, treatment = assistant_text(step)
            message = dict(role="assistant", content=text)
            if interleaved_thinking and step.get("reasoning_content"):
                message["reasoning_content"] = step["reasoning_content"]
            messages.append(message)
            observations = (step.get("observation") or {}).get("results") or []
            contents = []
            for observation in observations:
                if not isinstance(observation.get("content"), str):
                    raise ValueError("Missing terminal feedback text")
                contents.append(observation["content"])
            if contents:
                messages.append(dict(role="user", content="\n".join(contents)))
            used.append(
                dict(
                    step_id=step["step_id"],
                    treatment=treatment,
                    observation_parts=len(contents),
                )
            )
        else:
            raise ValueError(f"Unsupported ATIF source: {source}")
    if not messages or messages[-1]["role"] not in ("user", "system"):
        raise ValueError("No recorded incoming message preceding target answer")
    return dict(messages=messages), dict(
        contributing_steps=used,
        selected_response_excluded=True,
        initial_prompt="original rendered benchmark prompt, retained verbatim",
        tool_protocol="Terminus text JSON commands, not native API tools",
        historical_reasoning_included=interleaved_thinking,
        exact_original_http_request=False,
        limitations=[
            "Original formatting and unrecorded parser retries are unrecoverable",
            "Provider cache boundary unknown; compute native prefix later",
        ],
    )


def main() -> None:
    """Create two15-request corpora without inference or cache writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir", type=Path, default=Path("/tmp/public_serving15_v1044")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--harbor",
        type=Path,
        default=Path("/home/zbuser02/agentic-longtraces/src/harbor/src/harbor"),
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    template = args.harbor / "agents/terminus_2/templates/terminus-json-plain.txt"
    template_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    agent_source = args.harbor / "agents/terminus_2/terminus_2.py"
    all_rows = []
    for dataset in ("terminalbench_2.1", "swe_bench_pro"):
        selection_path = args.selection_dir / f"{dataset}.json"
        selected = json.loads(selection_path.read_text())
        choices = selected["requests"]
        if len(choices) != 15 or len({r["source_id"] for r in choices}) != 15:
            raise ValueError("Expected15 distinct unchanged source requests")
        folder = args.output / dataset
        folder.mkdir()
        requests = []
        for index, choice in enumerate(choices):
            original = choice["original"]
            path = Path(original["trajectory"])
            raw = path.read_bytes()
            trajectory = json.loads(raw)
            config = json.loads((path.parent.parent / "config.json").read_text())
            kwargs = config.get("agent", {}).get("kwargs", {})
            step_id = original["step_id"]
            target = next(s for s in trajectory["steps"] if s["step_id"] == step_id)
            metrics = target["metrics"]
            observed = (
                metrics["cached_tokens"],
                metrics["prompt_tokens"],
                metrics["completion_tokens"],
            )
            if observed != (choice["H"], choice["H"] + choice["A"], choice["O"]):
                raise ValueError("Source usage changed")
            body, provenance = reconstruct_request(
                trajectory,
                step_id,
                interleaved_thinking=kwargs.get("interleaved_thinking", False),
            )
            record = dict(
                source_id=choice["source_id"],
                dataset=dataset,
                cluster=choice["cluster"],
                request_body=body,
                source_usage=dict(H=choice["H"], A=choice["A"], O=choice["O"]),
                replay_kind="authorized_trajectory_reconstruction",
                provenance=dict(
                    provenance,
                    trajectory=str(path),
                    trajectory_sha256=hashlib.sha256(raw).hexdigest(),
                    step_id=step_id,
                    template_sha256=template_hash,
                    agent_source_sha256=hashlib.sha256(
                        agent_source.read_bytes()
                    ).hexdigest(),
                ),
            )
            body_path = folder / f"case{index}.json"
            body_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            requests.append(
                dict(
                    case=index,
                    source_id=choice["source_id"],
                    cluster=choice["cluster"],
                    O=choice["O"],
                    source_H=choice["H"],
                    source_A=choice["A"],
                    reconstructed_file=str(body_path),
                    body_sha256=canonical_hash(body),
                    messages=len(body["messages"]),
                )
            )
        manifest = dict(
            dataset=dataset,
            selected_count=15,
            requests=requests,
            replay_kind="authorized_trajectory_reconstruction",
            source_selection_sha256=hashlib.sha256(
                selection_path.read_bytes()
            ).hexdigest(),
            original_output_preserved=True,
            native_tokens_pending=True,
            inference_requests=0,
        )
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        all_rows.append(
            dict(
                dataset=dataset,
                reconstructed=len(requests),
                min_output=min(r["O"] for r in requests),
                max_output=max(r["O"] for r in requests),
            )
        )
    (args.output / "COMPLETE.json").write_text(json.dumps(all_rows, indent=2))
    print(json.dumps(all_rows), flush=True)


if __name__ == "__main__":
    main()

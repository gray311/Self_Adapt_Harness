"""Convert an outer-round grpo_batch.jsonl into Weave/slime GRPO replay format.

We reuse Weave_v2's proven offline-GRPO stack (slime FSDP, Qwen3.5-9B LoRA,
already working on this cluster). Its replay row format
(actions/grpo/grpo_prep.py):

    {"messages": [...], "tools": [...], "metadata": {"advantage": A, ...}}

Our proposer rows carry the FULL H1 agent trajectory (`cat agent.yaml` ->
targeted file inspection/editing -> validate_harness -> submit_harness). We
normalize it with Weave's
common.qwen35_format.normalize_qwen35_messages — the same function Weave uses
on its own NexAU trajectories — and pass the H1 tool schemas so the chat
template renders tool definitions identically at train time.

Loss-mask note: Weave's Qwen35MultiTurnLossMaskGenerator only counts assistant
turns that are CLOSED by a later user/tool message. Tool-calling turns are
closed by their tool results; only a trailing plain-assistant turn would be
unmasked, so we append a terminal user turn ("ok") when the trajectory ends on
an assistant message.

Usage:
    python grpo_to_replay.py --rounds <round_dir> [<round_dir> ...] --out replay.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WEAVE_ROOT = Path(os.environ.get(
    "WEAVE_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/code/Weave_v2"))
H1_TOOLS_DIR = Path(__file__).resolve().parents[1] / "outer" / "harness" / "tools"
H1_AGENT_YAML = H1_TOOLS_DIR.parent / "agent.yaml"
CLOSE_TURN = {"role": "user", "content": "ok"}


def _h1_tool_schemas() -> list:
    import yaml
    schemas = []
    agent = yaml.safe_load(H1_AGENT_YAML.read_text()) or {}
    mounted = agent.get("tools") or []
    seen = set()
    for row in mounted:
        if not isinstance(row, dict) or not row.get("name") or not row.get("yaml_path"):
            raise ValueError("every mounted H1 tool needs name and yaml_path")
        name = str(row["name"])
        if name in seen:
            raise ValueError(f"duplicate mounted H1 tool: {name}")
        seen.add(name)
        f = (H1_AGENT_YAML.parent / str(row["yaml_path"])).resolve()
        try:
            f.relative_to(H1_AGENT_YAML.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"H1 tool schema escapes package: {f}") from exc
        if not f.is_file():
            raise FileNotFoundError(f"mounted H1 tool schema missing: {f}")
        doc = yaml.safe_load(f.read_text())
        if doc.get("name") != name:
            raise ValueError(
                f"mounted H1 tool {name!r} does not match schema name {doc.get('name')!r}"
            )
        schemas.append({"type": "function", "function": {
            "name": doc["name"], "description": doc.get("description", ""),
            "parameters": doc.get("input_schema", {})}})
    return schemas


def _normalizer():
    if str(WEAVE_ROOT) not in sys.path:
        sys.path.insert(0, str(WEAVE_ROOT))
    from common.qwen35_format import normalize_qwen35_messages  # noqa: E402
    return normalize_qwen35_messages


# Qwen3.5's chat template accepts only these roles. NexAU middlewares inject
# FRAMEWORK-role reminders; the policy saw them as extra instructions, so map
# them to user turns. Anything else is dropped defensively.
_ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant",
             "tool": "tool", "framework": "user"}


def _stringify_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and (
                item.get("type") == "text" or "text" in item
            ):
                parts.append(str(item.get("text") or ""))
            else:
                raise ValueError(
                    f"unsupported nested tool-result content block: {item!r}"
                )
        return "\n".join(part for part in parts if part)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _nexau_content_blocks_to_openai(msgs: list) -> list[dict]:
    """Flatten NexAU's Anthropic-style content blocks for Qwen chat templates.

    NexAU stores text, ``tool_use``, and ``tool_result`` blocks inside each
    message's ``content`` list.  Qwen3.5 accepts text/image/video content blocks
    only, so passing the raw trajectory makes its Jinja template reject the
    first tool block.  Convert tool uses to ordinary OpenAI ``tool_calls``;
    Weave's pinned normalizer then renders the exact Qwen3.5 XML action format.
    """
    output: list[dict] = []
    for index, original in enumerate(msgs):
        if not isinstance(original, dict):
            raise ValueError(f"trajectory message {index} is not an object")
        role = str(original.get("role", "")).lower()
        content = original.get("content")
        if isinstance(content, str) or content is None:
            message = {**original, "content": content or ""}
            output.append(message)
            continue
        if not isinstance(content, list):
            raise ValueError(
                f"trajectory message {index} has unsupported content type "
                f"{type(content).__name__}"
            )

        text_parts: list[str] = []
        extracted_calls: list[dict] = []
        tool_call_id: str | None = None
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                raise ValueError(
                    f"trajectory message {index} has invalid content block {block!r}"
                )
            block_type = str(block.get("type", "")).lower()
            if block_type == "text" or (not block_type and "text" in block):
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                if role != "assistant":
                    raise ValueError(
                        f"trajectory message {index}: tool_use outside assistant"
                    )
                name = str(block.get("name") or "").strip()
                arguments = block.get("input", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not name or not isinstance(arguments, dict):
                    raise ValueError(
                        f"trajectory message {index}: malformed tool_use block"
                    )
                extracted_calls.append({
                    "id": str(block.get("id") or f"nexau-call-{index}"),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            elif block_type == "tool_result":
                if role != "tool":
                    raise ValueError(
                        f"trajectory message {index}: tool_result outside tool role"
                    )
                text_parts.append(_stringify_content(block.get("content")))
                if block.get("tool_use_id") is not None:
                    tool_call_id = str(block["tool_use_id"])
            else:
                raise ValueError(
                    f"trajectory message {index} has unsupported content block "
                    f"type {block_type!r}"
                )

        message = {
            key: value
            for key, value in original.items()
            if key not in {"content", "tool_calls", "function_call",
                           "provider_specific_fields"}
        }
        message["content"] = "\n".join(part for part in text_parts if part)
        existing_calls = original.get("tool_calls") or []
        if existing_calls and not isinstance(existing_calls, list):
            raise ValueError(f"trajectory message {index}: tool_calls is not a list")
        calls = list(existing_calls) + extracted_calls
        if calls:
            message["tool_calls"] = calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        output.append(message)
    return output


def _sanitize_roles(msgs: list) -> list:
    out = []
    for m in msgs:
        role = _ROLE_MAP.get(str(m.get("role", "")).lower())
        if role is None:
            continue
        out.append({**m, "role": role})
    return out


def _group_id(row: dict) -> str:
    """Stable GRPO group key shared by every candidate for one task/round."""
    return f"r{row['round']:03d}_{row.get('task_id', 'all')}"


def convert_row(row: dict, tools: list, normalize) -> dict:
    traj = row.get("trajectory") or []
    if traj and normalize is not None:
        msgs = normalize(_nexau_content_blocks_to_openai(traj))
        # the round stores only the run's messages; ensure system turn leads
        if not msgs or str(msgs[0].get("role")).lower() != "system":
            msgs = [{"role": "system", "content": row["system"]}] + msgs
    else:  # fallback: single-turn (system, user, assistant=raw submission)
        msgs = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["response"]}]
    msgs = _sanitize_roles(msgs)
    if not msgs:
        raise ValueError("GRPO trajectory has no Qwen3.5-compatible messages")
    if msgs[-1].get("role") == "assistant":
        msgs = msgs + [CLOSE_TURN]  # close the trailing turn for the loss mask
    group_id = _group_id(row)
    return {
        "messages": msgs,
        "tools": tools,
        "metadata": {
            "advantage": row["advantage"], "reward": row["reward"],
            # Weave treats metadata.seed as a GRPO group identifier.  All K
            # candidates for one task/round therefore share it; sample_id is
            # the unique trajectory key used only for provenance.
            "seed": group_id,
            "group_id": group_id,
            "sample_id": f"{group_id}_c{row['k']:02d}",
            "round": row["round"], "k": row["k"], "task_id": row.get("task_id"),
            "valid": row["valid"], "spec_hash": row.get("spec_hash", ""),
            "tools": tools,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", nargs="+", required=True,
                    help="round dirs containing grpo_batch.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-zero", action="store_true",
                    help="keep |advantage| < eps rows (default: drop, matching Weave)")
    ap.add_argument(
        "--allow-single-turn-fallback", action="store_true",
        help=("permit raw-submission fallback when the Weave normalizer cannot "
              "be imported; forbidden by default for canonical training"),
    )
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    tools = _h1_tool_schemas()
    try:
        normalize = _normalizer()
    except Exception as e:
        if not args.allow_single_turn_fallback:
            raise RuntimeError(
                "Weave trajectory normalizer is required for canonical GRPO replay"
            ) from e
        print(f"[grpo_to_replay] WARNING: no Weave normalizer ({e}); single-turn fallback")
        normalize = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(out, "w") as f:
        for rd in args.rounds:
            for line in (Path(rd) / "grpo_batch.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                n_in += 1
                row = json.loads(line)
                if not (row.get("trajectory") or row.get("response", "").strip()):
                    continue  # nothing to train on
                if not args.keep_zero and abs(row["advantage"]) < args.eps:
                    continue
                f.write(json.dumps(convert_row(row, tools, normalize),
                                   ensure_ascii=False) + "\n")
                n_out += 1
    print(f"[grpo_to_replay] {n_in} rows in -> {n_out} trainable rows -> {out}")


if __name__ == "__main__":
    main()

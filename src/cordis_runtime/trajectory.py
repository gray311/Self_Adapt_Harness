"""Read Cordis JSONL persistence and project it to SAH message trajectories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional


PACKED_TYPES = {"text-chunks", "reasoning-chunks", "tool-call-chunks"}


def decode_storage_record(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a lossless packed DSH chunk record, if present."""

    tag = row.get("type")
    if tag not in PACKED_TYPES:
        return [row]
    data = row.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"malformed {tag} storage row")
    payload_key = "args" if tag == "tool-call-chunks" else "texts"
    members, gaps = data.get(payload_key), data.get("dt")
    if (
        not isinstance(row.get("seq0"), int)
        or not isinstance(row.get("time0"), int)
        or not isinstance(members, list)
        or not members
        or any(not isinstance(value, str) for value in members)
        or not isinstance(gaps, list)
        or any(not isinstance(value, int) for value in gaps)
        or len(gaps) != len(members) - 1
    ):
        raise ValueError(f"malformed {tag} storage row")

    sequence, timestamp = row["seq0"], row["time0"]
    events: list[dict[str, Any]] = []
    for offset, member in enumerate(members):
        if offset:
            timestamp += gaps[offset - 1]
        if tag == "text-chunks":
            chunk = {"type": "text-delta", "index": data["index"], "text": member}
        elif tag == "reasoning-chunks":
            chunk = {
                "type": "reasoning-delta",
                "index": data["index"],
                "text": member,
            }
        else:
            chunk = {
                "type": "tool-call-delta",
                "index": data["index"],
                "id": data["id"],
                "argumentsDelta": member,
            }
            if "name" in data:
                chunk["name"] = data["name"]
        events.append(
            {
                "type": "assistant/chunk",
                "seq": sequence + offset,
                "time": timestamp,
                "data": {
                    "turn": data["turn"],
                    "step": data["step"],
                    "chunk": chunk,
                },
            }
        )
    return events


def load_session_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate one Cordis ``session.jsonl``."""

    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("type") != "session":
        raise ValueError(f"{path}: missing Cordis session header")
    header = records[0]
    events = [event for row in records[1:] for event in decode_storage_record(row)]
    for expected, event in enumerate(events):
        if event.get("seq") != expected:
            raise ValueError(
                f"{path}: non-contiguous event seq at {expected}: {event.get('seq')}"
            )
    return header, events


def find_top_level_session(root: Path) -> Optional[Path]:
    """Return the newest depth-zero Cordis session under ``root``."""

    candidates: list[tuple[int, Path]] = []
    for path in sorted(Path(root).rglob("session.jsonl")):
        try:
            header, _events = load_session_log(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if header.get("delegationDepth") == 0:
            candidates.append((int(header.get("createdAt", 0)), path))
    return max(candidates, default=(0, None), key=lambda row: row[0])[1]


def _text_content(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    values = []
    for block in blocks:
        if isinstance(block, str):
            values.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "reasoning"}:
            values.append(str(block.get("text", "")))
    return "\n".join(value for value in values if value)


def _parsed_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Cordis tool-call arguments are not JSON")
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Cordis tool-call arguments must decode to an object")
    return parsed


def cordis_events_to_messages(
    events: Iterable[dict[str, Any]], *, include_plugin_messages: bool = True
) -> list[dict[str, Any]]:
    """Project durable Cordis events to SAH's legacy message shape.

    Keeping this boundary stable lets reward analysis and the existing Qwen3.5
    replay converter consume Cordis rollouts while raw Cordis JSONL remains the
    authoritative provenance artifact.
    """

    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "user/message":
            source = data.get("source") or {}
            if source.get("kind") == "plugin" and not include_plugin_messages:
                continue
            text = _text_content(data.get("content"))
            if text:
                role = "framework" if source.get("kind") == "plugin" else "user"
                messages.append({"role": role, "content": text})
            continue

        if event_type == "assistant/message":
            message = data.get("message") or {}
            blocks: list[dict[str, Any]] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind in {"text", "reasoning"}:
                    text = str(block.get("text", ""))
                    if text:
                        blocks.append({"type": "text", "text": text})
                elif kind == "tool-call":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(block.get("id", "")),
                            "name": str(block.get("name", "")),
                            "input": _parsed_arguments(block.get("arguments", "{}")),
                        }
                    )
            messages.append({"role": "assistant", "content": blocks})
            continue

        if event_type == "tool/result":
            message = data.get("message") or {}
            converted: list[dict[str, Any]] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool-result":
                    continue
                converted.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(block.get("toolCallId", "")),
                        "content": _text_content(block.get("content")),
                        **({"is_error": True} if block.get("isError") else {}),
                    }
                )
            if converted:
                messages.append({"role": "tool", "content": converted})
    return messages


def count_model_calls(events: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if event.get("type") == "assistant/message"
        and (event.get("data", {}).get("message", {}).get("source", {}) or {}).get(
            "kind"
        )
        == "model"
    )


def completed_turn(events: Iterable[dict[str, Any]]) -> bool:
    return any(
        event.get("type") == "turn/end"
        and event.get("data", {}).get("reason", {}).get("kind") == "completed"
        for event in events
    )

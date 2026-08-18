#!/usr/bin/env python3
"""Select, validate, and export the top-level Cordis session trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PACKED_TYPES = {"text-chunks", "reasoning-chunks", "tool-call-chunks"}


def decode_storage_record(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the lossless packed chunk rows used by DSH JSONL persistence."""
    tag = row.get("type")
    if tag not in PACKED_TYPES:
        return [row]
    if set(row) != {"type", "seq0", "time0", "data"}:
        raise ValueError(f"malformed {tag} storage row envelope")
    seq, timestamp, data = row["seq0"], row["time0"], row["data"]
    if not isinstance(seq, int) or seq < 0 or not isinstance(timestamp, int):
        raise ValueError(f"malformed {tag} storage row sequence/time")
    if not isinstance(data, dict):
        raise ValueError(f"malformed {tag} storage row data")
    payload_key = "args" if tag == "tool-call-chunks" else "texts"
    members, gaps = data.get(payload_key), data.get("dt")
    if (
        not isinstance(members, list)
        or not members
        or any(not isinstance(value, str) for value in members)
        or not isinstance(gaps, list)
        or any(not isinstance(value, int) for value in gaps)
        or len(gaps) != len(members) - 1
    ):
        raise ValueError(f"malformed {tag} storage row payload")
    if any(not isinstance(data.get(key), (int, float)) for key in ("turn", "step", "index")):
        raise ValueError(f"malformed {tag} storage row placement")

    events = []
    for offset, member in enumerate(members):
        if offset:
            timestamp += gaps[offset - 1]
        if tag == "text-chunks":
            chunk = {"type": "text-delta", "index": data["index"], "text": member}
        elif tag == "reasoning-chunks":
            chunk = {"type": "reasoning-delta", "index": data["index"], "text": member}
        else:
            if not isinstance(data.get("id"), str):
                raise ValueError("malformed tool-call-chunks storage row id")
            chunk = {
                "type": "tool-call-delta",
                "index": data["index"],
                "id": data["id"],
                "argumentsDelta": member,
            }
            if "name" in data:
                if not isinstance(data["name"], str):
                    raise ValueError("malformed tool-call-chunks storage row name")
                chunk["name"] = data["name"]
        events.append(
            {
                "type": "assistant/chunk",
                "seq": seq + offset,
                "time": timestamp,
                "data": {"turn": data["turn"], "step": data["step"], "chunk": chunk},
            }
        )
    return events


def load_log(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("type") != "session":
        raise ValueError(f"{path}: missing Cordis session header")
    header = rows[0]
    events = [event for row in rows[1:] for event in decode_storage_record(row)]
    for expected, event in enumerate(events):
        if event.get("seq") != expected:
            raise ValueError(
                f"{path}: non-contiguous event seq at index {expected}: {event.get('seq')}"
            )
    return header, events


def source_models(events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    for event in events:
        if event.get("type") == "request/header":
            config = event.get("data", {}).get("header", {}).get("config", {})
            if config.get("provider") and config.get("model"):
                found.append({"provider": str(config["provider"]), "model": str(config["model"])})
        if event.get("type") == "assistant/message":
            source = event.get("data", {}).get("message", {}).get("source", {})
            if source.get("kind") == "model" and source.get("provider") and source.get("model"):
                found.append({"provider": str(source["provider"]), "model": str(source["model"])})
    unique = {(row["provider"], row["model"]): row for row in found}
    return [unique[key] for key in sorted(unique)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.trajectory_root.rglob("session.jsonl")):
        header, events = load_log(path)
        if header.get("delegationDepth") == 0:
            candidates.append((header, events, path))
    if not candidates:
        raise SystemExit(f"no top-level Cordis session trajectory under {args.trajectory_root}")
    candidates.sort(key=lambda row: str(row[0].get("createdAt", "")))
    header, events, source = candidates[-1]

    types = Counter(str(event.get("type")) for event in events)
    required = {"turn/start", "request/header", "assistant/message", "turn/end"}
    missing = sorted(required.difference(types))
    if missing:
        raise SystemExit(f"trajectory is missing required events: {', '.join(missing)}")
    completed = any(
        event.get("type") == "turn/end"
        and event.get("data", {}).get("reason", {}).get("kind") == "completed"
        for event in events
    )
    if not completed:
        raise SystemExit("trajectory has no completed turn/end")
    models = source_models(events)
    if not models:
        raise SystemExit("trajectory has no model provider/source record")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in [header, *events]:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema": "sah.cordis.trajectory.v1",
        "source": str(source.resolve()),
        "trajectory": str(args.output.resolve()),
        "sha256": digest,
        "session_id": header.get("id"),
        "event_count": len(events),
        "event_types": dict(sorted(types.items())),
        "models": models,
        "completed": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"TRAJECTORY_OK path={args.output} events={len(events)} "
        f"models={','.join(row['provider'] + '/' + row['model'] for row in models)}"
    )


if __name__ == "__main__":
    main()

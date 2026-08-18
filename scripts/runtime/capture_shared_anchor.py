#!/usr/bin/env python3
"""Copy and verify exact x=1 anchor programs into an immutable run namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(args.index.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    captured = {}
    for task in args.task:
        expected = index["tasks"][task]
        source = Path(expected["source_summary"])
        payload = json.loads(source.read_text())
        rows = payload if isinstance(payload, list) else [payload]
        row = next((value for value in rows if value.get("task_id") == task), None)
        if row is None or not row.get("best_program"):
            raise SystemExit(f"anchor source has no program for {task}: {source}")
        program = str(row["best_program"])
        digest = hashlib.sha256(program.encode()).hexdigest()
        if digest != expected["program_sha256"]:
            raise SystemExit(f"anchor program hash mismatch for {task}")
        if abs(float(row["best_score"]) - float(expected["score"])) > 1e-12:
            raise SystemExit(f"anchor score mismatch for {task}")
        target = args.out_dir / f"{task}.py"
        if target.exists():
            if hashlib.sha256(target.read_text().encode()).hexdigest() != digest:
                raise SystemExit(f"refusing to replace different captured anchor: {target}")
        else:
            tmp_target = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp_target.write_text(program)
            os.replace(tmp_target, target)
        captured[task] = {
            "score": float(expected["score"]),
            "program": str(target.resolve()),
            "program_sha256": digest,
            "source_summary": str(source),
        }
    manifest = {
        "schema": "captured-shared-anchors/1.0",
        "source_index": str(args.index.resolve()),
        "source_index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "tasks": captured,
    }
    path = args.out_dir / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        for task, row in captured.items():
            old = (previous.get("tasks") or {}).get(task)
            if old is None or old.get("program_sha256") != row["program_sha256"] \
                    or abs(float(old.get("score")) - row["score"]) > 1e-12:
                raise SystemExit(
                    f"refusing to mutate captured anchor manifest for {task}: {path}"
                )
        # Preserve the original index provenance on an idempotent resume.
        manifest = previous
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)
    print(path)


if __name__ == "__main__":
    main()

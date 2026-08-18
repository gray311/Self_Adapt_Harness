#!/usr/bin/env python3
"""Collect an auditable manifest from a complete or straggler-cut TTT eval."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

try:  # importable both as ``scripts.runtime.*`` and as a directly-run script
    from scripts.runtime.hash_h2_package import h2_sha256
except ImportError:  # pragma: no cover - direct CLI path
    from hash_h2_package import h2_sha256

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "src" / "inner" / "harness"


def harness_hash() -> str:
    """Stable content hash, independent of mount alias and process cwd."""
    return h2_sha256(HARNESS)


def rows(out: Path, task: str) -> list[dict]:
    found = []
    for kdir in sorted((p for p in out.glob("k*")
                        if p.is_dir() and p.name[1:].isdigit()),
                       key=lambda p: int(p.name[1:])):
        summaries = sorted(kdir.glob("*/summary.json"))
        if not summaries:
            continue
        try:
            data = json.loads(summaries[-1].read_text())
        except Exception:
            continue
        data = data if isinstance(data, list) else [data]
        row = next((x for x in data if x.get("task_id") == task), None)
        if row and row.get("best_score") is not None and row.get("best_program"):
            found.append({"k": int(kdir.name[1:]), "summary": str(summaries[-1]),
                          "score": float(row["best_score"]),
                          "evaluations": int(row.get("evaluations") or 0),
                          "executor_model_calls": int(
                              (row.get("ledger") or {}).get("llm_calls") or 0
                          )})
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--launched", type=int, required=True)
    ap.add_argument("--worker-rc", type=int, default=0)
    ap.add_argument("--reason", default="complete")
    ap.add_argument("--min-usable", type=int, default=2)
    ap.add_argument("--batch-index", type=int)
    ap.add_argument("--seed-base", type=int, default=200000)
    ap.add_argument("--seed-stride", type=int, default=16)
    ap.add_argument("--slurm-job-id")
    ap.add_argument("--expect-h2-sha256")
    args = ap.parse_args()
    if args.seed_stride < args.target:
        raise SystemExit(
            f"seed stride {args.seed_stride} is smaller than target {args.target}"
        )

    items = rows(args.out_dir, args.task)
    if len(items) < args.min_usable:
        raise SystemExit(f"only {len(items)} usable rows; need {args.min_usable}")
    if args.launched < len(items):
        raise SystemExit(
            f"launched={args.launched} is smaller than usable={len(items)}"
        )
    batch_index = args.batch_index
    if batch_index is None:
        match = re.search(r"_u(\d+)$", args.out_dir.name)
        batch_index = int(match.group(1)) if match else None
    observed_h2 = harness_hash()
    if args.expect_h2_sha256 and observed_h2 != args.expect_h2_sha256:
        raise SystemExit(
            "fixed H2 changed during executor comparison: "
            f"expected {args.expect_h2_sha256}, observed {observed_h2}"
        )
    manifest = {
        "schema": 1, "task": args.task, "checkpoint": args.checkpoint,
        "slurm_job_id": args.slurm_job_id,
        "target": args.target, "launched": args.launched,
        "attempt_launched": args.launched, "usable": len(items),
        "evaluator_calls": sum(x["evaluations"] for x in items),
        "executor_model_calls": sum(x["executor_model_calls"] for x in items),
        "batch_best": max((x["score"] for x in items), default=None),
        "worker_rc": args.worker_rc,
        "partial": len(items) < args.target, "completion_reason": args.reason,
        "fixed_harness_sha256": observed_h2,
        "fixed_harness_hash_scheme": "canonical-h2-v1",
        "h2_batch_index": batch_index,
        "h2_decode_seed_base": args.seed_base,
        "h2_decode_seed_stride": args.seed_stride,
        "h2_decode_seeds": (
            [args.seed_base + batch_index * args.seed_stride + k
             for k in range(args.target)]
            if batch_index is not None else None
        ),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    target = args.out_dir / "eval_manifest.json"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

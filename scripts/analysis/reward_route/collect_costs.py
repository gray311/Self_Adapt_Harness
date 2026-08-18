#!/usr/bin/env python3
"""Aggregate matched inference work and actual Slurm GPU-hours by route."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROUTES = ("proposer", "context", "executor")


def job_ids(batch: dict[str, Any]) -> set[int]:
    ids = set()
    for key in ("round_job", "eval_job"):
        if batch.get(key):
            ids.add(int(batch[key]))
    for value in (batch.get("outgoing_update", {}).get("jobs") or {}).values():
        if value:
            ids.add(int(value))
    return ids


def sacct_rows(ids: set[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    command = [
        "sacct", "-X", "-n", "-P", "-j", ",".join(map(str, sorted(ids))),
        "-o", "JobIDRaw,State,ElapsedRaw,AllocTRES",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"sacct failed: {result.stderr.strip()}")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split("|", 3)
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        jid, state, elapsed, tres = fields
        match = re.search(r"(?:gres/)?gpu=(\d+)", tres)
        gpus = int(match.group(1)) if match else 0
        elapsed_seconds = int(elapsed or 0)
        rows[int(jid)] = {
            "state": state, "elapsed_seconds": elapsed_seconds,
            "allocated_gpus": gpus,
            "allocated_gpu_hours": gpus * elapsed_seconds / 3600.0,
            "alloc_tres": tres,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-sacct", action="store_true")
    args = parser.parse_args()
    effects = json.loads(args.effects.read_text())
    if effects.get("status") != "complete":
        raise RuntimeError("cost finalization requires a complete effect ledger")

    totals = {route: defaultdict(float) for route in ROUTES}
    route_jobs = {route: set() for route in ROUTES}
    for task_data in effects["tasks"].values():
        for route in ROUTES:
            batches = task_data["routes"][route]["batches"]
            for batch in batches:
                totals[route]["batches"] += 1
                totals[route]["h1_trajectories"] += int(batch["h1_trajectories"])
                totals[route]["h2_trajectories"] += int(batch["h2_trajectories"])
                totals[route]["evaluator_calls"] += int(batch["rollout"]["evaluator_calls"])
                totals[route]["terminal_summaries"] += int(
                    batch["rollout"].get("terminal_summaries") or 0
                )
                totals[route]["executor_model_calls"] += int(
                    batch["rollout"].get("executor_model_calls") or 0
                )
                totals[route]["h1_model_calls"] += int(batch.get("h1_model_calls") or 0)
                totals[route]["analyzer_model_calls"] += int(
                    (batch.get("analysis") or {}).get("model_calls") or 0
                )
                update = batch["outgoing_update"]
                totals[route]["update_opportunities"] += bool(update.get("opportunity"))
                totals[route]["eligible_updates"] += bool(update.get("eligible"))
                totals[route]["applied_updates"] += bool(update.get("applied"))
                training = update.get("training_input") or {}
                if training.get("exists") and training.get("path"):
                    totals[route]["training_input_rows"] += sum(
                        bool(line.strip()) for line in Path(training["path"]).read_text().splitlines()
                    )
                if update.get("target") == "proposer_weights":
                    totals[route]["generated_trainable_rows"] += int(
                        update.get("trainable_h1_rows") or 0
                    )
                    totals[route]["optimizer_rows"] += int(
                        update.get("optimizer_rows") or 0
                    )
                    totals[route]["archive_mixed_rows"] += int(
                        update.get("archive_mixed_rows") or 0
                    )
                    totals[route]["zero_advantage_padding_rows"] += int(
                        update.get("zero_advantage_padding_rows") or 0
                    )
                elif update.get("target") == "executor_weights":
                    # Executor replay contains distinct usable trajectories and
                    # is not padded in the inference-16 matched arm.
                    rows = int((batch.get("outgoing_update") or {}).get(
                        "training_rows", 0
                    ) or 0)
                    if not rows and training.get("exists") and training.get("path"):
                        rows = sum(
                            bool(line.strip())
                            for line in Path(training["path"]).read_text().splitlines()
                        )
                    totals[route]["generated_trainable_rows"] += rows
                    totals[route]["optimizer_rows"] += rows
                route_jobs[route].update(job_ids(batch))
    all_jobs = set().union(*route_jobs.values())
    accounting = {} if args.skip_sacct else sacct_rows(all_jobs)
    output = {
        "schema": "reward-route-costs/1.0",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "complete" if args.skip_sacct or len(accounting) == len(all_jobs) else "incomplete",
        "axis_fairness": {
            "unit": "generated agent trajectory",
            "equal_trajectory_count": True,
            "equal_evaluator_calls": False,
            "equal_tokens": False,
            "equal_flops": False,
            "post_submit_reviewer_model_calls": 0,
            "call_count_semantics": (
                "observed lower bounds from terminal summaries/manifests; "
                "failed launched trajectories may consume calls before leaving no summary"
            ),
            "training_row_semantics": (
                "generated_trainable_rows excludes proposer archive rows and "
                "zero-advantage geometry padding; optimizer_rows includes them"
            ),
        },
        "routes": {},
        "slurm_jobs": {str(key): value for key, value in sorted(accounting.items())},
    }
    for route in ROUTES:
        row = {key: int(value) for key, value in totals[route].items()}
        row["generated_agent_trajectories"] = row.get("h1_trajectories", 0) + row.get("h2_trajectories", 0)
        row["slurm_job_ids"] = sorted(route_jobs[route])
        row["allocated_gpu_hours"] = sum(
            accounting[job]["allocated_gpu_hours"]
            for job in route_jobs[route] if job in accounting
        )
        row["slurm_jobs_missing_from_sacct"] = sorted(route_jobs[route] - set(accounting))
        output["routes"][route] = row
    trajectory_counts = {
        row["generated_agent_trajectories"] for row in output["routes"].values()
    }
    if len(trajectory_counts) != 1:
        raise RuntimeError(f"trajectory budgets are not matched: {trajectory_counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

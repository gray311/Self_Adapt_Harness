#!/usr/bin/env python3
"""Freeze Slurm GPU accounting for the five-task reward-routing comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.analysis.collect import sota7_sacct as base  # noqa: E402
TASKS = (
    "eft__ahc_simpletes__ahc039",
    "eft__ahc_simpletes__ahc058",
    "adrs__eplb",
    "adrs__prism",
    "adrs__llm_sql",
)


def is_zero_allocation_submission(row: dict[str, Any]) -> bool:
    """A submitted job cancelled before Slurm granted any resources."""
    return (
        int(row.get("allocated_gpus_sacct") or 0) == 0
        and int(row.get("elapsed_seconds_sacct") or 0) == 0
        and str(row.get("start") or "") in ("", "None", "Unknown")
        and str(row.get("state") or "").startswith("CANCELLED")
    )


def add(roles: dict[str, set[str]], job: Any, role: str) -> None:
    value = str(job or "")
    if not value:
        return
    assert value.isdigit(), f"invalid GPU job {value!r} for {role}"
    roles.setdefault(value, set()).add(role)


def jobs_from_costs(
    roles: dict[str, set[str]], audit: dict[str, Any]
) -> None:
    for task in TASKS:
        for route in ("proposer", "context", "executor"):
            costs = audit["tasks"][task]["costs"][route]
            if route == "executor":
                for job_role, jobs in (costs.get("jobs") or {}).items():
                    for job in jobs:
                        add(roles, job, f"accepted:{route}:{job_role}:{task}")
            else:
                for job_role in ("outer", "train", "merge"):
                    for job in costs.get(f"{job_role}_jobs") or []:
                        add(roles, job, f"accepted:{route}:{job_role}:{task}")


def tasks_of(entry: dict[str, Any]) -> set[str]:
    tasks = set(str(task) for task in entry.get("task_ids") or [])
    if entry.get("task"):
        tasks.add(str(entry["task"]))
    return tasks


def add_registry_entries(
    roles: dict[str, set[str]], entries: list[dict[str, Any]], prefix: str
) -> None:
    selected = set(TASKS)
    for entry in entries:
        entry_tasks = tasks_of(entry)
        if entry_tasks and not entry_tasks.intersection(selected):
            continue
        route = str(entry.get("route") or "context")
        for key, value in entry.items():
            if key == "wrapper_job" or key == "cpu_controller_jobs":
                continue
            if key.endswith("_job"):
                add(roles, value, f"{prefix}:{route}:{key}")
            elif key.endswith("_jobs") and isinstance(value, list):
                for job in value:
                    add(roles, job, f"{prefix}:{route}:{key}")


def requested_roles(audit: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, set[str]] = {}
    jobs_from_costs(roles, audit)
    retries = json.loads(
        (REPO / "results/sota7_operational_retries.json").read_text()
    )
    add_registry_entries(roles, retries.get("entries") or [], "retry")
    excluded = json.loads(
        (REPO / "results/sota7_excluded_campaigns.json").read_text()
    )
    add_registry_entries(
        roles, excluded.get("entries") or [], "excluded_campaign"
    )
    superseded = json.loads(
        (REPO / "results/core_sota5_superseded_proposer_as_run.json").read_text()
    )
    assert superseded.get("status") == "immutable_pre_replacement_snapshot"
    for entry in superseded.get("entries") or []:
        for job_role, jobs in (entry.get("jobs") or {}).items():
            for job in jobs:
                add(
                    roles, job,
                    f"superseded_campaign:proposer:{job_role}:"
                    f"{entry['task_ids'][0]}",
                )
    rejections = (audit.get("analysis_required_rejection_costs") or {}).get(
        "entries"
    ) or []
    add_registry_entries(roles, rejections, "analysis_rejection")
    assert roles
    return {
        job: sorted(job_roles)
        for job, job_roles in sorted(roles.items(), key=lambda item: int(item[0]))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", default="results/score_compute_curves_sota7_final_audit.json"
    )
    parser.add_argument(
        "--out", default="results/score_compute_curves_clean5_sacct_snapshot.json"
    )
    parser.add_argument(
        "--view", default="papers/figures/score_compute_curves_clean5_final_data.json"
    )
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=30.0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text())
    view_path = Path(args.view).resolve()
    view = json.loads(view_path.read_text())
    assert tuple((view.get("tasks") or {}).keys()) == TASKS
    roles = requested_roles(audit)
    requested = list(roles)
    collected: dict[str, dict[str, Any]] = {}
    last_error = ""
    for attempt in range(1, args.attempts + 1):
        missing = [job for job in requested if job not in collected]
        if not missing:
            break
        print(
            f"attempt {attempt}/{args.attempts}: querying {len(missing)} jobs",
            flush=True,
        )
        for offset in range(0, len(missing), 100):
            rows, error = base.query(missing[offset:offset + 100])
            collected.update(rows)
            if error:
                last_error = error
        if len(collected) < len(requested) and attempt < args.attempts:
            time.sleep(args.delay_seconds)

    missing = [job for job in requested if job not in collected]
    active = {
        job: row["state"] for job, row in collected.items()
        if str(row["state"]).split("+", 1)[0] in base.ACTIVE_STATES
    }
    zero_allocation = {
        job: row["state"] for job, row in collected.items()
        if is_zero_allocation_submission(row)
    }
    zero_gpu = {
        job: row["alloc_tres"] for job, row in collected.items()
        if int(row["allocated_gpus_sacct"]) <= 0
        and job not in zero_allocation
    }
    for job, row in collected.items():
        row["roles"] = roles[job]
        assert math.isclose(
            float(row["allocated_gpu_hours_sacct"]),
            int(row["elapsed_seconds_sacct"])
            * int(row["allocated_gpus_sacct"]) / 3600.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    complete = not missing and not active and not zero_gpu
    if not complete and not args.allow_partial:
        raise SystemExit(
            f"refusing incomplete clean5 snapshot: missing={missing}, "
            f"active={active}, zero_gpu={zero_gpu}, service={last_error}"
        )
    payload = {
        "schema": 1,
        "status": "complete" if complete else "partial_diagnostic",
        "task_scope": list(TASKS),
        "source_audit": str(audit_path),
        "source_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "source_clean5_view": str(view_path),
        "source_clean5_view_sha256": hashlib.sha256(
            view_path.read_bytes()
        ).hexdigest(),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_semantics": (
            "sacct -X top-level JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End"
        ),
        "cost_scope": (
            "accepted clean-five protocol jobs plus selected-task retries, "
            "excluded campaigns, analyzer rejections, and superseded "
            "AHC039/EPLB proposer campaigns"
        ),
        "requested_jobs": requested,
        "rows": {job: collected[job] for job in sorted(collected, key=int)},
        "missing_jobs": missing,
        "active_jobs": active,
        "zero_allocation_submissions": zero_allocation,
        "zero_gpu_jobs": zero_gpu,
        "last_service_error": last_error or None,
    }
    base.atomic_write(Path(args.out).resolve(), payload)
    print(
        f"wrote {args.out}: {len(collected)}/{len(requested)} jobs, "
        f"status={payload['status']}"
    )


if __name__ == "__main__":
    main()

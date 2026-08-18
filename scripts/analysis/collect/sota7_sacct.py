#!/usr/bin/env python3
"""Freeze authoritative Slurm accounting for the final SOTA7 audit.

The input is an allow-incomplete audit created after every campaign job has
closed.  This collector queries top-level jobs only (``sacct -X``), retries a
temporarily unavailable accounting service, and writes no snapshot unless all
requested GPU jobs have authoritative rows with terminal states and nonzero GPU
allocations.  ``--allow-partial`` is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ACTIVE_STATES = {
    "CONFIGURING", "COMPLETING", "PENDING", "REQUEUED",
    "REQUEUE_FED", "REQUEUE_HOLD", "RESIZING", "RUNNING", "SIGNALING",
    "SPECIAL_EXIT", "STAGE_OUT", "SUSPENDED",
}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def requested_jobs(audit: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, set[str]] = {}
    for route in ("proposer", "context", "executor"):
        ledger = audit["compute_timing_proxy"][route]
        for row in ledger["jobs"]:
            job = str(row["job"])
            roles.setdefault(job, set()).add(
                f"accepted:{route}:{row.get('role') or 'unknown'}"
            )
        retry = audit["operational_retry_costs"]["timing"][route]
        for row in retry["jobs"]:
            job = str(row["job"])
            roles.setdefault(job, set()).add(
                f"retry:{route}:{row.get('role') or 'unknown'}"
            )
    rejected = (audit.get("analysis_required_rejection_costs") or {}).get(
        "timing"
    ) or {}
    for row in rejected.get("jobs") or []:
        job = str(row["job"])
        roles.setdefault(job, set()).add(
            f"analysis_rejection:context:{row.get('role') or 'unknown'}"
        )
    excluded = (audit.get("excluded_campaign_costs") or {}).get("timing") or {}
    for route, ledger in excluded.items():
        for row in ledger.get("jobs") or []:
            job = str(row["job"])
            roles.setdefault(job, set()).add(
                f"excluded_campaign:{route}:{row.get('role') or 'unknown'}"
            )
    assert roles and all(job.isdigit() for job in roles)
    return {
        job: sorted(job_roles) for job, job_roles in
        sorted(roles.items(), key=lambda item: int(item[0]))
    }


def query(chunk: list[str]) -> tuple[dict[str, dict[str, Any]], str]:
    proc = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(chunk),
            "-o", "JobIDRaw,State,ElapsedRaw,AllocTRES%160,Start,End",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return {}, proc.stderr.strip()
    wanted = set(chunk)
    rows: dict[str, dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        job, state, elapsed_text, alloc_tres, start, end = fields[:6]
        if job not in wanted or not elapsed_text.isdigit():
            continue
        match = re.search(
            r"(?:^|,)gres/gpu(?::[^=,]+)?=([0-9]+)(?:,|$)", alloc_tres
        )
        allocated_gpus = int(match.group(1)) if match else 0
        elapsed = int(elapsed_text)
        rows[job] = {
            "job": job,
            "state": state,
            "start": start,
            "end": end,
            "elapsed_seconds_sacct": elapsed,
            "alloc_tres": alloc_tres,
            "allocated_gpus_sacct": allocated_gpus,
            "allocated_gpu_hours_sacct": allocated_gpus * elapsed / 3600.0,
        }
    return rows, proc.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", default="results/score_compute_curves_sota7_audit.json"
    )
    parser.add_argument(
        "--out", default="results/sota7_sacct_snapshot.json"
    )
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=30.0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.delay_seconds < 0:
        raise SystemExit("attempts must be positive and delay nonnegative")

    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text())
    roles = requested_jobs(audit)
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
            chunk = missing[offset:offset + 100]
            try:
                rows, error = query(chunk)
            except (OSError, subprocess.TimeoutExpired) as exc:
                rows, error = {}, f"{type(exc).__name__}: {exc}"
            collected.update(rows)
            if error:
                last_error = error
        if len(collected) < len(requested) and attempt < args.attempts:
            time.sleep(args.delay_seconds)

    missing = [job for job in requested if job not in collected]
    active = {
        job: row["state"] for job, row in collected.items()
        if str(row["state"]).split("+", 1)[0] in ACTIVE_STATES
    }
    zero_gpu = {
        job: row["alloc_tres"] for job, row in collected.items()
        if int(row["allocated_gpus_sacct"]) <= 0
    }
    for job, row in collected.items():
        assert math.isclose(
            float(row["allocated_gpu_hours_sacct"]),
            int(row["elapsed_seconds_sacct"]) *
            int(row["allocated_gpus_sacct"]) / 3600.0,
            rel_tol=1e-12, abs_tol=1e-12,
        )
        row["roles"] = roles[job]

    complete = not missing and not active and not zero_gpu
    if not complete and not args.allow_partial:
        print(
            json.dumps({
                "missing": missing,
                "active": active,
                "zero_gpu": zero_gpu,
                "last_error": last_error,
            }, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(
            "refusing to write a publishable snapshot with incomplete, active, "
            "or zero-GPU accounting rows"
        )

    payload = {
        "schema": 1,
        "status": "complete" if complete else "partial_diagnostic",
        "source_audit": str(audit_path),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_semantics": (
            "sacct -X top-level JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End"
        ),
        "requested_jobs": requested,
        "rows": {
            job: collected[job]
            for job in sorted(collected, key=int)
        },
        "missing_jobs": missing,
        "active_jobs": active,
        "zero_gpu_jobs": zero_gpu,
        "last_service_error": last_error or None,
    }
    atomic_write(Path(args.out).resolve(), payload)
    print(
        f"wrote {args.out}: {len(collected)}/{len(requested)} jobs, "
        f"status={payload['status']}"
    )


if __name__ == "__main__":
    main()

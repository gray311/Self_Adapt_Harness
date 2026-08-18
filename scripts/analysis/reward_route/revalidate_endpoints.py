#!/usr/bin/env python3
"""Repeated evaluator-only validation of reward-route endpoint programs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from inner.tasks.eft_task import get_task  # noqa: E402
from inner.evaluation.eval_runner import evaluate_program  # noqa: E402
sys.path.insert(0, str(REPO))
from scripts.runtime.hash_h2_package import h2_sha256  # noqa: E402


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in runs if row.get("score") is not None]
    result: dict[str, Any] = {
        "attempts": len(runs),
        "valid_runs": len(scores),
        "errors": len(runs) - len(scores),
    }
    if scores:
        result.update({
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        })
    return result


def evaluate_once(
    task_id: str,
    program: str,
    *,
    timeout: int,
    run_index: int,
) -> dict[str, Any]:
    """Evaluate one program once; safe to execute for distinct programs in parallel."""
    task = get_task(task_id)
    started = time.time()
    try:
        evaluated = evaluate_program(task, program, timeout_s=timeout)
        error = evaluated.error
        if not evaluated.valid:
            error = error or f"validity={evaluated.validity:g}"
        score = float(evaluated.combined_score) if evaluated.valid else None
        metrics = evaluated.metrics
    except Exception as exc:  # preserve failure evidence and continue
        error, score, metrics = f"{type(exc).__name__}: {exc}", None, None
    return {
        "run": run_index,
        "score": score,
        "metrics": metrics,
        "error": error,
        "wall_seconds": time.time() - started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "maximum distinct endpoint programs evaluated concurrently; "
            "repeats of one program remain sequential"
        ),
    )
    args = parser.parse_args()
    if args.n_runs < 1:
        raise SystemExit("--n-runs must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    cases_path = Path(args.cases).resolve()
    source = json.loads(cases_path.read_text())
    if source.get("protocol") != "reward-route-inference16-v1":
        raise RuntimeError("endpoint manifest has the wrong protocol")
    source_digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    revalidator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        result = json.loads(out.read_text())
        if result.get("source_cases_sha256") != source_digest:
            raise RuntimeError("refusing to resume against a changed endpoint manifest")
        if int(result.get("requested_runs") or 0) != args.n_runs:
            raise RuntimeError("refusing to resume with a different --n-runs")
        if result.get("revalidator_sha256") != revalidator_digest:
            raise RuntimeError("refusing to resume after revalidator code changed")
    else:
        result = {
            "schema": 1,
            "source_cases": str(cases_path),
            "source_cases_sha256": source_digest,
            "revalidator_sha256": revalidator_digest,
            "requested_runs": args.n_runs,
            "timeout_seconds": args.timeout,
            "max_distinct_program_workers": args.workers,
            "unique_program_results": {},
            "case_results": {},
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for case in source["cases"]:
        program = str(case.get("program") or "")
        if hashlib.sha256(program.encode()).hexdigest() != case.get("program_sha256"):
            raise RuntimeError(f"program hash mismatch in {case.get('case_id')}")
        package = Path(str(case.get("h2_package") or ""))
        if not package.is_dir() or h2_sha256(package) != case.get("h2_sha256"):
            raise RuntimeError(f"H2 package/hash mismatch in {case.get('case_id')}")
        key = (str(case["task"]), str(case["program_sha256"]))
        unique.setdefault(key, case)

    cases_by_key: dict[str, dict[str, Any]] = {}
    for (task_id, digest), case in unique.items():
        key = f"{task_id}::{digest}"
        cases_by_key[key] = case
        entry = result["unique_program_results"].setdefault(key, {
            "task": task_id,
            "program_sha256": digest,
            "source_summary": case["source_summary"],
            "historical_source_summary": case.get("historical_source_summary"),
            "h2_package": case.get("h2_package"),
            "h2_sha256": case.get("h2_sha256"),
            "h2_provenance_role": case.get("h2_provenance_role"),
            "program_origin_h2_available": case.get(
                "program_origin_h2_available"
            ),
            "program_origin_h2_sha256": case.get(
                "program_origin_h2_sha256"
            ),
            "program_origin_caveat": case.get("program_origin_caveat"),
            "executor_checkpoint": case.get("executor_checkpoint"),
            "runs": [],
        })
        # Refuse a malformed partial file with duplicate/out-of-order runs.
        assert [int(row["run"]) for row in entry["runs"]] == list(
            range(1, len(entry["runs"]) + 1)
        )

    def submit_next(
        pool: concurrent.futures.ThreadPoolExecutor,
        pending: dict[concurrent.futures.Future[dict[str, Any]], str],
        key: str,
    ) -> None:
        entry = result["unique_program_results"][key]
        if len(entry["runs"]) >= args.n_runs:
            return
        task_id, _ = key.split("::", 1)
        case = cases_by_key[key]
        future = pool.submit(
            evaluate_once,
            task_id,
            str(case["program"]),
            timeout=args.timeout,
            run_index=len(entry["runs"]) + 1,
        )
        pending[future] = key

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, len(cases_by_key))
    ) as pool:
        pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        for key in sorted(cases_by_key):
            submit_next(pool, pending, key)
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                key = pending.pop(future)
                task_id, digest = key.split("::", 1)
                row = future.result()
                entry = result["unique_program_results"][key]
                assert int(row["run"]) == len(entry["runs"]) + 1
                entry["runs"].append(row)
                entry["statistics"] = stats(entry["runs"])
                atomic_write(out, result)
                print(
                    f"[{task_id} {digest[:10]}] {row['run']}/{args.n_runs}: "
                    f"score={row['score']} error={row['error']}", flush=True,
                )
                submit_next(pool, pending, key)

    for case in source["cases"]:
        key = f"{case['task']}::{case['program_sha256']}"
        validation = result["unique_program_results"][key]
        summary = validation["statistics"]
        target = float(case["reported_curve_endpoint_score"])
        result["case_results"][case["case_id"]] = {
            "task": case["task"],
            "method": case["method"],
            "program_sha256": case["program_sha256"],
            "reported_curve_endpoint_score": target,
            "source_summary": case["source_summary"],
            "historical_source_summary": case.get("historical_source_summary"),
            "endpoint_program_resolution": case.get("endpoint_program_resolution"),
            "h2_package": case.get("h2_package"),
            "h2_sha256": case.get("h2_sha256"),
            "h2_provenance_role": case.get("h2_provenance_role"),
            "program_origin_h2_available": case.get(
                "program_origin_h2_available"
            ),
            "program_origin_h2_sha256": case.get(
                "program_origin_h2_sha256"
            ),
            "program_origin_caveat": case.get("program_origin_caveat"),
            "proposer_checkpoint": case.get("proposer_checkpoint"),
            "executor_checkpoint": case.get("executor_checkpoint"),
            "statistics": summary,
            "reported_endpoint_inside_revalidation_range": bool(
                summary.get("valid_runs")
                and float(summary["min"]) - 1e-12 <= target
                <= float(summary["max"]) + 1e-12
            ),
            "mean_minus_reported": (
                float(summary["mean"]) - target
                if summary.get("valid_runs") else None
            ),
            "reported_endpoint_validation_caveat": case.get(
                "reported_endpoint_validation_caveat"
            ),
        }

    result["status"] = "complete"
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["all_runs_valid"] = all(
        int(row["statistics"]["valid_runs"]) == args.n_runs
        for row in result["unique_program_results"].values()
    )
    atomic_write(out, result)
    print(f"wrote complete validation {out}")


if __name__ == "__main__":
    main()

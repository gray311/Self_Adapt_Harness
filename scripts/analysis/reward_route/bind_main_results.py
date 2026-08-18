#!/usr/bin/env python3
"""Create the sole main-result registry from N>=5 endpoint revalidation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PROTOCOL = "reward-route-inference16-v1"
METHODS = ("proposer", "context", "executor")


def display_value(task: str, combined: float) -> float:
    if task == "eft__math__erdos_min_overlap":
        return 0.380922 / combined
    if task == "eft__math__second_autocorr_ineq":
        return combined * 0.896280
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "complete" or not validation.get("all_runs_valid"):
        raise RuntimeError("main results require a complete all-valid revalidation")
    if int(validation.get("requested_runs") or 0) < 5:
        raise RuntimeError("main results require at least five validation runs")
    human = json.loads(args.human.read_text())["tasks"]
    tasks: dict[str, dict] = {}
    for case_id, case in validation["case_results"].items():
        task, method = case["task"], case["method"]
        if method not in METHODS:
            continue
        stats = case["statistics"]
        combined = float(stats["mean"])
        tasks.setdefault(task, {})[method] = {
            "combined_score_mean": combined,
            "combined_score_std": float(stats["std"]),
            "display_value": display_value(task, combined),
            "human_normalized": combined / float(human[task]["human_best_combined_score"]),
            "validation_runs": int(stats["valid_runs"]),
            "program_sha256": case["program_sha256"],
            "h2_sha256": case.get("h2_sha256"),
            "h2_provenance_role": case.get("h2_provenance_role"),
            "program_origin_h2_available": case.get(
                "program_origin_h2_available"
            ),
            "program_origin_h2_sha256": case.get(
                "program_origin_h2_sha256"
            ),
            "program_origin_caveat": case.get("program_origin_caveat"),
            "search_endpoint_score": float(case["reported_curve_endpoint_score"]),
        }
    if any(set(routes) != set(METHODS) for routes in tasks.values()) or len(tasks) != 4:
        raise RuntimeError("expected exactly four tasks x three methods")
    payload = {
        "schema": "reward-route-main-results/1.0",
        "status": "complete",
        "protocol": PROTOCOL,
        "binder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "authoritative_value": "endpoint_revalidation_mean",
        "validation": str(args.validation.resolve()),
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"wrote authoritative main-result registry: {args.out}")


if __name__ == "__main__":
    main()

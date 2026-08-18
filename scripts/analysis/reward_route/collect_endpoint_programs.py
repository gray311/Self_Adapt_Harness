#!/usr/bin/env python3
"""Bind every inference16 curve endpoint to the exact program and H2 bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.runtime.hash_h2_package import h2_sha256  # noqa: E402

PROTOCOL = "reward-route-inference16-v1"
TASKS = (
    ("erdos", "eft__math__erdos_min_overlap", 2000, 3000),
    ("ac2", "eft__math__second_autocorr_ineq", 2100, 3100),
    ("hadamard", "eft__math__hadamard_maximal_det", 2200, 3200),
    ("eplb", "adrs__eplb", 2300, 3300),
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tolerance(task: str) -> float:
    return 1e-8 if task.startswith("adrs__") else 1e-10


def anchor_case(
    captured_manifest: Path, task: str,
) -> tuple[str, float, str, str | None]:
    """Read only the immutable anchor copy captured inside this run."""

    capture = json.loads(captured_manifest.read_text())
    record = capture["tasks"][task]
    source = Path(record["program"])
    program = source.read_text()
    if sha_text(program) != record["program_sha256"]:
        raise RuntimeError(f"shared anchor changed for {task}")
    return (
        program, float(record["score"]), str(source),
        record.get("source_summary"),
    )


def result_row(path: Path, task: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else [payload]
    row = next((value for value in rows if value.get("task_id") == task), None)
    if row is None:
        raise RuntimeError(f"rollout result has no row for {task}: {path}")
    return row


def h1_case(
    run_root: Path, method: str, tag: str, task: str, round_base: int,
    target: float,
) -> dict[str, Any]:
    outer = run_root / "self_adapt_harness" / f"outer-{PROTOCOL}-{method}-{tag}"
    round_dir = outer / f"round{round_base + 18:03d}"
    programs = json.loads((round_dir / "best_programs_after.json").read_text())
    entry = programs.get(task) or {}
    resolution = "strict_single_program_incumbent"
    historical_source_summary = None
    if entry.get("program") and abs(float(entry["score"]) - target) <= tolerance(task):
        program = str(entry["program"])
        score = float(entry["score"])
        if not entry.get("result_path"):
            raise RuntimeError(f"{task}/{method}: incumbent lacks result provenance")
        source = str(entry["result_path"])
        origin_round = int(entry["round"])
        origin_k = int(entry["k"])
        package = outer / f"round{origin_round:03d}" / "tasks" / task / f"cand{origin_k:02d}"
        origin_meta = json.loads(
            (outer / f"round{origin_round:03d}" / "round.json").read_text()
        )
        proposer_checkpoint = (origin_meta.get("proposer") or {}).get("checkpoint")
        executor_checkpoint = (origin_meta.get("proposer") or {}).get(
            "executor_checkpoint"
        )
        observed_result = result_row(Path(source), task)
        if sha_text(str(observed_result.get("best_program") or "")) != sha_text(program):
            raise RuntimeError(f"{task}/{method}: incumbent/result program mismatch")
        rollout_h2 = observed_result.get("h2_package_provenance") or {}
        if rollout_h2.get("stable_during_rollout") is not True \
                or rollout_h2.get("sha256") != h2_sha256(package):
            raise RuntimeError(f"{task}/{method}: rollout H2 provenance mismatch")
        program_origin_h2_available = True
        program_origin_caveat = None
    else:
        program, score, source, historical_source_summary = anchor_case(
            run_root / "self_adapt_harness" / PROTOCOL / f"{method}_{tag}"
            / "shared_anchor" / "manifest.json",
            task,
        )
        resolution = "shared_x1_anchor_remains_best"
        package = REPO / "src" / "inner" / "harness"
        run_manifest = json.loads(
            (run_root / "self_adapt_harness" / PROTOCOL / f"{method}_{tag}"
             / "run_manifest.json").read_text()
        )
        observed_initial_h2 = h2_sha256(package)
        if observed_initial_h2 != run_manifest["initial_h2_sha256"]:
            raise RuntimeError(f"{task}/{method}: initial H2 bytes changed")
        first_meta = json.loads((outer / f"round{round_base:03d}" / "round.json").read_text())
        proposer_checkpoint = (first_meta.get("proposer") or {}).get("checkpoint")
        executor_checkpoint = (first_meta.get("proposer") or {}).get(
            "executor_checkpoint"
        )
        # The shared x=1 program and its score are byte-bound to the historical
        # source summary, but that legacy source did not record its H2/checkpoint.
        # The package below is the route's verified initial H2, not a fabricated
        # claim about which historical H2 originally generated the anchor.
        program_origin_h2_available = False
        program_origin_caveat = (
            "legacy shared anchor records exact program/score but no origin H2 "
            "hash or executor checkpoint; h2_sha256 binds the route initial H2"
        )
    if abs(score - target) > tolerance(task):
        raise RuntimeError(
            f"{task}/{method}: endpoint {target} has no exact program (closest {score})"
        )
    package_hash = h2_sha256(package)
    return {
        "program": program,
        "program_sha256": sha_text(program),
        "source_summary": source,
        "historical_source_summary": historical_source_summary,
        "endpoint_program_resolution": resolution,
        "h2_package": str(package),
        "h2_sha256": package_hash,
        "h2_provenance_role": (
            "program_origin_h2" if program_origin_h2_available
            else "route_initial_h2_for_legacy_anchor"
        ),
        "program_origin_h2_available": program_origin_h2_available,
        "program_origin_h2_sha256": (
            package_hash if program_origin_h2_available else None
        ),
        "program_origin_caveat": program_origin_caveat,
        "proposer_checkpoint": proposer_checkpoint,
        "executor_checkpoint": executor_checkpoint,
        "search_score": target,
    }


def executor_case(
    run_root: Path, tag: str, task: str, target: float,
) -> dict[str, Any]:
    state_dir = run_root / "self_adapt_harness" / PROTOCOL / "executor" / tag
    state = json.loads((state_dir / "state.json").read_text())
    historical_source_summary = None
    matches = [
        node for node in (state.get("archive") or {}).values()
        if node.get("program") and node.get("score") is not None
        and abs(float(node["score"]) - target) <= tolerance(task)
    ]
    if matches:
        node = min(matches, key=lambda row: (str(row.get("program_hash")), str(row.get("id"))))
        program = str(node["program"])
        score = float(node["score"])
        if not node.get("source_summary") or not node.get("executor_checkpoint"):
            raise RuntimeError(
                f"{task}/executor: archive node lacks exact rollout provenance"
            )
        source = str(node["source_summary"])
        origin_step = int(node["created_step"])
        executor_checkpoint = str(node["executor_checkpoint"])
        resolution = "executor_archive_state"
        program_origin_h2_available = True
        program_origin_caveat = None
    else:
        program, score, source, historical_source_summary = anchor_case(
            run_root / "self_adapt_harness" / PROTOCOL / "executor"
            / "shared_anchor" / "manifest.json",
            task,
        )
        resolution = "shared_x1_anchor_remains_best"
        origin_step = 0
        executor_checkpoint = json.loads(
            (state_dir / "eval_rri16e_u0" / "eval_manifest.json").read_text()
        )["checkpoint"]
        program_origin_h2_available = False
        program_origin_caveat = (
            "legacy shared anchor records exact program/score but no origin H2 "
            "hash or executor checkpoint; h2_sha256 binds executor route batch 0"
        )
    if abs(score - target) > tolerance(task):
        raise RuntimeError(f"{task}/executor: endpoint {target} has no exact program")
    manifest = json.loads(
        (state_dir / f"eval_rri16e_u{origin_step}" / "eval_manifest.json").read_text()
    )
    if manifest["checkpoint"] != executor_checkpoint:
        raise RuntimeError(f"{task}/executor: archive checkpoint provenance mismatch")
    return {
        "program": program,
        "program_sha256": sha_text(program),
        "source_summary": source,
        "historical_source_summary": historical_source_summary,
        "endpoint_program_resolution": resolution,
        "h2_package": str(REPO / "src" / "inner" / "harness"),
        "h2_sha256": manifest["fixed_harness_sha256"],
        "h2_provenance_role": (
            "program_origin_h2" if program_origin_h2_available
            else "executor_route_batch0_h2_for_legacy_anchor"
        ),
        "program_origin_h2_available": program_origin_h2_available,
        "program_origin_h2_sha256": (
            manifest["fixed_harness_sha256"]
            if program_origin_h2_available else None
        ),
        "program_origin_caveat": program_origin_caveat,
        "executor_checkpoint": executor_checkpoint,
        "proposer_checkpoint": None,
        "search_score": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    view = json.loads(args.data.read_text())
    if view.get("status") != "complete" or view.get("protocol") != PROTOCOL:
        raise RuntimeError("endpoint collection requires a complete canonical view")
    cases = []
    for tag, task, proposer_base, context_base in TASKS:
        task_view = view["tasks"][task]
        for method, round_base in (("proposer", proposer_base), ("context", context_base)):
            target = float(task_view["series"][method][-1]["score"])
            case = h1_case(
                args.run_root, method, tag, task, round_base, target
            )
            cases.append({
                "case_id": f"{task}::{method}", "task": task, "method": method,
                "reported_curve_endpoint_score": target, **case,
            })
        target = float(task_view["series"]["executor"][-1]["score"])
        case = executor_case(args.run_root, tag, task, target)
        cases.append({
            "case_id": f"{task}::executor", "task": task, "method": "executor",
            "reported_curve_endpoint_score": target, **case,
        })
    payload = {
        "schema": "reward-route-endpoint-cases/1.0",
        "protocol": PROTOCOL,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_plot_data": str(args.data.resolve()),
        "source_plot_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({len(cases)} exact endpoint programs)")


if __name__ == "__main__":
    main()

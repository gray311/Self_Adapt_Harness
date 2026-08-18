#!/usr/bin/env python3
"""Fail-closed audit for the configured inference-16 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
TASKS = (
    "eft__math__erdos_min_overlap",
    "eft__math__second_autocorr_ineq",
    "eft__math__hadamard_maximal_det",
    "adrs__eplb",
)
EXPECTED_X = [1] + [1 + 16 * (index + 1) for index in range(19)]


def audit_static_configuration(repo: Path = REPO) -> dict[str, Any]:
    config = (repo / "scripts" / "reward_route_inference16_config.sh").read_text()
    expected_literals = (
        "RR_H1_TRAJECTORIES=8",
        "RR_H2_TRAJECTORIES_H1_ROUTE=8",
        "RR_H2_TRAJECTORIES_EXECUTOR_ROUTE=16",
        "RR_TRAJECTORIES_PER_ROUND=16",
        "RR_ROUNDS=19",
        "RR_WEIGHT_UPDATES=18",
        "RR_LORA_RANK=64",
        "RR_LORA_ALPHA=128",
        "RR_EPOCHS=3",
        "RR_LR=3e-5",
        "RR_KL=0.05",
        "RR_H1_MAX_ITERATIONS=24",
        "RR_PROGRAM_RATCHET_MODE=strict_single",
        "RR_MIN_TRAINABLE_H1_ROWS=4",
        "RR_H2_SEED_BASE=200000",
    )
    for literal in expected_literals:
        assert literal in config, f"missing locked config literal: {literal}"

    worker = (repo / "scripts" / "_outer_round_worker.sh").read_text()
    assert "outer.reward.trajectory_budget" in worker
    assert "incumbent fallbacks" in worker
    assert "fixed inference slots are incompatible with cascade" in worker
    # The causal-paired mode may reserve a wider seed block, while the locked
    # K=8, repeats=1 inference-16 route still resolves to stride 16 exactly.
    assert "PAIR_SEED_STRIDE=$((K * PAIRED_REPEATS))" in worker
    assert 'PAIR_SEED_STRIDE" -ge 16' in worker
    assert "logical_index * PAIR_SEED_STRIDE + repeat * K + k" in worker
    assert 'ROUND_SEED_PROGRAMS_FILE="$ROUND_DIR/seed_programs_in.json"' in worker
    assert '--seed-programs-file "$ROUND_SEED_PROGRAMS_FILE"' in worker

    h1_agent = (repo / "src" / "outer" / "harness" / "agent.yaml").read_text()
    assert "max_iterations: 32" in h1_agent
    assert "remind_from_iteration: 6" in h1_agent
    proposer = (repo / "src" / "outer" / "proposing" / "propose.py").read_text()
    assert "_make_repair_fn" not in proposer
    assert "repair_tool_code" not in proposer
    outer_round = (repo / "src" / "outer" / "rounds" / "outer_round.py").read_text()
    assert 'seed_snapshot = round_dir / "seed_programs_in.json"' in outer_round
    assert "update_program_ratchet" in outer_round
    assert "unchanged_program_attribution_failed" in outer_round
    runner = (repo / "src" / "inner" / "runtime" / "harness_runner.py").read_text()
    assert 'stop_reason != "harness_error"' in runner

    train = (repo / "scripts" / "train_mphi_step.sh").read_text()
    assert 'MIN_TRAINABLE_ROWS="${MIN_TRAINABLE_ROWS:-4}"' in train
    assert 'REPLAY_MANIFEST="$ROUND_DIR/replay_manifest.json"' in train

    executor_train = (repo / "scripts" / "submit_ttt_executor_update.sh").read_text()
    for literal in (
        "TTT_LORA_RANK=64",
        "TTT_LORA_ALPHA=128",
        "TTT_EPOCHS=3",
        'TTT_KL="${TTT_KL_COEF:-0.05}"',
        'TTT_LEARNING_RATE="${TTT_LR:-3e-5}"',
    ):
        assert literal in executor_train

    executor_driver = (repo / "scripts" / "drive_ttt_executor_12h.sh").read_text()
    assert "FIXED_LAUNCHED_BATCH" in executor_driver
    assert "fixed comparison forbids top-ups" in executor_driver
    assert "curve4)" in executor_driver
    assert "collect_eval_manifest" in executor_driver
    assert "--expect-h2-sha256" in executor_driver

    runtime_provenance = (
        repo / "scripts" / "runtime" / "provenance.py"
    ).read_text()
    assert "immutable snapshot bytes changed" in runtime_provenance
    assert 'ROOT_FILES: set[str] = set()' in runtime_provenance

    for name in (
        "drive_reward_route_inference16_h1.sh",
        "drive_reward_route_inference16_executor.sh",
    ):
        driver = (repo / "scripts" / name).read_text()
        assert 'RR_RUN_CONFIRMED:-NO' in driver
        assert 'RR_RUN_CONFIRMED=YES' in driver

    refs = json.loads((repo / "results" / "human_best_references.json").read_text())
    assert set(TASKS).issubset(refs["tasks"])
    return {
        "status": "configured_not_run",
        "protocol": "reward-route-inference16-v1",
        "tasks": list(TASKS),
        "rounds": 19,
        "evaluated_weight_updates": 18,
        "trajectory_grid": EXPECTED_X,
        "common_final_x": 305,
        "run_guard_required": "RR_RUN_CONFIRMED=YES",
    }


def audit_completed_view(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["status"] == "complete"
    assert payload["protocol"] == "reward-route-inference16-v1"
    assert int(payload["common_final_x"]) == 305
    assert set(payload["tasks"]) == set(TASKS)
    for task in TASKS:
        for route in ("proposer", "context", "executor"):
            points = payload["tasks"][task]["series"][route]
            assert [int(row["x"]) for row in points] == EXPECTED_X
            for row in points[1:]:
                h1 = int(row["h1_trajectories"])
                h2 = int(row["h2_trajectories"])
                assert h1 + h2 == 16
                if route == "executor":
                    assert (h1, h2) == (0, 16)
                else:
                    assert (h1, h2) == (8, 8)
    return {
        "status": "complete_view_verified",
        "path": str(path.resolve()),
        "tasks": len(TASKS),
        "routes": 3,
        "common_final_x": 305,
    }


def audit_endpoint_validation(path: Path, view_path: Path) -> dict[str, Any]:
    validation = json.loads(path.read_text())
    view = json.loads(view_path.read_text())
    assert validation["status"] == "complete"
    assert validation.get("all_runs_valid") is True
    assert int(validation["requested_runs"]) >= 5
    cases = validation.get("case_results") or {}
    assert len(cases) == len(TASKS) * 3
    for task in TASKS:
        for route in ("proposer", "context", "executor"):
            case = cases[f"{task}::{route}"]
            expected = float(view["tasks"][task]["series"][route][-1]["score"])
            assert abs(float(case["reported_curve_endpoint_score"]) - expected) <= 1e-10
            assert len(str(case["program_sha256"])) == 64
            assert len(str(case["h2_sha256"])) == 64
            assert case.get("h2_provenance_role") in {
                "program_origin_h2",
                "route_initial_h2_for_legacy_anchor",
                "executor_route_batch0_h2_for_legacy_anchor",
            }
            assert isinstance(case.get("program_origin_h2_available"), bool)
            if case["program_origin_h2_available"]:
                assert case.get("program_origin_h2_sha256") == case["h2_sha256"]
            else:
                assert case.get("program_origin_h2_sha256") is None
                assert case.get("program_origin_caveat")
            assert int(case["statistics"]["valid_runs"]) >= 5
    return {"status": "all_12_endpoints_revalidated", "path": str(path.resolve())}


def audit_main_binding(path: Path, validation_path: Path) -> dict[str, Any]:
    binding = json.loads(path.read_text())
    validation = json.loads(validation_path.read_text())
    assert binding["status"] == "complete"
    assert binding["protocol"] == "reward-route-inference16-v1"
    assert binding["authoritative_value"] == "endpoint_revalidation_mean"
    assert binding["validation_sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()
    for task in TASKS:
        assert set(binding["tasks"][task]) == {"proposer", "context", "executor"}
        for route in ("proposer", "context", "executor"):
            bound = binding["tasks"][task][route]
            source = validation["case_results"][f"{task}::{route}"]
            assert abs(float(bound["combined_score_mean"]) - float(source["statistics"]["mean"])) <= 1e-12
            assert bound["program_sha256"] == source["program_sha256"]
            assert bound["h2_sha256"] == source["h2_sha256"]
    return {"status": "main_results_bound_to_validation", "path": str(path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--main-binding", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"configuration": audit_static_configuration()}
    if args.view:
        result["view"] = audit_completed_view(args.view)
    if args.validation:
        if not args.view:
            raise SystemExit("--validation requires --view")
        result["validation"] = audit_endpoint_validation(args.validation, args.view)
    if args.main_binding:
        if not args.validation:
            raise SystemExit("--main-binding requires --validation")
        result["main_binding"] = audit_main_binding(args.main_binding, args.validation)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

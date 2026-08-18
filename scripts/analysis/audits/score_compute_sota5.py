#!/usr/bin/env python3
"""Audit the seven-priority-task reward-route figure and compute accounting.

This audit intentionally separates executor-sample efficiency from total GPU
compute.  The former is recoverable exactly from per-trajectory ledgers; the
latter is not fully reconstructible for the historical proposer campaigns.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SLURM_LOG = Path("/lustre/fsw/portfolios/av/users/yingzim/logs/slurm")
RUN_ROOT = Path(
    "/lustre/fsw/portfolios/av/users/yingzim/runs/self_adapt_harness"
)
BASE = ("/lustre/fsw/portfolios/av/users/yingzim/model_weights/base/"
        "Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
TASKS = (
    "eft__math__hadamard_maximal_det",
    "eft__ahc_simpletes__ahc039",
    "eft__ahc_simpletes__ahc058",
    "adrs__eplb",
    "adrs__prism",
    "adrs__llm_sql",
    "adrs__txn_scheduling",
)
REQUIRED_EXECUTOR_UPDATES = {
    "eft__math__hadamard_maximal_det": 7,
    "eft__ahc_simpletes__ahc039": 7,
    "eft__ahc_simpletes__ahc058": 7,
    "adrs__eplb": 7,
    "adrs__prism": 7,
    "adrs__llm_sql": 7,
    "adrs__txn_scheduling": 7,
}
REQUIRED_CONTEXT_ROUNDS = 10
HISTORICAL_OUTER_ROOT = Path(
    "/lustre/fsw/portfolios/av/users/yingzim/runs/self_adapt_harness/outer"
)
SQL_TASK = "adrs__llm_sql"
TXN_TASK = "adrs__txn_scheduling"
PRISM_TASK = "adrs__prism"
HADAMARD_TASK = "eft__math__hadamard_maximal_det"
AHC039_TASK = "eft__ahc_simpletes__ahc039"
EPLB_TASK = "adrs__eplb"
SQL_CLEAN_START = 0.09343955531989306
SQL_V1_FORENSIC_AUDIT = (
    REPO / "results/provenance_quarantine/"
    "sql_round900_overwrite_20260804T073442/recovery_audit_v2.json"
)
PRISM_CLEAN_START = 24.021666874072007
REFERENCE_REPORTING_HALF_UNIT = {
    "eft__math__hadamard_maximal_det": 0.0000005,
    "eft__ahc_simpletes__ahc039": 0.5 / 225_000,
    "eft__ahc_simpletes__ahc058": 0.5 / 4.5e8,
    "adrs__eplb": 0.00005,   # published as 0.1270
    "adrs__prism": 0.005,    # published as 24.70
    "adrs__llm_sql": 0.00005,  # published as 0.7341
    "adrs__txn_scheduling": 0.005,  # reported as 4761.90 (uncorrected evaluator)
}
REFERENCE_CAVEATS: dict[str, str] = {}
DATASET = Path(
    "/lustre/fsw/portfolios/av/users/yingzim/datasets/self_adapt_harness/raw"
)
INITIAL_PROGRAMS = {
    "eft__math__hadamard_maximal_det": (
        DATASET / "simpletes-b7e0367/datasets/hadamard_maximal_det/"
        "hadamard_maximal_det_29/init_program.py"),
    "eft__ahc_simpletes__ahc039": (
        DATASET / "simpletes-b7e0367/datasets/ahc/ahc039/init_program.py"),
    "eft__ahc_simpletes__ahc058": (
        DATASET / "simpletes-b7e0367/datasets/ahc/ahc058/init_program.py"),
    "adrs__eplb": (
        DATASET / "eft-aac2e79/benchmarks/ADRS/eplb/initial_program.py"),
    "adrs__prism": (
        DATASET / "eft-aac2e79/benchmarks/ADRS/prism/initial_program.py"),
    "adrs__llm_sql": (
        DATASET / "eft-aac2e79/benchmarks/ADRS/llm_sql/initial_program.py"),
    "adrs__txn_scheduling": (
        DATASET / "eft-aac2e79/benchmarks/ADRS/txn_scheduling/initial_program.py"),
}
REPORTING_DISPLAY_TOLERANCE = {
    "eft__math__hadamard_maximal_det": 0.5e-6,
    "eft__ahc_simpletes__ahc039": 0.5,
    "eft__ahc_simpletes__ahc058": 0.5,
    "adrs__eplb": 0.5e-4,
    "adrs__prism": 0.5e-2,
    "adrs__llm_sql": 0.5e-4,
    "adrs__txn_scheduling": 0.5e-2,
}
HUMAN_BEST_REFERENCE = REPO / "results/human_best_references.json"
OPERATIONAL_RETRIES = REPO / "results/sota7_operational_retries.json"
EXCLUDED_CAMPAIGNS = REPO / "results/sota7_excluded_campaigns.json"
ACCEPTED_JOB_ANOMALIES = (
    REPO / "results/sota7_accepted_job_anomalies.json"
)
SACCT_SNAPSHOT = Path(os.environ.get(
    "SOTA7_SACCT_SNAPSHOT",
    str(REPO / "results/sota7_sacct_snapshot.json"),
)).resolve()


def prompt_seed_excerpt(prompt: str) -> str:
    marker = "## Seed program (the executor edits the EVOLVE-BLOCK region)"
    assert marker in prompt and "```python\n" in prompt.split(marker, 1)[1]
    return prompt.split(marker, 1)[1].split(
        "```python\n", 1)[1].split("\n```", 1)[0]


def context_base_program(task: str, base: dict[str, Any]) -> str:
    """Resolve the task program represented by a context ``bases_in`` row.

    Context ratchets store the harness package and score in ``next_bases``;
    the corresponding task program remains in the selected candidate's
    terminal rollout summary.  Resolving it here lets the strict audit prove
    that a resumed segment did not silently switch to another workspace's
    ``best_programs.json`` while retaining a superficially plausible score.
    """
    inline = base.get("program")
    if isinstance(inline, str) and inline:
        return inline
    package = Path(str(base.get("package") or "")).resolve()
    if package == (REPO / "src/inner/harness").resolve():
        return INITIAL_PROGRAMS[task].read_text().strip()
    # Expected layout:
    #   roundNNN/tasks/<task>/candKK
    assert package.name.startswith("cand") and package.parent.name == task, \
        f"{task}: cannot resolve context base package {package}"
    round_dir = package.parents[2]
    assert round_dir.name.startswith("round"), \
        f"{task}: context base package is outside a round: {package}"
    summaries = sorted(
        (round_dir / "rollouts" / task / package.name).glob("*/summary.json")
    )
    assert summaries, f"{task}: selected context base has no terminal summary"
    candidates: list[tuple[float, str]] = []
    for source in summaries:
        payload = json.loads(source.read_text())
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict) or row.get("task_id") != task:
                continue
            if row.get("best_score") is None or not row.get("best_program"):
                continue
            candidates.append((float(row["best_score"]), str(row["best_program"])))
    assert candidates, f"{task}: selected context base has no valid task program"
    return max(candidates, key=lambda item: item[0])[1]


def audit_clean_proposer_checkpoint_chain(
    *,
    task: str,
    adaptive: list[dict[str, Any]],
    workspace: Path,
    outer_root: Path,
    require_complete: bool,
    forbidden_jobs: tuple[int, ...] = (),
    expected_phi_by_round: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Bind every accepted proposer point to its preceding committed update."""
    rounds = [int(point["round"]) for point in adaptive]
    jobs = [int(point["job"]) for point in adaptive]
    assert len(rounds) == len(jobs) == len(adaptive)
    assert rounds == sorted(set(rounds))
    assert len(jobs) == len(set(jobs))

    driver_lines = (workspace / "driver.log").read_text().splitlines()
    latest_trained = Path(BASE).name
    pending_event: dict[str, Any] | None = None
    job_events: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(driver_lines, start=1):
        trained_match = re.search(r"trained -> (mphi_[A-Za-z0-9_.-]+)", line)
        if trained_match:
            latest_trained = trained_match.group(1)
        propose_match = re.search(
            r"round([0-9]+) propose \(phi=([^)]+)\)", line
        )
        if propose_match:
            pending_event = {
                "round": int(propose_match.group(1)),
                "phi": propose_match.group(2),
                "latest_committed_phi": latest_trained,
                "propose_line": line_no,
            }
            continue
        job_match = re.search(r"\bjob ([0-9]+)\s*$", line)
        if job_match and pending_event is not None:
            job = int(job_match.group(1))
            assert job not in job_events, f"{task}: driver repeats outer job {job}"
            job_events[job] = {
                **pending_event, "job": job, "job_line": line_no
            }
            pending_event = None

    by_round: dict[int, dict[str, Any]] = {}
    for point in adaptive:
        round_id, job = int(point["round"]), int(point["job"])
        assert job in job_events, (
            f"{task} round{round_id}: outer job {job} is absent from driver.log"
        )
        event = job_events[job]
        assert int(event["round"]) == round_id
        assert event["phi"] == event["latest_committed_phi"], (
            f"{task} round{round_id}: used {event['phi']} while driver latest "
            f"committed phi was {event['latest_committed_phi']}"
        )
        serve_log = SLURM_LOG / f"sah-outer-{job}.out"
        assert serve_log.is_file(), f"{task} round{round_id}: missing {serve_log}"
        marker = "replica 0 serves: "
        serve_lines = [
            line for line in serve_log.read_text(errors="ignore").splitlines()
            if marker in line
        ]
        assert len(serve_lines) == 1, (
            f"{task} round{round_id}: expected one replica-0 serve record"
        )
        served_phi = Path(serve_lines[0].split(marker, 1)[1]).name
        assert served_phi == event["phi"], (
            f"{task} round{round_id}: driver says {event['phi']}, replica 0 "
            f"served {served_phi}"
        )
        by_round[round_id] = {**event, "served_phi": served_phi}

    assert not set(forbidden_jobs).intersection(jobs), (
        f"{task}: forbidden jobs entered the canonical curve: "
        f"{sorted(set(forbidden_jobs).intersection(jobs))}"
    )
    for round_id, phi in (expected_phi_by_round or {}).items():
        if round_id in by_round:
            assert by_round[round_id]["phi"] == phi, (
                f"{task} round{round_id}: expected {phi}, observed "
                f"{by_round[round_id]['phi']}"
            )

    # A next proposal must follow the preceding round's own update, not merely
    # observe some older checkpoint.  This closes the next_bases-before-train
    # race that produced the excluded PRISM job 2823642.
    for left_round, right_round in zip(rounds, rounds[1:]):
        left, right = by_round[left_round], by_round[right_round]
        left_meta = json.loads(
            (outer_root / f"round{left_round:03d}" / "round.json").read_text()
        )
        valid = sum(
            1 for candidate in left_meta["per_task"][task]["candidates"]
            if candidate.get("valid")
        )
        left_group = json.loads(
            (outer_root / f"round{left_round:03d}" /
             "round_summary.json").read_text()
        )["groups"][task]
        has_training_signal = any(
            abs(float(row.get("advantage") or 0.0)) > 1e-12
            for row in left_group.get("rows") or []
        )
        between = driver_lines[
            int(left["job_line"]):int(right["propose_line"]) - 1
        ]
        committed = [
            match.group(1)
            for line in between
            if (match := re.search(
                r"trained -> (mphi_[A-Za-z0-9_.-]+)", line
            ))
        ]
        if valid >= 4 and has_training_signal:
            assert committed, (
                f"{task} round{right_round}: proposed before round{left_round}'s "
                "required proposer update committed"
            )
        expected = committed[-1] if committed else left["phi"]
        assert right["phi"] == expected, (
            f"{task} round{right_round}: used {right['phi']} rather than "
            f"post-round{left_round} checkpoint {expected}"
        )

    if require_complete:
        last_round, last = rounds[-1], by_round[rounds[-1]]
        last_meta = json.loads(
            (outer_root / f"round{last_round:03d}" / "round.json").read_text()
        )
        last_valid = sum(
            1 for candidate in last_meta["per_task"][task]["candidates"]
            if candidate.get("valid")
        )
        last_group = json.loads(
            (outer_root / f"round{last_round:03d}" /
             "round_summary.json").read_text()
        )["groups"][task]
        last_has_training_signal = any(
            abs(float(row.get("advantage") or 0.0)) > 1e-12
            for row in last_group.get("rows") or []
        )
        if last_valid >= 4 and last_has_training_signal:
            assert any(
                re.search(r"trained -> (mphi_[A-Za-z0-9_.-]+)", line)
                for line in driver_lines[int(last["job_line"]):]
            ), f"{task} round{last_round}: final proposer update was not committed"

    return {
        "status": "accepted_rounds_conditioned_on_preceding_committed_phi",
        "driver_log": str(workspace / "driver.log"),
        "rounds": [by_round[round_id] for round_id in rounds],
        "forbidden_jobs_absent": list(forbidden_jobs),
        "expected_phi_by_round": expected_phi_by_round or {},
    }


def audit_context_round_continuity(
    task: str,
    points: list[dict[str, Any]],
    outer_root: Path,
    initial_bases: dict[str, Any],
) -> dict[str, Any]:
    """Prove base and program continuity across every plotted context round."""
    expected_base = initial_bases[task]
    rows: list[dict[str, Any]] = []
    for point in points:
        round_id = int(point["round"])
        round_dir = outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        actual_base = meta["bases_in"][task]
        assert actual_base == expected_base, (
            f"{task}/round{round_id}: bases_in is not the previous task-local "
            "next_bases row"
        )
        prompts = json.loads((round_dir / "prompts.json").read_text())
        excerpt = prompt_seed_excerpt(str(prompts[task]))
        expected_program = context_base_program(task, actual_base)
        expected_excerpt = expected_program.strip()[:5000]
        assert excerpt == expected_excerpt, (
            f"{task}/round{round_id}: H1 prompt did not receive the program "
            "represented by its task-local bases_in row"
        )
        rows.append({
            "round": round_id,
            "bases_sha256": hashlib.sha256(
                json.dumps(actual_base, sort_keys=True).encode()
            ).hexdigest(),
            "prompt_seed_excerpt_sha256": hashlib.sha256(
                excerpt.encode()
            ).hexdigest(),
            "source_program_sha256": hashlib.sha256(
                expected_program.encode()
            ).hexdigest(),
        })
        expected_base = json.loads(
            (round_dir / "next_bases.json").read_text()
        )[task]
    return {
        "rounds_checked": len(rows),
        "every_bases_in_matches_previous_task_local_next_bases": True,
        "every_prompt_seed_matches_resolved_campaign_incumbent": True,
        "rows": rows,
    }


def audit_reporting_condition_alignment(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Bind the three reported conditions to one canonical run manifest."""

    def display(task: str, score: float) -> float:
        if task == AHC039_TASK:
            return score * 225_000
        if task == "eft__ahc_simpletes__ahc058":
            return score * 4.5e8
        return score

    expected_rows: dict[str, dict[str, float]] = {
        "initial": {
            task: display(task, float(payload["anchors"][task][0]))
            for task in TASKS
        },
        "context": {
            task: display(
                task,
                float(payload["tasks"][task]["series"]["context"][
                    "points"
                ][-1]["score"]),
            )
            for task in TASKS
        },
        "proposer": {
            task: float(
                payload["tasks"][task]["reported_proposer_display_score"]
            )
            for task in TASKS
        },
    }
    rows: dict[str, Any] = {}
    mismatches: list[str] = []
    for label in ("initial", "context", "proposer"):
        method_rows = {}
        for task in TASKS:
            curve_value = expected_rows[label][task]
            if label == "initial":
                source_score = float(payload["anchors"][task][0])
            elif label == "context":
                source_score = float(
                    payload["tasks"][task]["series"]["context"]["points"][-1][
                        "score"
                    ]
                )
            else:
                source_score = float(
                    payload["tasks"][task]["series"]["proposer_full"][
                        "points"
                    ][-1]["score"]
                )
            source_display = display(task, source_score)
            tolerance = REPORTING_DISPLAY_TOLERANCE[task]
            aligned = abs(source_display - curve_value) <= tolerance + 1e-12
            method_rows[task] = {
                "source_combined_score": source_score,
                "source_display_value": source_display,
                "curve_display_value": curve_value,
                "rounding_tolerance": tolerance,
                "aligned": aligned,
            }
            if not aligned:
                mismatches.append(f"{label}:{task}")
        rows[label] = method_rows
    if require_complete:
        assert not mismatches, \
            f"reported conditions diverge from plotted routes: {mismatches}"
    return {
        "status": "all_reported_conditions_derive_from_canonical_routes",
        "rows": ["initial", "context", "proposer"],
        "tasks_checked_per_row": len(TASKS),
        "mismatches": mismatches,
        "strictly_aligned": not mismatches,
        "values": rows,
    }


def audit_human_best_reference(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind every y=1 value to the frozen human-reference manifest."""
    reference_payload = json.loads(HUMAN_BEST_REFERENCE.read_text())
    assert int(reference_payload.get("schema") or 0) == 1
    assert reference_payload.get("status") == "frozen"
    assert reference_payload.get("direction") == "higher_is_better"
    assert reference_payload.get("normalization") == (
        "combined_score / human_best_combined_score"
    )
    reference_tasks = reference_payload.get("tasks") or {}
    assert set(TASKS).issubset(reference_tasks)

    def display(task: str, score: float) -> float:
        if task == AHC039_TASK:
            return score * 225_000
        if task == "eft__ahc_simpletes__ahc058":
            return score * 4.5e8
        return score

    rows: dict[str, Any] = {}
    for task in TASKS:
        frozen = reference_tasks[task]
        reference = float(payload["anchors"][task][1])
        observed = display(task, reference)
        frozen_score = float(frozen["human_best_combined_score"])
        frozen_display = float(frozen["human_best_display_value"])
        tolerance = float(frozen["display_rounding_half_unit"])
        assert math.isclose(reference, frozen_score, rel_tol=0.0, abs_tol=1e-12), (
            f"{task}: plotted y=1 reference is not the frozen human reference"
        )
        assert abs(observed - frozen_display) <= tolerance + 1e-12, (
            f"{task}: human-reference display conversion is inconsistent"
        )
        rows[task] = {
            "human_best_display_value": frozen_display,
            "human_best_combined_score": reference,
            "display_tolerance": tolerance,
            "aligned": True,
        }
    return {
        "status": "all_y_equals_one_values_match_frozen_human_references",
        "reference_manifest": str(HUMAN_BEST_REFERENCE),
        "reference_manifest_sha256": hashlib.sha256(
            HUMAN_BEST_REFERENCE.read_bytes()
        ).hexdigest(),
        "normalization": "combined_score / human_best_combined_score",
        "tasks": rows,
    }


def audit_paper_reward_route_alignment(*, require_complete: bool) -> dict[str, Any]:
    """Keep the selected main view honest and the full view discoverable."""
    experiment_path = REPO / "papers/subtex/experiment.tex"
    appendix_path = REPO / "papers/subtex/appendix.tex"
    paper_paths = sorted((REPO / "papers").glob("**/*.tex"))
    experiment = experiment_path.read_text()
    appendix = appendix_path.read_text()
    all_tex = "\n".join(path.read_text() for path in paper_paths)
    normalized_experiment = " ".join(experiment.split())

    violations: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    require(
        "figures/score_compute_curves_sota4_final.pdf" in experiment,
        "main text does not include the final selected 1x4 figure",
    )
    require(
        "figures/score_compute_curves_sota7_final.pdf" in appendix,
        "appendix does not include the final full 1x7 figure",
    )
    require(
        "after interim inspection" in normalized_experiment,
        "main caption/text does not disclose post-interim task selection",
    )
    require(
        "illustrative" in normalized_experiment,
        "main text does not label the selected four-task view illustrative",
    )
    require(
        "executor-trajectory sample efficiency" in normalized_experiment,
        "main text does not scope efficiency to executor trajectories",
    )
    require(
        "full seven-task" in normalized_experiment,
        "main text does not point aggregate interpretation to the full view",
    )
    require(
        "score_compute_curves_sota5" not in all_tex,
        "superseded sota5 figure reference remains in the paper",
    )
    require(
        "on the five tasks where" not in normalized_experiment,
        "superseded five-task selection claim remains",
    )
    require(
        "adds a further gain on \\emph{every}" not in normalized_experiment,
        "stale every-task proposer-gain claim remains",
    )
    require(
        "full ordering \\method~(initial) $<$ \\method~(context) $<$ \\method~(weight) holds" not in normalized_experiment,
        "stale full-ordering claim remains",
    )
    if require_complete:
        assert not violations, (
            "paper reward-route/main-view alignment is incomplete: " +
            "; ".join(violations)
        )
    return {
        "status": "aligned" if not violations else "pending_paper_sync",
        "main_text": str(experiment_path),
        "appendix": str(appendix_path),
        "selection_status": "post_interim_illustrative_not_confirmatory",
        "standalone_four_task_aggregate_claim_allowed": False,
        "violations": violations,
    }


def audit_sql_proposer_lineage(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Fail closed if the reported SQL endpoint used curated program leakage.

    The old exploratory SQL result had an ``analyst_note`` that named a 0.728
    program and instructed the solver to copy it.  Audit the actual serialized
    H1 prompts, rather than a mutable feedback file, and close the isolated
    task-local incumbent chain from fixed H2 to the plotted/reported score.
    """
    forbidden_prompt_fragments = (
        "analyst note:",
        "alternative approach scoring 0.728",
        "adopt that alternative verbatim",
        "do not keep iterating on the current 0.45",
        "current 0.45 program",
    )
    task_payload = payload["tasks"][SQL_TASK]
    proposer = task_payload["series"]["proposer_full"]
    adaptive_points = [point for point in proposer["points"]
                       if point.get("round") is not None]
    assert adaptive_points, "clean SQL proposer curve has no adaptive points"
    clean_rounds = [int(point["round"]) for point in adaptive_points]
    assert clean_rounds == sorted(set(clean_rounds)), \
        "clean SQL proposer rounds are duplicated or out of order"
    outer_root = Path(task_payload["proposer_outer_root"])
    workspace = Path(task_payload["proposer_workspace"])
    assert workspace.name == "proposer_sota7_sql_clean_v2", (
        "final SQL proposer must use the independent clean-v2 workspace"
    )
    assert outer_root.name == "outer-proposer-sota7-sql-clean-v2", (
        "final SQL proposer must exclude the overwritten v1 outer lineage"
    )
    run_manifest = json.loads((workspace / "run_manifest.json").read_text())
    assert int(run_manifest.get("schema") or 0) == 2
    assert run_manifest.get("method") == "update proposer weights (ours)"
    assert run_manifest.get("task") == SQL_TASK
    assert run_manifest.get("isolated_feedback") is True
    assert run_manifest.get("curated_notes_allowed") is False
    assert Path(run_manifest["outer_root"]).resolve() == outer_root.resolve()
    assert run_manifest.get("h2_program_inherited") is False
    assert run_manifest.get("completed_rounds_immutable") is True
    excluded = run_manifest.get("excluded_lineage") or {}
    assert excluded.get("workspace") == "proposer_sota5_sql_clean_v1"
    assert {int(job) for job in excluded.get("excluded_jobs_charged") or []} == {
        2821934, 2821956,
    }
    forensic = json.loads(SQL_V1_FORENSIC_AUDIT.read_text())
    assert forensic.get("status") == (
        "forensic_recovery_verified_v1_excluded_from_final_curve"
    )
    assert forensic.get("task") == SQL_TASK and int(forensic.get("round")) == 900
    assert forensic.get("preincident_prompt_sha256") == (
        forensic.get("recovered_prompt_sha256")
    )
    assert forensic.get("preincident_prompt_sha256") == (
        "87bc6ab9cc01a2822f277c5c2b56d55175c4a9e5fd292e321c47d193da690dcd"
    )
    assert forensic.get("curve_treatment", "").startswith("exclude all SQL-v1")
    sql_training = run_manifest.get("training") or {}
    assert sql_training.get("adapter_state") == (
        "continued from the preceding proposer update"
    ), "clean SQL manifest misstates adapter continuation"
    assert sql_training.get("optimizer_state") == (
        "reinitialized for every training job"
    ), "clean SQL manifest misstates optimizer persistence"
    assert sql_training.get("scheduler_state") == (
        "reinitialized for every training job"
    ), "clean SQL manifest misstates scheduler persistence"
    checkpoint_chain = audit_clean_proposer_checkpoint_chain(
        task=SQL_TASK,
        adaptive=adaptive_points,
        workspace=workspace,
        outer_root=outer_root,
        require_complete=require_complete,
    )
    if require_complete:
        assert (workspace / "CANONICAL_COMPLETE").is_file(), \
            "clean SQL proposer campaign has not passed the plateau/budget review"

    incumbent = SQL_CLEAN_START
    incumbent_package = str(REPO / "src/inner/harness")
    rounds: list[dict[str, Any]] = []
    initial_program = INITIAL_PROGRAMS[SQL_TASK]
    initial_excerpt = initial_program.read_text().strip()[:5000]

    for index, round_id in enumerate(clean_rounds):
        round_dir = outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        summary = json.loads((round_dir / "round_summary.json").read_text())
        prompts = json.loads((round_dir / "prompts.json").read_text())
        prompt = str(prompts[SQL_TASK])
        prompt_lower = prompt.lower()
        leak_hits = [fragment for fragment in forbidden_prompt_fragments
                     if fragment in prompt_lower]

        assert int(meta["round"]) == round_id
        assert meta["tasks_order"] == [SQL_TASK], \
            f"round{round_id}: SQL fresh lineage is not task isolated"
        assert int(meta["k"]) == 8 and int(meta["max_evals"]) == 20, \
            f"round{round_id}: unexpected SQL proposal/evaluation budget"
        assert str((meta.get("proposer") or {}).get("model")) == "qwen3.5-9b", \
            f"round{round_id}: unexpected SQL proposer model"
        base = meta["bases_in"][SQL_TASK]
        assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12), \
            f"round{round_id}: SQL incumbent score chain is broken"
        assert Path(base["package"]).resolve() == Path(incumbent_package).resolve(), \
            f"round{round_id}: SQL incumbent package chain is broken"
        assert not leak_hits, \
            f"round{round_id}: curated SQL leak reached serialized H1 prompt: {leak_hits}"
        if index == 0:
            seed_excerpt = prompt_seed_excerpt(prompt)
            assert seed_excerpt == initial_excerpt, \
                "first clean SQL adaptive batch did not show task.initial_program"

        group = summary["groups"][SQL_TASK]
        assert math.isclose(float(group["base_score"]), incumbent, abs_tol=1e-12)
        best_k = group.get("best_k")
        best_score = group.get("best_score")
        if best_k is not None:
            selected = [row for row in group["rows"]
                        if int(row["k"]) == int(best_k)]
            assert len(selected) == 1 and selected[0]["valid"], \
                f"round{round_id}: selected SQL candidate is not uniquely valid"
            assert math.isclose(float(selected[0]["score"]), float(best_score),
                                abs_tol=1e-12)
        if group.get("improved"):
            assert best_k is not None and float(best_score) > incumbent
            incumbent = float(best_score)
            incumbent_package = str(
                round_dir / "tasks" / SQL_TASK / f"cand{int(best_k):02d}"
            )

        rounds.append({
            "round": round_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "serialized_prompt_leak_hits": leak_hits,
            "base_score": float(group["base_score"]),
            "best_score": (float(best_score) if best_score is not None else None),
            "best_k": best_k,
            "improved": bool(group.get("improved")),
            "next_incumbent_score": incumbent,
            "next_incumbent_package": incumbent_package,
        })

    curve_endpoint = float(proposer["points"][-1]["score"])
    reported_endpoint = float(task_payload["reported_proposer_combined_score"])
    assert math.isclose(incumbent, curve_endpoint, abs_tol=1e-12), \
        "clean SQL proposer lineage does not close at the plotted endpoint"
    assert math.isclose(incumbent, reported_endpoint, abs_tol=1e-12), \
        "clean SQL proposer endpoint is not aligned with the reported value"

    leaked_prompt = str(json.loads(
        (HISTORICAL_OUTER_ROOT / "round471/prompts.json").read_text()
    )[SQL_TASK]).lower()
    historical_hits = [fragment for fragment in forbidden_prompt_fragments
                       if fragment in leaked_prompt]
    assert historical_hits, \
        "expected historical SQL leak evidence is missing; re-audit exclusion"
    return {
        "status": "clean_task_local_lineage_verified",
        "task": SQL_TASK,
        "outer_root": str(outer_root),
        "workspace": str(workspace),
        "excluded_v1_forensic_audit": str(SQL_V1_FORENSIC_AUDIT),
        "rounds": clean_rounds,
        "common_start_h2_score": SQL_CLEAN_START,
        "endpoint_score": incumbent,
        "first_round_base_package": str(REPO / "src/inner/harness"),
        "first_round_seed_program": str(initial_program),
        "first_round_seed_excerpt_sha256": hashlib.sha256(
            initial_excerpt.encode()).hexdigest(),
        "serialized_prompts_checked": len(rounds),
        "curated_leak_fragments_found": 0,
        "excluded_historical_rounds": [471, 472, 473, 474, 475],
        "excluded_historical_prompt_leak_hits": historical_hits,
        "checkpoint_chain": checkpoint_chain,
        "lineage": rounds,
        "interpretation": (
            "The stored proposal prompts contain no curated analyst note or named "
            "0.728 program; each later base is the verified winner (or unchanged "
            "incumbent) from this isolated SQL campaign."
        ),
    }


def audit_txn_proposer_lineage(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Close the clean Txn lineage and prove exclusion of round450 leakage."""
    task_payload = payload["tasks"][TXN_TASK]
    proposer = task_payload["series"]["proposer_full"]
    adaptive = [point for point in proposer["points"]
                if point.get("round") is not None]
    assert adaptive, "clean Txn proposer curve has no adaptive points"
    rounds = [int(point["round"]) for point in adaptive]
    assert rounds == sorted(set(rounds))
    outer_root = Path(task_payload["proposer_outer_root"])
    workspace = Path(task_payload["proposer_workspace"])
    assert workspace.name == "proposer_sota7_txn_clean_v1"
    assert outer_root.name == "outer-proposer-sota7-txn-clean-v1"
    manifest = json.loads((workspace / "run_manifest.json").read_text())
    assert manifest.get("method") == "update proposer weights (ours)"
    assert manifest.get("task") == TXN_TASK
    assert manifest.get("isolated_feedback") is True
    assert manifest.get("curated_notes_allowed") is False
    assert manifest.get("h2_program_inherited") is False
    assert manifest.get("first_adaptive_batch_program") == "task.initial_program"
    assert Path(manifest["outer_root"]).resolve() == outer_root.resolve()
    worker = REPO / "src/inner/evaluation/_eval_worker.py"
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    assert manifest.get("eval_worker_sha256") == worker_sha
    guard_fields = (payload.get("protocol_guard_proof") or {}).get("fields") or {}
    proof_job = int(guard_fields.get("job") or 0)
    proposer_jobs = [int(point["job"]) for point in adaptive if point.get("job")]
    assert proof_job > 0 and proposer_jobs and all(job > proof_job for job in proposer_jobs), \
        "clean Txn proposer round predates the current legality-guard proof"
    checkpoint_chain = audit_clean_proposer_checkpoint_chain(
        task=TXN_TASK,
        adaptive=adaptive,
        workspace=workspace,
        outer_root=outer_root,
        require_complete=require_complete,
    )
    excluded = manifest.get("excluded_historical_lineage") or {}
    illegal_hash = str(excluded.get("illegal_seed_program_sha256") or "")
    assert illegal_hash == (
        "8229c2ee216faef5140c8e6e6560612e8d2fdb13a63f9ebe42db255a3e58e2a3"
    )
    if require_complete:
        assert (workspace / "CANONICAL_COMPLETE").is_file(), \
            "clean Txn proposer campaign has not passed plateau/budget review"

    initial_excerpt = INITIAL_PROGRAMS[TXN_TASK].read_text().strip()[:5000]
    first_prompt = json.loads(
        (outer_root / f"round{rounds[0]:03d}" / "prompts.json").read_text()
    )[TXN_TASK]
    clean_seed = prompt_seed_excerpt(str(first_prompt))
    assert clean_seed == initial_excerpt, \
        "clean Txn first proposer batch did not show task.initial_program"
    assert hashlib.sha256(clean_seed.encode()).hexdigest() != illegal_hash

    contaminated_prompt = json.loads(
        (HISTORICAL_OUTER_ROOT / "round460/prompts.json").read_text()
    )[TXN_TASK]
    contaminated_seed = prompt_seed_excerpt(str(contaminated_prompt))
    contaminated_hash = hashlib.sha256(contaminated_seed.encode()).hexdigest()
    assert contaminated_hash == illegal_hash, \
        "historical Txn contamination evidence changed; re-audit exclusion"

    collector_fallbacks = []
    for round_id in rounds:
        round_dir = outer_root / f"round{round_id:03d}"
        collect_log = round_dir / "collect.log"
        if collect_log.is_file() and "SAH_ADV=v3" in collect_log.read_text(
            errors="ignore"
        ):
            continue
        group = json.loads(
            (round_dir / "round_summary.json").read_text()
        )["groups"][TXN_TASK]
        adv_mode = str(group.get("adv_mode") or "")
        # The driver can salvage a fully materialized round after an outer-job
        # epilogue failure.  Its default collector is v2; rewards.py guarantees
        # v2 and v3 are byte-identical whenever within-group signal exists.
        # A tied/no-signal fallback would change the intended training target
        # and is therefore inadmissible.
        assert adv_mode.startswith("rloo+max(") or adv_mode.startswith(
            "hist_rescue("
        ) or adv_mode == "no_signal(true-plateau)", \
            f"round{round_id}: unproved v2/v3 collector fallback ({adv_mode})"
        collector_fallbacks.append({
            "round": round_id,
            "collect_log": str(collect_log),
            "adv_mode": adv_mode,
            "equivalence": (
                "v2 and v3 byte-identical because within-group reward variance is nonzero"
                if adv_mode.startswith("rloo+max(") else
                "summary mode itself proves the v3 collector path"
            ),
        })

    curve_endpoint = float(proposer["points"][-1]["score"])
    assert math.isclose(
        curve_endpoint,
        float(task_payload["reported_proposer_combined_score"]),
        abs_tol=1e-12,
    )
    return {
        "status": "clean_task_local_current_guard_lineage_verified",
        "task": TXN_TASK,
        "workspace": str(workspace),
        "outer_root": str(outer_root),
        "rounds": rounds,
        "endpoint_score": curve_endpoint,
        "first_round_seed_program": str(INITIAL_PROGRAMS[TXN_TASK]),
        "first_round_seed_sha256": hashlib.sha256(clean_seed.encode()).hexdigest(),
        "excluded_historical_rounds": excluded.get("rounds"),
        "excluded_illegal_seed_program_sha256": contaminated_hash,
        "collector_fallback_equivalence": collector_fallbacks,
        "checkpoint_chain": checkpoint_chain,
        "interpretation": (
            "round460--463 is excluded despite legal endpoint outputs because "
            "its H1 prompt inherited the invalid round450 one-element program"
        ),
    }


def audit_prism_proposer_lineage(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Close the clean PRISM lineage and prove exclusion of round410--417."""
    task_payload = payload["tasks"][PRISM_TASK]
    proposer = task_payload["series"]["proposer_full"]
    adaptive = [point for point in proposer["points"]
                if point.get("round") is not None]
    assert adaptive, "clean PRISM proposer curve has no adaptive points"
    rounds = [int(point["round"]) for point in adaptive]
    assert rounds == sorted(set(rounds))
    outer_root = Path(task_payload["proposer_outer_root"])
    workspace = Path(task_payload["proposer_workspace"])
    assert workspace.name == "proposer_sota7_prism_clean_v1"
    assert outer_root.name == "outer-proposer-sota7-prism-clean-v1"
    manifest = json.loads((workspace / "run_manifest.json").read_text())
    assert manifest.get("method") == "update proposer weights (ours)"
    assert manifest.get("task") == PRISM_TASK
    assert manifest.get("isolated_feedback") is True
    assert manifest.get("curated_notes_allowed") is False
    assert manifest.get("h2_program_inherited") is False
    assert manifest.get("first_adaptive_batch_program") == "task.initial_program"
    assert Path(manifest["outer_root"]).resolve() == outer_root.resolve()
    worker = REPO / "src/inner/evaluation/_eval_worker.py"
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    assert manifest.get("eval_worker_sha256") == worker_sha
    guard_fields = (payload.get("protocol_guard_proof") or {}).get("fields") or {}
    proof_job = int(guard_fields.get("job") or 0)
    proposer_jobs = [int(point["job"]) for point in adaptive if point.get("job")]
    assert proof_job > 0 and len(proposer_jobs) == len(adaptive)
    assert all(job > proof_job for job in proposer_jobs), \
        "clean PRISM proposer round predates the full-success guard proof"

    checkpoint_chain = audit_clean_proposer_checkpoint_chain(
        task=PRISM_TASK,
        adaptive=adaptive,
        workspace=workspace,
        outer_root=outer_root,
        require_complete=require_complete,
        forbidden_jobs=(2823642, 2824284),
        expected_phi_by_round={1030: "mphi_sota7_prism_clean_v1_09"},
    )
    phi_by_round = {
        int(row["round"]): row for row in checkpoint_chain["rounds"]
    }
    excluded = manifest.get("excluded_historical_lineage") or {}
    invalid_hash = str(excluded.get("selected_invalid_program_sha256") or "")
    assert invalid_hash == (
        "53f907522e5ad378453fa98820be6be95d09e59aa93e904f93850701f0292b6e"
    )
    assert excluded.get("rounds") == list(range(410, 418))
    if require_complete:
        assert (workspace / "CANONICAL_COMPLETE").is_file(), \
            "clean PRISM proposer campaign has not passed plateau/budget review"

    initial_excerpt = INITIAL_PROGRAMS[PRISM_TASK].read_text().strip()[:5000]
    incumbent = PRISM_CLEAN_START
    incumbent_package = str(REPO / "src/inner/harness")
    lineage: list[dict[str, Any]] = []
    for index, round_id in enumerate(rounds):
        round_dir = outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        summary = json.loads((round_dir / "round_summary.json").read_text())
        prompt = str(json.loads((round_dir / "prompts.json").read_text())[PRISM_TASK])
        assert meta.get("tasks_order") == [PRISM_TASK]
        assert int(meta.get("k") or 0) == 8
        assert int(meta.get("max_evals") or 0) == 20
        assert str((meta.get("proposer") or {}).get("model")) == "qwen3.5-9b"
        base = meta["bases_in"][PRISM_TASK]
        assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12)
        assert Path(base["package"]).resolve() == Path(incumbent_package).resolve()
        if index == 0:
            assert prompt_seed_excerpt(prompt) == initial_excerpt, \
                "clean PRISM first proposer batch did not show task.initial_program"

        group = summary["groups"][PRISM_TASK]
        assert math.isclose(float(group["base_score"]), incumbent, abs_tol=1e-12)
        best_k = group.get("best_k")
        best_score = group.get("best_score")
        selected_success_rate = None
        selected_program_sha = None
        if group.get("improved"):
            assert best_k is not None and float(best_score) > incumbent
            selected = [row for row in group["rows"]
                        if int(row["k"]) == int(best_k)]
            assert len(selected) == 1 and selected[0]["valid"]
            assert math.isclose(float(selected[0]["score"]), float(best_score),
                                abs_tol=1e-12)
            candidate_root = (
                round_dir / "rollouts" / PRISM_TASK / f"cand{int(best_k):02d}"
            )
            summaries = list(candidate_root.glob("*/summary.json"))
            assert summaries, f"round{round_id}: selected PRISM summary missing"
            selected_summary = json.loads(
                max(summaries, key=lambda path: path.stat().st_mtime).read_text()
            )
            if isinstance(selected_summary, list):
                assert len(selected_summary) == 1
                selected_summary = selected_summary[0]
            selected_success_rate = float(
                (selected_summary.get("best_metrics") or {}).get("success_rate", 0.0)
            )
            assert math.isclose(selected_success_rate, 1.0, abs_tol=1e-12), \
                f"round{round_id}: selected PRISM program is not full-success"
            selected_program = str(selected_summary.get("best_program") or "")
            assert selected_program
            selected_program_sha = hashlib.sha256(
                selected_program.encode()).hexdigest()
            incumbent = float(best_score)
            incumbent_package = str(
                round_dir / "tasks" / PRISM_TASK / f"cand{int(best_k):02d}"
            )

        next_base = json.loads((round_dir / "next_bases.json").read_text())[PRISM_TASK]
        assert math.isclose(float(next_base["score"]), incumbent, abs_tol=1e-12)
        assert Path(next_base["package"]).resolve() == Path(incumbent_package).resolve()
        lineage.append({
            "round": round_id,
            "outer_job": int(phi_by_round[round_id]["job"]),
            "proposer_phi": phi_by_round[round_id]["phi"],
            "latest_committed_phi": phi_by_round[round_id][
                "latest_committed_phi"
            ],
            "base_score": float(group["base_score"]),
            "best_score": float(best_score) if best_score is not None else None,
            "best_k": best_k,
            "improved": bool(group.get("improved")),
            "selected_success_rate": selected_success_rate,
            "selected_program_sha256": selected_program_sha,
            "next_incumbent_score": incumbent,
            "next_incumbent_package": incumbent_package,
        })

    historical_round = HISTORICAL_OUTER_ROOT / "round410"
    historical_group = json.loads(
        (historical_round / "round_summary.json").read_text()
    )["groups"][PRISM_TASK]
    assert historical_group.get("improved") is True
    assert int(historical_group.get("best_k")) == 4
    historical_summary_path = max(
        (historical_round / "rollouts" / PRISM_TASK / "cand04").glob(
            "*/summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    historical_summary = json.loads(historical_summary_path.read_text())
    if isinstance(historical_summary, list):
        assert len(historical_summary) == 1
        historical_summary = historical_summary[0]
    historical_success = float(
        (historical_summary.get("best_metrics") or {}).get("success_rate", 0.0)
    )
    historical_program_hash = hashlib.sha256(
        str(historical_summary.get("best_program") or "").encode()
    ).hexdigest()
    assert math.isclose(historical_success, 0.98, abs_tol=1e-12)
    assert historical_program_hash == invalid_hash
    historical_package = str(historical_round / "tasks" / PRISM_TASK / "cand04")
    old_next = json.loads((historical_round / "next_bases.json").read_text())[PRISM_TASK]
    old_round411 = json.loads(
        (HISTORICAL_OUTER_ROOT / "round411/round.json").read_text()
    )["bases_in"][PRISM_TASK]
    assert Path(old_next["package"]).resolve() == Path(historical_package).resolve()
    assert Path(old_round411["package"]).resolve() == Path(historical_package).resolve()

    curve_endpoint = float(proposer["points"][-1]["score"])
    assert math.isclose(incumbent, curve_endpoint, abs_tol=1e-12)
    assert math.isclose(
        curve_endpoint,
        float(task_payload["reported_proposer_combined_score"]),
        abs_tol=1e-12,
    )
    return {
        "status": "clean_task_local_current_guard_lineage_verified",
        "task": PRISM_TASK,
        "workspace": str(workspace),
        "outer_root": str(outer_root),
        "rounds": rounds,
        "endpoint_score": curve_endpoint,
        "first_round_seed_program": str(INITIAL_PROGRAMS[PRISM_TASK]),
        "first_round_seed_sha256": hashlib.sha256(
            initial_excerpt.encode()).hexdigest(),
        "excluded_historical_rounds": excluded.get("rounds"),
        "excluded_selected_invalid_program_sha256": historical_program_hash,
        "excluded_selected_success_rate": historical_success,
        "excluded_harness_inherited_by_round411": historical_package,
        "checkpoint_chain": checkpoint_chain,
        "lineage": lineage,
        "interpretation": (
            "round410--417 is excluded because the partial-success round410 "
            "winner changed the H1 harness lineage; every selected winner in "
            "the clean campaign has success_rate=1.0"
        ),
    }


def authoritative_terminal_score(
    round_dir: Path, task: str, k: int
) -> float | None:
    """Return terminal score; checkpoint fallback is interruption-only."""
    root = round_dir / "rollouts" / task / f"cand{k:02d}"
    best: float | None = None
    saw_terminal = False
    for source in root.glob("*/summary.json"):
        try:
            payload = json.loads(source.read_text())
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, dict) or row.get("task_id") != task:
                    continue
                saw_terminal = True
                if row.get("best_score") is not None:
                    score = float(row["best_score"])
                    best = score if best is None else max(best, score)
        except Exception:
            continue
    if saw_terminal:
        return best
    for source in root.glob(f"*/checkpoints/{task}.json"):
        try:
            score = float(json.loads(source.read_text())["best_score"])
            best = score if best is None else max(best, score)
        except Exception:
            continue
    return best


def audit_rewardfix_proposer_lineage(
    payload: dict[str, Any], task: str, kind: str, *, require_complete: bool
) -> dict[str, Any]:
    """Close Hadamard/AHC058 reruns after the checkpoint-credit bug."""
    task_payload = payload["tasks"][task]
    proposer = task_payload["series"]["proposer_full"]
    adaptive = [point for point in proposer["points"]
                if point.get("round") is not None]
    assert adaptive, f"clean {kind} proposer curve has no adaptive points"
    rounds = [int(point["round"]) for point in adaptive]
    assert rounds == sorted(set(rounds))
    outer_root = Path(task_payload["proposer_outer_root"])
    workspace = Path(task_payload["proposer_workspace"])
    assert workspace.name == f"proposer_sota7_{kind}_rewardfix_v1"
    assert outer_root.name == f"outer-proposer-sota7-{kind}-rewardfix-v1"
    manifest = json.loads((workspace / "run_manifest.json").read_text())
    assert manifest.get("method") == "update proposer weights (ours)"
    assert manifest.get("task") == task
    assert manifest.get("isolated_feedback") is True
    assert manifest.get("curated_notes_allowed") is False
    assert manifest.get("h2_program_inherited") is False
    assert manifest.get("first_adaptive_batch_program") == "task.initial_program"
    assert manifest.get("terminal_summary_authoritative") is True
    assert manifest.get(
        "checkpoint_fallback_only_without_terminal_task_row"
    ) is True
    assert Path(manifest["outer_root"]).resolve() == outer_root.resolve()
    reward_sha = hashlib.sha256(
        (REPO / "src/outer/reward/rewards.py").read_bytes()
    ).hexdigest()
    assert manifest.get("reward_loader_sha256") == reward_sha
    excluded = manifest.get("excluded_historical_lineage") or {}
    assert excluded.get("rounds_with_false_checkpoint_credit"), \
        f"{kind} manifest lacks the excluded historical rounds"
    checkpoint_chain = audit_clean_proposer_checkpoint_chain(
        task=task,
        adaptive=adaptive,
        workspace=workspace,
        outer_root=outer_root,
        require_complete=require_complete,
    )
    if require_complete:
        assert (workspace / "CANONICAL_COMPLETE").is_file(), \
            f"clean {kind} proposer campaign lacks plateau/cap review"

    baseline = json.loads(
        (REPO / "results/baseline_h2_20ev.json").read_text()
    )["baseline"][task]
    incumbent = float(baseline["h2_best"])
    incumbent_package = str(REPO / "src/inner/harness")
    initial_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
    lineage: list[dict[str, Any]] = []
    for index, round_id in enumerate(rounds):
        round_dir = outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        summary = json.loads((round_dir / "round_summary.json").read_text())
        prompt = str(json.loads(
            (round_dir / "prompts.json").read_text()
        )[task])
        assert meta.get("tasks_order") == [task]
        assert int(meta.get("k") or 0) == 8
        assert int(meta.get("max_evals") or 0) == 20
        base = meta["bases_in"][task]
        assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12)
        assert Path(base["package"]).resolve() == Path(incumbent_package).resolve()
        if index == 0:
            assert prompt_seed_excerpt(prompt) == initial_excerpt, \
                f"clean {kind} first batch did not show task.initial_program"

        group = summary["groups"][task]
        assert math.isclose(float(group["base_score"]), incumbent, abs_tol=1e-12)
        attribution_rows = []
        for row in group.get("rows") or []:
            if not row.get("valid"):
                continue
            k = int(row["k"])
            credited = row.get("score")
            terminal = authoritative_terminal_score(round_dir, task, k)
            assert (credited is None) == (terminal is None), \
                f"round{round_id}/cand{k}: terminal/credited null mismatch"
            if credited is not None:
                assert math.isclose(
                    float(credited), float(terminal),
                    rel_tol=1e-12, abs_tol=1e-12,
                ), f"round{round_id}/cand{k}: false checkpoint credit"
            attribution_rows.append({
                "k": k, "credited_score": credited,
                "terminal_score": terminal,
            })
        best_k = group.get("best_k")
        best_score = group.get("best_score")
        if group.get("improved"):
            assert best_k is not None and float(best_score) > incumbent
            assert authoritative_terminal_score(
                round_dir, task, int(best_k)
            ) is not None
            incumbent = float(best_score)
            incumbent_package = str(
                round_dir / "tasks" / task / f"cand{int(best_k):02d}"
            )
        next_base = json.loads(
            (round_dir / "next_bases.json").read_text()
        )[task]
        assert math.isclose(float(next_base["score"]), incumbent, abs_tol=1e-12)
        assert Path(next_base["package"]).resolve() == Path(
            incumbent_package
        ).resolve()
        lineage.append({
            "round": round_id,
            "base_score": float(group["base_score"]),
            "best_score": float(best_score) if best_score is not None else None,
            "best_k": best_k,
            "improved": bool(group.get("improved")),
            "attribution_rows": attribution_rows,
            "next_incumbent_score": incumbent,
        })

    endpoint = float(proposer["points"][-1]["score"])
    assert math.isclose(endpoint, incumbent, abs_tol=1e-12)
    assert math.isclose(
        endpoint, float(task_payload["reported_proposer_combined_score"]),
        abs_tol=1e-12,
    )
    historical_false_credits = []
    for round_id in excluded["rounds_with_false_checkpoint_credit"]:
        round_dir = HISTORICAL_OUTER_ROOT / f"round{int(round_id):03d}"
        group = json.loads(
            (round_dir / "round_summary.json").read_text()
        )["groups"][task]
        for row in group.get("rows") or []:
            if not row.get("valid") or row.get("score") is None:
                continue
            k = int(row["k"])
            terminal = authoritative_terminal_score(round_dir, task, k)
            if terminal is None:
                historical_false_credits.append({
                    "round": int(round_id), "k": k,
                    "credited_seed_checkpoint_score": float(row["score"]),
                    "authoritative_terminal_score": None,
                })
    assert {row["round"] for row in historical_false_credits} == set(
        excluded["rounds_with_false_checkpoint_credit"]
    ), f"{kind} historical false-credit evidence does not cover its exclusion"
    return {
        "status": "clean_terminal_attribution_lineage_verified",
        "task": task,
        "workspace": str(workspace),
        "outer_root": str(outer_root),
        "rounds": rounds,
        "endpoint_score": endpoint,
        "reward_loader_sha256": reward_sha,
        "excluded_historical_rounds": excluded.get(
            "rounds_with_false_checkpoint_credit"
        ),
        "historical_false_checkpoint_credits": historical_false_credits,
        "checkpoint_chain": checkpoint_chain,
        "lineage": lineage,
    }


def audit_core_clean_proposer_lineage(
    payload: dict[str, Any], task: str, kind: str, *, require_complete: bool
) -> dict[str, Any]:
    """Close the cadence-matched AHC039/EPLB proposer replacement lineage.

    The live renderer is allowed to retain the earlier locally ledgered curve
    while these replacements are still starting.  A strict/final audit is not:
    it must observe one task-isolated K=8/max-evals=20 optimizer lineage, rooted
    at the shared H2 score but showing ``task.initial_program`` to the first H1
    batch.  This function deliberately does not call the old multi-restart
    AHC039 curve a matched ablation.
    """
    assert (task, kind) in (
        (AHC039_TASK, "ahc039"),
        (EPLB_TASK, "eplb"),
    )
    expected_workspace = RUN_ROOT / f"proposer_sota5_{kind}_clean_v1"
    expected_outer_root = RUN_ROOT / f"outer-proposer-sota5-{kind}-clean-v1"
    task_payload = payload["tasks"][task]
    observed_workspace = Path(str(task_payload.get("proposer_workspace") or ""))
    observed_outer_root = Path(str(task_payload.get("proposer_outer_root") or ""))
    route_is_clean = (
        observed_workspace.name == expected_workspace.name
        and observed_outer_root.name == expected_outer_root.name
    )
    if not route_is_clean:
        assert not require_complete, (
            f"{task}: final curve still uses {observed_workspace} / "
            f"{observed_outer_root}, not the cadence-matched clean replacement"
        )
        return {
            "status": "pending_clean_route_not_yet_selected",
            "task": task,
            "expected_workspace": str(expected_workspace),
            "expected_outer_root": str(expected_outer_root),
            "observed_workspace": str(observed_workspace),
            "observed_outer_root": str(observed_outer_root),
            "historical_fallback_allowed_only_for_live_render": True,
        }

    proposer = task_payload["series"]["proposer_full"]
    adaptive = [
        point for point in proposer["points"]
        if point.get("round") is not None
    ]
    if not adaptive:
        assert not require_complete, f"{task}: clean proposer has no adaptive point"
        return {
            "status": "pending_no_completed_clean_round",
            "task": task,
            "workspace": str(expected_workspace),
            "outer_root": str(expected_outer_root),
        }
    rounds = [int(point["round"]) for point in adaptive]
    jobs = [int(point["job"]) for point in adaptive]
    assert rounds == sorted(set(rounds))
    assert len(jobs) == len(set(jobs))

    manifest = json.loads((expected_workspace / "run_manifest.json").read_text())
    assert int(manifest.get("schema") or 0) == 1
    assert manifest.get("method") == "update proposer weights (ours)"
    assert manifest.get("task") == task
    assert manifest.get("isolated_feedback") is True
    assert manifest.get("curated_notes_allowed") is False
    assert manifest.get("first_adaptive_batch_program") == "task.initial_program"
    assert manifest.get("h2_program_inherited") is False
    assert Path(manifest["outer_root"]).resolve() == expected_outer_root.resolve()
    assert int(manifest.get("round_base") or -1) == {
        "ahc039": 1300, "eplb": 1320,
    }[kind]
    assert int(manifest.get("k_per_round") or 0) == 8
    assert int(manifest.get("max_evals_per_trajectory") or 0) == 20
    assert int(manifest.get("eval_timeout_seconds") or 0) == 420
    assert manifest.get("proposer_model") == "Qwen3.5-9B"
    assert manifest.get("executor_model") == "Qwen3.5-9B frozen base"
    assert "excluded from the primary clean" in str(
        manifest.get("superseded_curve_treatment") or ""
    )

    source_paths = {
        "eval_worker_sha256": REPO / "src/inner/evaluation/_eval_worker.py",
        "reward_loader_sha256": REPO / "src/outer/reward/rewards.py",
        "outer_round_sha256": REPO / "src/outer/rounds/outer_round.py",
        "fixed_h1_agent_yaml_sha256": REPO / "src/outer/harness/agent.yaml",
    }
    stored_hashes = manifest.get("source_hashes") or {}
    verified_hashes: dict[str, str] = {}
    for key, path in source_paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert stored_hashes.get(key) == actual, f"{task}: source drift in {key}"
        verified_hashes[key] = actual

    training = manifest.get("training") or {}
    assert int(training.get("lora_rank") or 0) == 64
    assert int(training.get("lora_alpha") or 0) == 128
    assert int(training.get("epochs") or 0) == 3
    assert math.isclose(float(training.get("learning_rate") or 0.0), 3e-5)
    assert math.isclose(float(training.get("kl_coefficient") or 0.0), 0.05)
    assert int(training.get("global_batch_size") or 0) == 8
    assert int(training.get("archive_mix", -1)) == 0
    assert training.get("adapter_state") == (
        "continued from the preceding proposer update"
    )
    assert training.get("optimizer_state") == (
        "reinitialized for every training job"
    )
    assert training.get("scheduler_state") == (
        "reinitialized for every training job"
    )

    guard_fields = (payload.get("protocol_guard_proof") or {}).get("fields") or {}
    guard_proof = payload.get("protocol_guard_proof") or {}
    assert guard_proof.get("matches_current_worker") is True, (
        f"{task}: protocol guard proof does not bind the current evaluator"
    )
    proof_job = int(guard_fields.get("job") or 0)
    assert proof_job > 0 and all(job > proof_job for job in jobs), (
        f"{task}: a clean proposer round predates the current guard proof"
    )
    if task == EPLB_TASK:
        assert guard_fields.get("eplb_topology_guard") == "ok"
        assert "topology guard" in str(manifest.get("validity_guard") or "")
    else:
        assert manifest.get("validity_guard") == (
            "aarch64-native official 150-case tester"
        )

    checkpoint_chain = audit_clean_proposer_checkpoint_chain(
        task=task,
        adaptive=adaptive,
        workspace=expected_workspace,
        outer_root=expected_outer_root,
        require_complete=require_complete,
    )

    baseline = json.loads(
        (REPO / "results/baseline_h2_20ev.json").read_text()
    )["baseline"][task]
    incumbent = float(baseline["h2_best"])
    assert math.isclose(
        float(manifest.get("shared_h2_score")), incumbent, abs_tol=1e-12
    )
    incumbent_package = str(REPO / "src/inner/harness")
    initial_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
    lineage: list[dict[str, Any]] = []
    for index, point in enumerate(adaptive):
        round_id = int(point["round"])
        round_dir = expected_outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        summary = json.loads((round_dir / "round_summary.json").read_text())
        prompt = str(json.loads(
            (round_dir / "prompts.json").read_text()
        )[task])
        assert int(meta.get("round") or -1) == round_id
        assert meta.get("tasks_order") == [task]
        assert int(meta.get("k") or 0) == 8
        assert int(meta.get("max_evals") or 0) == 20
        assert str((meta.get("proposer") or {}).get("model")) == "qwen3.5-9b"
        base = meta["bases_in"][task]
        assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12)
        assert Path(base["package"]).resolve() == Path(
            incumbent_package
        ).resolve()
        if index == 0:
            assert prompt_seed_excerpt(prompt) == initial_excerpt, (
                f"{task}: first clean proposer batch did not show "
                "task.initial_program"
            )

        group = summary["groups"][task]
        assert math.isclose(float(group["base_score"]), incumbent, abs_tol=1e-12)
        best_k = group.get("best_k")
        best_score = group.get("best_score")
        selected_program_sha = None
        if group.get("improved"):
            assert best_k is not None and float(best_score) > incumbent
            selected = [
                row for row in group.get("rows") or []
                if int(row["k"]) == int(best_k)
            ]
            assert len(selected) == 1 and selected[0].get("valid") is True
            assert math.isclose(
                float(selected[0]["score"]), float(best_score), abs_tol=1e-12
            )
            candidate_root = (
                round_dir / "rollouts" / task / f"cand{int(best_k):02d}"
            )
            summaries = sorted(candidate_root.glob("*/summary.json"))
            assert summaries, f"{task}/round{round_id}: selected summary missing"
            selected_summary = json.loads(
                max(summaries, key=lambda path: path.stat().st_mtime).read_text()
            )
            rows = selected_summary if isinstance(selected_summary, list) else [
                selected_summary
            ]
            task_rows = [row for row in rows if row.get("task_id") == task]
            assert len(task_rows) == 1 and task_rows[0].get("best_program")
            selected_program_sha = hashlib.sha256(
                str(task_rows[0]["best_program"]).encode()
            ).hexdigest()
            incumbent = float(best_score)
            incumbent_package = str(
                round_dir / "tasks" / task / f"cand{int(best_k):02d}"
            )

        next_base = json.loads(
            (round_dir / "next_bases.json").read_text()
        )[task]
        assert math.isclose(float(next_base["score"]), incumbent, abs_tol=1e-12)
        assert Path(next_base["package"]).resolve() == Path(
            incumbent_package
        ).resolve()
        lineage.append({
            "round": round_id,
            "job": int(point["job"]),
            "launched": int(point.get("launched") or 0),
            "max_evals_per_trajectory": int(
                point.get("max_evals_per_trajectory") or 0
            ),
            "base_score": float(group["base_score"]),
            "best_k": best_k,
            "best_score": float(best_score) if best_score is not None else None,
            "improved": bool(group.get("improved")),
            "selected_program_sha256": selected_program_sha,
            "next_incumbent_score": incumbent,
            "next_incumbent_package": incumbent_package,
        })

    endpoint = float(proposer["points"][-1]["score"])
    assert math.isclose(endpoint, incumbent, abs_tol=1e-12)
    review_path = expected_workspace / "plateau_review.json"
    review = json.loads(review_path.read_text()) if review_path.is_file() else None
    if require_complete:
        assert (expected_workspace / "CANONICAL_COMPLETE").is_file(), (
            f"{task}: clean proposer has not passed plateau/cap review"
        )
        assert review is not None
        assert review.get("status") in (
            "three_transition_empirical_plateau",
            "budget_limited_at_explicit_cap",
        )
        assert int(review.get("completed_round") or -1) == rounds[-1]
        assert math.isclose(
            endpoint,
            float(task_payload["reported_proposer_combined_score"]),
            abs_tol=1e-12,
        )

    return {
        "status": (
            "clean_cadence_matched_lineage_verified_complete"
            if review is not None and (expected_workspace / "CANONICAL_COMPLETE").is_file()
            else "clean_cadence_matched_lineage_verified_live"
        ),
        "task": task,
        "workspace": str(expected_workspace),
        "outer_root": str(expected_outer_root),
        "rounds": rounds,
        "jobs": jobs,
        "common_start_h2_score": float(baseline["h2_best"]),
        "first_round_seed_program": str(INITIAL_PROGRAMS[task]),
        "first_round_seed_sha256": hashlib.sha256(
            initial_excerpt.encode()
        ).hexdigest(),
        "endpoint_score": endpoint,
        "source_hashes": verified_hashes,
        "validity_guard": manifest["validity_guard"],
        "plateau_review": review,
        "checkpoint_chain": checkpoint_chain,
        "lineage": lineage,
        "superseded_historical_curve_in_primary": False,
    }


def audit_rewardfix_context_lineage(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Verify the post-reward-fix PRISM frozen-weight context rerun.

    AHC058 is deliberately not accepted from this shared workspace: round1101
    improved without an attached analyzer brief and changed the subsequent
    task-local ratchet.  Its clean replacement is audited separately by
    :func:`audit_ahc058_analysis_required_context_lineage`.
    """
    tasks = (PRISM_TASK,)
    campaign_tasks = ("eft__ahc_simpletes__ahc058", PRISM_TASK)
    workspace = RUN_ROOT / "context_sota7_rewardfix_v1"
    run_manifest = json.loads((workspace / "run_manifest.json").read_text())
    integrity = json.loads((workspace / "integrity_manifest.json").read_text())
    outer_root = Path(run_manifest["outer_root"])
    assert outer_root.name == "outer-context-sota7-rewardfix-v1"
    assert run_manifest.get("method") == (
        "context-only analyzer; frozen proposer and executor weights"
    )
    assert set(run_manifest.get("tasks") or []) == set(campaign_tasks)
    assert int(run_manifest.get("k_per_round") or 0) == 8
    assert int(run_manifest.get("max_evals_per_trajectory") or 0) == 20
    assert integrity.get("terminal_summary_authoritative") is True
    assert integrity.get(
        "checkpoint_fallback_only_without_terminal_task_row"
    ) is True
    reward_sha = hashlib.sha256(
        (REPO / "src/outer/reward/rewards.py").read_bytes()
    ).hexdigest()
    assert integrity.get("reward_loader_sha256") == reward_sha
    if require_complete:
        assert (workspace / "CANONICAL_CONTEXT_COMPLETE").is_file(), \
            "reward-fix context campaign lacks plateau/cap review"

    baseline = json.loads(
        (REPO / "results/baseline_h2_20ev.json").read_text()
    )["baseline"]
    task_results: dict[str, Any] = {}
    for task in tasks:
        task_payload = payload["tasks"][task]
        assert task_payload.get("context_workspace") == workspace.name
        context = task_payload["series"]["context"]
        adaptive = [point for point in context["points"]
                    if point.get("round") is not None]
        assert adaptive, f"clean context curve has no adaptive {task} point"
        rounds = [int(point["round"]) for point in adaptive]
        assert rounds == sorted(set(rounds))
        incumbent = float(baseline[task]["h2_best"])
        incumbent_package = str(REPO / "src/inner/harness")
        initial_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
        lineage = []
        for index, round_id in enumerate(rounds):
            round_dir = outer_root / f"round{round_id:03d}"
            meta = json.loads((round_dir / "round.json").read_text())
            summary = json.loads((round_dir / "round_summary.json").read_text())
            prompt = str(json.loads(
                (round_dir / "prompts.json").read_text()
            )[task])
            assert set(meta.get("tasks_order") or []) == set(campaign_tasks)
            assert int(meta.get("k") or 0) == 8
            assert int(meta.get("max_evals") or 0) == 20
            base = meta["bases_in"][task]
            assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12)
            assert Path(base["package"]).resolve() == Path(
                incumbent_package
            ).resolve()
            if index == 0:
                assert prompt_seed_excerpt(prompt) == initial_excerpt
            group = summary["groups"][task]
            assert math.isclose(
                float(group["base_score"]), incumbent, abs_tol=1e-12
            )
            checked = 0
            failed = 0
            for row in group.get("rows") or []:
                if not row.get("valid"):
                    continue
                k = int(row["k"])
                credited = row.get("score")
                terminal = authoritative_terminal_score(round_dir, task, k)
                assert (credited is None) == (terminal is None), \
                    f"{task} round{round_id}/cand{k}: false checkpoint credit"
                if credited is not None:
                    assert math.isclose(
                        float(credited), float(terminal),
                        rel_tol=1e-12, abs_tol=1e-12,
                    )
                else:
                    failed += 1
                checked += 1
            best_k = group.get("best_k")
            best_score = group.get("best_score")
            if group.get("improved"):
                assert best_k is not None and float(best_score) > incumbent
                incumbent = float(best_score)
                incumbent_package = str(
                    round_dir / "tasks" / task / f"cand{int(best_k):02d}"
                )
            next_base = json.loads(
                (round_dir / "next_bases.json").read_text()
            )[task]
            assert math.isclose(
                float(next_base["score"]), incumbent, abs_tol=1e-12
            )
            assert Path(next_base["package"]).resolve() == Path(
                incumbent_package
            ).resolve()
            lineage.append({
                "round": round_id,
                "rows_checked": checked,
                "terminal_failures_correctly_uncredited": failed,
                "best_score": best_score,
                "next_incumbent_score": incumbent,
            })
        endpoint = float(context["points"][-1]["score"])
        assert math.isclose(endpoint, incumbent, abs_tol=1e-12)
        task_results[task] = {
            "rounds": rounds,
            "endpoint_score": endpoint,
            "lineage": lineage,
        }
    historical_specs = {
        "eft__ahc_simpletes__ahc058": (
            RUN_ROOT / "outer-context-sota5-ahc-clean", 2, (1,)
        ),
        PRISM_TASK: (
            RUN_ROOT / "outer-context-sota5-sys-guarded", 2, (1, 5, 7)
        ),
    }
    historical_false_credits = {}
    for task, (root, round_id, candidates) in historical_specs.items():
        round_dir = root / f"round{round_id:03d}"
        group = json.loads(
            (round_dir / "round_summary.json").read_text()
        )["groups"][task]
        rows = {int(row["k"]): row for row in group.get("rows") or []}
        evidence = []
        for k in candidates:
            assert rows[k].get("score") is not None
            assert authoritative_terminal_score(round_dir, task, k) is None
            evidence.append({
                "round": round_id, "k": k,
                "credited_seed_checkpoint_score": float(rows[k]["score"]),
                "authoritative_terminal_score": None,
            })
        historical_false_credits[task] = evidence
    return {
        "status": "clean_frozen_weight_terminal_attribution_verified",
        "workspace": str(workspace),
        "outer_root": str(outer_root),
        "reward_loader_sha256": reward_sha,
        "excluded_historical_false_checkpoint_credits": (
            historical_false_credits
        ),
        "tasks": task_results,
    }


def audit_ahc058_analysis_required_context_lineage(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Verify the AHC058 fork where every post-cold round has an analyzer.

    The ordinary analyzer is intentionally fail-open.  That behavior produced
    a useful but inadmissible round1101 result without an AHC058 brief.  The
    replacement controller retries from the pre-round local state and archives
    every missing-brief attempt, so no rejected harness can enter the ratchet.
    """
    task = "eft__ahc_simpletes__ahc058"
    workspace = RUN_ROOT / "context_sota7_ahc058_analysis_required_v1"
    manifest_path = workspace / "run_manifest.json"
    integrity_path = workspace / "integrity_manifest.json"
    task_payload = payload["tasks"][task]
    assert task_payload.get("context_workspace") == workspace.name

    if not manifest_path.is_file():
        assert not require_complete, "analysis-required AHC058 context run is absent"
        return {
            "status": "pending_clean_recovery_submission",
            "workspace": str(workspace),
            "excluded_workspace": str(RUN_ROOT / "context_sota7_rewardfix_v1"),
            "excluded_round": 1101,
        }

    run_manifest = json.loads(manifest_path.read_text())
    integrity = json.loads(integrity_path.read_text())
    outer_root = Path(run_manifest["outer_root"])
    assert outer_root.name == "outer-context-sota7-ahc058-analysis-required-v1"
    assert run_manifest.get("method") == (
        "context-only analyzer; frozen proposer and executor weights"
    )
    assert run_manifest.get("tasks") == [task]
    assert int(run_manifest.get("k_per_round") or 0) == 8
    assert int(run_manifest.get("max_evals_per_trajectory") or 0) == 20
    assert integrity.get("method") == (
        "context/analyzer; proposer and executor weights frozen"
    )
    assert integrity.get("first_adaptive_batch_program") == "task.initial_program"
    assert integrity.get("h2_program_inherited") is False
    assert integrity.get("missing_brief_policy") == (
        "reject whole round, restore pre-round local state, retry"
    )
    excluded = integrity.get("excluded_lineage") or {}
    assert excluded.get("workspace") == "context_sota7_rewardfix_v1"
    assert int(excluded.get("first_invalid_round") or -1) == 1101
    worker_sha = hashlib.sha256(
        (REPO / "src/inner/evaluation/_eval_worker.py").read_bytes()
    ).hexdigest()
    assert integrity.get("eval_worker_sha256") == worker_sha
    if require_complete:
        assert (workspace / "CANONICAL_CONTEXT_COMPLETE").is_file(), \
            "analysis-required AHC058 context campaign lacks plateau/cap review"

    context = task_payload["series"]["context"]
    adaptive = [point for point in context["points"]
                if point.get("round") is not None]
    if not adaptive:
        assert not require_complete, "analysis-required AHC058 curve is empty"
        return {
            "status": "clean_recovery_started_no_completed_round",
            "workspace": str(workspace),
            "outer_root": str(outer_root),
            "integrity_manifest": str(integrity_path),
            "excluded_round": 1101,
        }

    rounds = [int(point["round"]) for point in adaptive]
    assert rounds == sorted(set(rounds))
    cold_round = int(integrity["cold_round_may_omit_analyzer"])
    assert rounds[0] == cold_round
    assert int(integrity["analysis_required_from_round"]) == cold_round + 1
    baseline = json.loads(
        (REPO / "results/baseline_h2_20ev.json").read_text()
    )["baseline"]
    incumbent = float(baseline[task]["h2_best"])
    incumbent_package = str(REPO / "src/inner/harness")
    initial_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
    lineage = []
    for index, point in enumerate(adaptive):
        round_id = int(point["round"])
        round_dir = outer_root / f"round{round_id:03d}"
        meta = json.loads((round_dir / "round.json").read_text())
        summary = json.loads((round_dir / "round_summary.json").read_text())
        prompt = str(json.loads((round_dir / "prompts.json").read_text())[task])
        assert meta.get("tasks_order") == [task]
        assert int(meta.get("k") or 0) == 8
        assert int(meta.get("max_evals") or 0) == 20
        base = meta["bases_in"][task]
        assert math.isclose(float(base["score"]), incumbent, abs_tol=1e-12)
        assert Path(base["package"]).resolve() == Path(incumbent_package).resolve()
        if index == 0:
            assert prompt_seed_excerpt(prompt) == initial_excerpt
            assert int(point.get("analyst_briefs") or 0) == 0
        else:
            briefs = int(point.get("analyst_briefs") or 0)
            assert briefs >= 1, f"AHC058 round{round_id} lacks analyzer evidence"
            assert int(point.get("analyzer_model_calls") or 0) == 2 * briefs
            job = str(point.get("job") or "")
            assert job.isdigit()
            slurm_log = Path(
                "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm"
            ) / f"sah-outer-{job}.out"
            marker = f"[propose] {task}: analysis brief attached"
            assert slurm_log.read_text(errors="ignore").count(marker) == briefs

        group = summary["groups"][task]
        assert math.isclose(float(group["base_score"]), incumbent, abs_tol=1e-12)
        checked = 0
        terminal_failures = 0
        for row in group.get("rows") or []:
            if not row.get("valid"):
                continue
            k = int(row["k"])
            credited = row.get("score")
            terminal = authoritative_terminal_score(round_dir, task, k)
            assert (credited is None) == (terminal is None), \
                f"AHC058 round{round_id}/cand{k}: false checkpoint credit"
            if credited is not None:
                assert math.isclose(
                    float(credited), float(terminal),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
            else:
                terminal_failures += 1
            checked += 1
        best_k = group.get("best_k")
        best_score = group.get("best_score")
        if group.get("improved"):
            assert best_k is not None and float(best_score) > incumbent
            incumbent = float(best_score)
            incumbent_package = str(
                round_dir / "tasks" / task / f"cand{int(best_k):02d}"
            )
        next_base = json.loads(
            (round_dir / "next_bases.json").read_text()
        )[task]
        assert math.isclose(float(next_base["score"]), incumbent, abs_tol=1e-12)
        assert Path(next_base["package"]).resolve() == Path(
            incumbent_package
        ).resolve()
        lineage.append({
            "round": round_id,
            "job": int(point["job"]),
            "analyst_briefs": int(point.get("analyst_briefs") or 0),
            "rows_checked": checked,
            "terminal_failures_correctly_uncredited": terminal_failures,
            "best_score": best_score,
            "next_incumbent_score": incumbent,
        })

    endpoint = float(context["points"][-1]["score"])
    assert math.isclose(endpoint, incumbent, abs_tol=1e-12)

    # Prove why the tempting 1.2732477 result is excluded.
    old_round = RUN_ROOT / "outer-context-sota7-rewardfix-v1/round1101"
    old_group = json.loads(
        (old_round / "round_summary.json").read_text()
    )["groups"][task]
    assert old_group.get("improved") is True
    old_log = Path(
        "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/sah-outer-2816257.out"
    ).read_text(errors="ignore")
    assert f"[propose] {task}: analysis brief attached" not in old_log

    retries = []
    retry_ledger = workspace / "analysis_required_retries.jsonl"
    if retry_ledger.is_file():
        for line in retry_ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row.get("accepted_score_evidence") is False
            assert row.get("charge_full_allocation_as_run") is True
            assert Path(row["archive"]).is_dir()
            retries.append(row)
    return {
        "status": "clean_analysis_required_lineage_verified",
        "workspace": str(workspace),
        "outer_root": str(outer_root),
        "rounds": rounds,
        "endpoint_score": endpoint,
        "excluded_fail_open_round": 1101,
        "excluded_fail_open_score": float(old_group["best_score"]),
        "rejected_attempts_charged_as_run": retries,
        "lineage": lineage,
    }


def audit_all_plotted_reward_attribution(
    payload: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    """Every plotted H1 route must agree with its terminal summaries."""
    result: dict[str, Any] = {}
    for task in TASKS:
        task_result = {}
        for method, series_name in (
            ("proposer", "proposer_full"), ("context", "context")
        ):
            records = []
            cost_only_failed_rounds = []
            points = payload["tasks"][task]["series"][series_name]["points"]
            previous_score = float(points[0]["score"])
            for point in points:
                if point.get("round") is None:
                    continue
                source = Path(str(point.get("source") or ""))
                if point.get("status") == "failed_without_round_summary":
                    # A launched round can leave complete trajectory summaries
                    # yet fail before the collector writes reward rows.  It is
                    # legitimate x/cost evidence, but it must not improve the
                    # best-so-far curve or be treated as attributed reward.
                    assert method == "proposer"
                    assert source.name == "round.json" and source.is_file()
                    round_dir = source.parent
                    assert not (round_dir / "round_summary.json").exists()
                    assert math.isclose(
                        float(point["score"]), previous_score,
                        rel_tol=0.0, abs_tol=1e-12,
                    ), f"{task}/{method}/round{point['round']}: " \
                       "an uncollected round changed the plotted score"
                    terminal_summaries = list(
                        (round_dir / "rollouts" / task).glob(
                            "cand*/*/summary.json"
                        )
                    )
                    assert len(terminal_summaries) == int(
                        point.get("recorded_trajectory_summaries") or 0
                    )
                    cost_only_failed_rounds.append({
                        "round": int(point["round"]),
                        "round_ledger": str(source),
                        "launched": int(point.get("launched") or 0),
                        "terminal_summaries": len(terminal_summaries),
                        "score_credited": False,
                    })
                    continue
                round_dir = source.parent if source.name == "round_summary.json" else None
                if round_dir is None or not round_dir.is_dir():
                    if method == "proposer":
                        root = Path(payload["tasks"][task]["proposer_outer_root"])
                    else:
                        workspace = RUN_ROOT / payload["tasks"][task][
                            "context_workspace"
                        ]
                        root = Path(json.loads(
                            (workspace / "run_manifest.json").read_text()
                        )["outer_root"])
                    round_dir = root / f"round{int(point['round']):03d}"
                group = json.loads(
                    (round_dir / "round_summary.json").read_text()
                )["groups"][task]
                rollout_log_dir = round_dir / "rollout_logs"
                if rollout_log_dir.is_dir():
                    launched_logs = list(
                        rollout_log_dir.glob(f"{task}-cand*.log")
                    )
                    assert len(launched_logs) == int(
                        point.get("launched") or 0
                    ), f"{task}/{method}/round{point['round']}: " \
                       "curve launch count does not match rollout logs"
                checked = 0
                terminal_failures = 0
                for row in group.get("rows") or []:
                    if not row.get("valid"):
                        continue
                    k = int(row["k"])
                    credited = row.get("score")
                    terminal = authoritative_terminal_score(
                        round_dir, task, k
                    )
                    assert (credited is None) == (terminal is None), \
                        f"{task}/{method}/round{point['round']}/cand{k}: " \
                        "terminal/credited null mismatch"
                    if credited is not None:
                        assert math.isclose(
                            float(credited), float(terminal),
                            rel_tol=1e-12, abs_tol=1e-12,
                        ), f"{task}/{method}: false checkpoint credit"
                    else:
                        terminal_failures += 1
                    checked += 1
                records.append({
                    "round": int(point["round"]),
                    "rows_checked": checked,
                    "terminal_failures_correctly_uncredited": terminal_failures,
                    "round_summary": str(round_dir / "round_summary.json"),
                })
                previous_score = float(point["score"])
            if require_complete:
                assert records, (
                    f"{task}/{method}: no adaptive reward rows audited"
                )
            task_result[method] = {
                "status": (
                    "terminal_attribution_verified" if records else
                    "pending_no_clean_adaptive_round"
                ),
                "rounds_checked": len(records),
                "rows_checked": sum(row["rows_checked"] for row in records),
                "terminal_failures_correctly_uncredited": sum(
                    row["terminal_failures_correctly_uncredited"]
                    for row in records
                ),
                "records": records,
                "cost_only_failed_rounds": cost_only_failed_rounds,
            }
        result[task] = task_result
    pending = [
        f"{task}/{method}"
        for task, methods in result.items()
        for method, row in methods.items()
        if row["status"] != "terminal_attribution_verified"
    ]
    return {
        "status": (
            "all_plotted_h1_rewards_match_terminal_summaries"
            if not pending else "pending_clean_adaptive_rewards"
        ),
        "pending": pending,
        "tasks": result,
    }


def audit_proposer_prompt_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify that no plotted proposer-weight point received a curated note."""
    result: dict[str, Any] = {}
    for task in TASKS:
        task_payload = payload["tasks"][task]
        outer_root = Path(task_payload["proposer_outer_root"])
        rounds = [
            int(point["round"])
            for point in task_payload["series"]["proposer_full"]["points"]
            if point.get("round") is not None
        ]
        records = []
        initial_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
        for index, round_id in enumerate(rounds):
            round_dir = outer_root / f"round{round_id:03d}"
            path = round_dir / "prompts.json"
            prompts = json.loads(path.read_text())
            meta = json.loads((round_dir / "round.json").read_text())
            assert task in prompts, f"{task}/round{round_id}: missing serialized H1 prompt"
            assert meta.get("tasks_order") == [task], \
                f"{task}/round{round_id}: proposer campaign is not task isolated"
            prompt = str(prompts[task])
            assert "analyst note:" not in prompt.lower(), \
                f"{task}/round{round_id}: proposer-weight curve contains a curated note"
            if index == 0:
                assert prompt_seed_excerpt(prompt) == initial_excerpt, \
                    f"{task}: first proposer batch did not start from task.initial_program"
            records.append({
                "round": round_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "seed_excerpt_sha256": hashlib.sha256(
                    prompt_seed_excerpt(prompt).encode()).hexdigest(),
            })
        result[task] = {
            "outer_root": str(outer_root),
            "serialized_prompts_checked": len(records),
            "curated_notes_found": 0,
            "every_plotted_round_task_isolated": True,
            "first_batch_initial_program_verified": bool(records),
            "records": records,
        }
    return result


def at_budget(points: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    eligible = [p for p in points if int(p["x"]) <= budget]
    if not eligible:
        raise AssertionError(f"no point at or before x={budget}")
    return eligible[-1]


def ratio(score: float, reference: float) -> float:
    return float(score) / float(reference)


def log_step_auc(points: list[dict[str, Any]], budget: int,
                 reference: float) -> float:
    """Previous-value AUC; a batch earns its gain only when it completes."""
    pts = [(max(1, int(p["x"])), ratio(p["score"], reference))
           for p in points if int(p["x"]) <= budget]
    if not pts or budget <= 1:
        return pts[-1][1] if pts else 0.0
    if pts[0][0] != 1:
        pts.insert(0, (1, pts[0][1]))
    area = 0.0
    for index, (x, value) in enumerate(pts):
        right = pts[index + 1][0] if index + 1 < len(pts) else budget
        if right > x:
            area += value * (math.log(right) - math.log(x))
    return area / math.log(budget)


def first_crossing(points: list[dict[str, Any]], reference: float) -> int | None:
    for point in points:
        if float(point["score"]) >= reference:
            return int(point["x"])
    return None


def non_anchor(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in points
            if p.get("round") is not None or p.get("step") is not None]


def through_budget(points: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return [p for p in non_anchor(points) if int(p["x"]) <= budget]


def logical_cost_to_budget(
    method: str, points: list[dict[str, Any]], budget: int
) -> dict[str, Any]:
    """Count exact logical operations needed to produce points through budget.

    This deliberately does not convert heterogeneous inference and training
    operations into a made-up rollout equivalent.
    """
    charged = through_budget(points, budget)
    rollouts = sum(int(p.get("launched") or 0) for p in charged)
    result: dict[str, Any] = {
        "executor_trajectories": rollouts,
        "charged_evaluator_call_budget": sum(
            int(p.get("charged_evaluator_call_budget") or 0) for p in charged),
        "recorded_evaluator_calls_lower_bound": sum(
            int(p.get("recorded_evaluator_calls") or 0) for p in charged),
        "recorded_executor_model_calls_lower_bound": sum(
            int(p.get("recorded_executor_model_calls") or 0) for p in charged),
        "recorded_sandbox_seconds_lower_bound": sum(
            float(p.get("recorded_sandbox_seconds") or 0.0) for p in charged),
        "completed_batches": len(charged),
    }
    if method in ("proposer", "context"):
        proposals = sum(int(p.get("proposed") or 0) for p in charged)
        model_calls = sum(int(p.get("h1_model_calls") or 0) for p in charged)
        reviewer_calls = sum(
            int(p.get("reviewer_model_calls_lower_bound") or 0)
            for p in charged
        )
        result.update({
            "harness_proposals": proposals,
            "proposer_model_calls": model_calls,
            "reviewer_model_calls_lower_bound": reviewer_calls,
            "proposal_attempts_not_reaching_executor": max(0, proposals - rollouts),
            "analyzer_calls": (
                sum(int(p.get("analyzer_model_calls") or 0) for p in charged)
                if method == "context" else 0
            ),
            "weight_updates_used_by_evaluated_batches": (
                sum(1 for p in charged
                    if p.get("phi") and
                    str(p.get("phi")) != BASE.rsplit("/", 1)[-1])
                if method == "proposer" else 0
            ),
        })
    else:
        result.update({
            "harness_proposals": 0,
            "proposer_model_calls": 0,
            "reviewer_model_calls_lower_bound": 0,
            "analyzer_calls": 0,
            "weight_updates_used_by_evaluated_batches": sum(
                1 for p in charged if int(p.get("step") or 0) > 0),
        })
    return result


def point_model_calls(method: str, point: dict[str, Any]) -> int:
    """Recorded model requests needed for a plotted batch.

    Inner-trajectory and H1 ledgers are exact when a summary exists.  A process
    that dies before writing its summary can only make this number larger, so
    the audit labels the resulting budget as a lower bound.
    """
    calls = int(point.get("recorded_executor_model_calls") or 0)
    if method in ("proposer", "context"):
        calls += int(point.get("h1_model_calls") or 0)
        calls += int(point.get("reviewer_model_calls_lower_bound") or 0)
    if method == "context":
        calls += int(point.get("analyzer_model_calls") or 0)
    return calls


def model_call_budget_curve(
    method: str, points: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cumulative = 0
    out = [{"budget": 0, "score": float(points[0]["score"]), "anchor": True}]
    for point in non_anchor(points):
        cumulative += point_model_calls(method, point)
        out.append({
            "budget": cumulative,
            "score": float(point["score"]),
            "anchor": False,
            "source_x": int(point["x"]),
        })
    return out


def evaluator_budget_curve(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cumulative = 0
    out = [{"budget": 0, "score": float(points[0]["score"]), "anchor": True}]
    for point in non_anchor(points):
        cumulative += int(point.get("charged_evaluator_call_budget") or 0)
        out.append({
            "budget": cumulative,
            "score": float(point["score"]),
            "anchor": False,
            "source_x": int(point["x"]),
        })
    return out


def eval_score_at_budget(curve: list[dict[str, Any]], budget: int) -> float:
    eligible = [point for point in curve if int(point["budget"]) <= budget]
    if not eligible:
        raise AssertionError(f"no evaluator-budget point at or before {budget}")
    return float(eligible[-1]["score"])


def eval_first_crossing(
    curve: list[dict[str, Any]], reference: float
) -> int | None:
    for point in curve:
        if float(point["score"]) >= reference:
            return int(point["budget"])
    return None


def eval_log_step_auc(
    curve: list[dict[str, Any]], budget: int, reference: float
) -> float:
    """Previous-value AUC on log(1 + charged evaluator-call budget)."""
    if budget <= 0:
        return ratio(curve[0]["score"], reference)
    pts = [(int(p["budget"]), ratio(p["score"], reference))
           for p in curve if int(p["budget"]) <= budget]
    area = 0.0
    for index, (cost, value) in enumerate(pts):
        right = pts[index + 1][0] if index + 1 < len(pts) else budget
        if right > cost:
            area += value * (math.log1p(right) - math.log1p(cost))
    return area / math.log1p(budget)


def driver_update_count(sources: list[str]) -> tuple[list[str], list[str]]:
    drivers = sorted({s for s in sources if Path(s).name.startswith("driver")
                      and Path(s).suffix == ".log"})
    tags: list[str] = []
    for source in drivers:
        path = Path(source)
        if path.exists():
            tags.extend(re.findall(
                r"trained -> (mphi_[A-Za-z0-9_.-]+)",
                path.read_text(errors="ignore"),
            ))
    return tags, drivers


@functools.lru_cache(maxsize=1)
def proposer_update_job_index() -> dict[str, dict[str, list[str]]]:
    """Recover proposer train/merge job IDs from their immutable Slurm logs."""
    index: dict[str, dict[str, list[str]]] = {}
    patterns = (
        ("train", "wv-lora-*.log", r"^\+ SAVE_CKPT=.*/(mphi_[^\s/]+)$"),
        ("merge", "wv-merge-*.log", r"^\+ OUT=.*/(mphi_[^\s/]+)$"),
    )
    for role, glob_pattern, regex in patterns:
        for path in SLURM_LOG.glob(glob_pattern):
            match = re.search(regex, path.read_text(errors="ignore"), re.MULTILINE)
            job_match = re.search(r"-([0-9]+)\.log$", path.name)
            if not match or not job_match:
                continue
            row = index.setdefault(match.group(1), {"train": [], "merge": []})
            row[role].append(job_match.group(1))
    for row in index.values():
        for role in row:
            row[role] = sorted(set(row[role]), key=int)
    return index


def proposer_weight_job_ledger(tags: list[str]) -> dict[str, list[str]]:
    index = proposer_update_job_index()
    result = {"train": [], "merge": [], "unmapped_tags": []}
    for tag in sorted(set(tags)):
        row = index.get(tag)
        if not row:
            result["unmapped_tags"].append(tag)
            continue
        result["train"].extend(row["train"])
        result["merge"].extend(row["merge"])
    for role in ("train", "merge"):
        result[role] = sorted(set(result[role]), key=int)
    return result


def log_timing(job: str, role: str) -> dict[str, Any] | None:
    patterns = {
        "outer": f"sah-outer-{job}.out",
        "eval": f"ttt12-eval-{job}.out",
        "train": f"wv-lora-{job}.log",
        "merge": f"wv-merge-{job}.log",
    }
    path = SLURM_LOG / patterns[role]
    if not path.exists():
        return None
    try:
        birth = int(subprocess.check_output(
            ["stat", "-c", "%W", str(path)], text=True).strip())
    except Exception:
        birth = 0
    end = int(path.stat().st_mtime)
    if birth <= 0 or end < birth:
        return None
    elapsed = end - birth
    return {
        "job": job,
        "role": role,
        "log": str(path),
        "start_epoch": birth,
        "last_write_epoch": end,
        "elapsed_seconds_proxy": elapsed,
        "allocated_gpus": 4,
        "allocated_gpu_hours_proxy": 4 * elapsed / 3600.0,
    }


@functools.cache
def frozen_sacct_timing_index() -> dict[str, dict[str, Any]]:
    """Load a reproducible authoritative accounting snapshot when present."""
    if not SACCT_SNAPSHOT.is_file():
        return {}
    payload = json.loads(SACCT_SNAPSHOT.read_text())
    assert int(payload.get("schema") or 0) == 1, \
        "unexpected SOTA7 sacct snapshot schema"
    collected_at = str(payload.get("collected_at") or "")
    assert collected_at, "sacct snapshot lacks collected_at"
    rows = payload.get("rows") or {}
    assert isinstance(rows, dict)
    result: dict[str, dict[str, Any]] = {}
    for job, source in rows.items():
        assert str(job).isdigit(), f"invalid sacct snapshot job id: {job}"
        assert str(source.get("job")) == str(job)
        elapsed = int(source["elapsed_seconds_sacct"])
        allocated_gpus = int(source["allocated_gpus_sacct"])
        allocated_hours = float(source["allocated_gpu_hours_sacct"])
        assert elapsed >= 0 and allocated_gpus >= 0
        assert math.isclose(
            allocated_hours,
            elapsed * allocated_gpus / 3600.0,
            rel_tol=1e-12, abs_tol=1e-12,
        ), f"sacct snapshot GPU-hour arithmetic changed for job {job}"
        result[str(job)] = {
            **source,
            "job": str(job),
            "elapsed_seconds_sacct": elapsed,
            "allocated_gpus_sacct": allocated_gpus,
            "allocated_gpu_hours_sacct": allocated_hours,
            "accounting_source": "frozen_sacct_snapshot",
            "snapshot": str(SACCT_SNAPSHOT),
            "snapshot_collected_at": collected_at,
        }
    requested = {str(job) for job in payload.get("requested_jobs") or []}
    if payload.get("status") == "complete":
        assert requested == set(result), \
            "complete sacct snapshot requested_jobs/rows mismatch"
    else:
        assert set(result) <= requested, \
            "partial sacct snapshot contains an unrequested job"
    return result


def sacct_timing_index(jobs: list[str]) -> dict[str, dict[str, Any]]:
    """Recover authoritative allocation elapsed time when Slurm retains it.

    ``sacct`` is a cluster service and may be unavailable when the audit is run
    off-cluster.  The immutable-log proxy remains the portable fallback, but on
    cluster we record both and prefer neither silently.
    """
    unique = sorted(set(jobs), key=int)
    frozen = frozen_sacct_timing_index()
    result: dict[str, dict[str, Any]] = {
        job: dict(frozen[job]) for job in unique if job in frozen
    }
    missing_from_snapshot = [job for job in unique if job not in result]
    for offset in range(0, len(missing_from_snapshot), 100):
        chunk = missing_from_snapshot[offset:offset + 100]
        try:
            proc = subprocess.run(
                [
                    "sacct", "-X", "-n", "-P", "-j", ",".join(chunk),
                    "-o", "JobIDRaw,State,ElapsedRaw,AllocTRES%160,Start,End",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        wanted = set(chunk)
        for line in proc.stdout.splitlines():
            fields = line.split("|")
            if len(fields) < 6:
                continue
            job, state, elapsed_text, alloc_tres, start, end = fields[:6]
            if job not in wanted or not elapsed_text.isdigit():
                continue
            gpu_match = re.search(
                r"(?:^|,)gres/gpu(?::[^=,]+)?=([0-9]+)(?:,|$)",
                alloc_tres,
            )
            allocated_gpus = int(gpu_match.group(1)) if gpu_match else 0
            elapsed = int(elapsed_text)
            result[job] = {
                "job": job,
                "state": state,
                "start": start,
                "end": end,
                "elapsed_seconds_sacct": elapsed,
                "alloc_tres": alloc_tres,
                "allocated_gpus_sacct": allocated_gpus,
                "allocated_gpu_hours_sacct": (
                    allocated_gpus * elapsed / 3600.0
                ),
                "accounting_source": "live_sacct",
            }
    return result


def timing_ledger(job_roles: dict[str, list[str]]) -> dict[str, Any]:
    requested = [job for jobs in job_roles.values() for job in jobs]
    sacct_rows = sacct_timing_index(requested)
    rows, missing = [], []
    for role, jobs in job_roles.items():
        for job in jobs:
            row = log_timing(job, role)
            if row is None:
                missing.append({"job": job, "role": role})
                row = {"job": job, "role": role, "log_proxy_missing": True}
            accounting = sacct_rows.get(job)
            if accounting:
                row["slurm_accounting"] = accounting
            rows.append(row)
    sacct_missing = sorted(
        set(requested) - set(sacct_rows), key=int
    )
    return {
        "basis": (
            "authoritative sacct elapsed x allocated GPUs when available, plus "
            "a 4-GPU Slurm-log birth-to-last-write proxy; neither is FLOPs"
        ),
        "jobs_recovered": sum(not row.get("log_proxy_missing", False)
                              for row in rows),
        "jobs_missing": missing,
        "elapsed_job_hours_proxy": sum(
            int(row.get("elapsed_seconds_proxy") or 0) for row in rows
        ) / 3600.0,
        "allocated_gpu_hours_proxy": sum(
            float(row.get("allocated_gpu_hours_proxy") or 0.0) for row in rows),
        "sacct_jobs_recovered": len(sacct_rows),
        "sacct_jobs_from_frozen_snapshot": sum(
            row.get("accounting_source") == "frozen_sacct_snapshot"
            for row in sacct_rows.values()
        ),
        "sacct_snapshot": (
            str(SACCT_SNAPSHOT) if SACCT_SNAPSHOT.is_file() else None
        ),
        "sacct_jobs_missing": sacct_missing,
        "elapsed_job_hours_sacct": sum(
            int(row["elapsed_seconds_sacct"])
            for row in sacct_rows.values()
        ) / 3600.0,
        "allocated_gpu_hours_sacct": sum(
            float(row["allocated_gpu_hours_sacct"])
            for row in sacct_rows.values()
        ),
        "jobs": rows,
    }


def executor_train_config(job: str) -> dict[str, Any] | None:
    path = SLURM_LOG / f"wv-lora-{job}.log"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    flags = {
        "learning_rate": (r"--lr ([^\s]+)", float),
        "kl_coefficient": (r"--kl-loss-coef ([^\s]+)", float),
        "num_epochs": (r"--num-epoch ([0-9]+)", int),
        "rollout_batch_size": (r"--rollout-batch-size ([0-9]+)", int),
        "global_batch_size": (r"--global-batch-size ([0-9]+)", int),
        "lora_rank": (r"--lora-rank ([0-9]+)", int),
        "lora_alpha": (r"--lora-alpha ([0-9]+)", int),
        "adam_beta2": (r"--adam-beta2 ([^\s]+)", float),
        "weight_decay": (r"--weight-decay ([^\s]+)", float),
    }
    result: dict[str, Any] = {
        "job": job,
        "log": str(path),
        "loads_accumulated_adapter_weights_only": "--load-weights-only" in text,
        "optimizer_state_for_this_job": "fresh",
        "scheduler_state_for_this_job": "fresh",
    }
    parameter_matches = re.findall(
        r"trainable params:\s*([0-9,]+)\s*\|\|\s*all params:\s*([0-9,]+)"
        r"\s*\|\|\s*trainable%:\s*([0-9.]+)",
        text,
    )
    if parameter_matches:
        trainable, total, percent = parameter_matches[-1]
        result.update({
            "trainable_parameters": int(trainable.replace(",", "")),
            "all_parameters": int(total.replace(",", "")),
            "trainable_percent": float(percent),
        })
    for key, (regex, convert) in flags.items():
        matches = re.findall(regex, text)
        # A running Slurm script may have printed the literal shell default
        # (for example ``"${LR:-1e-5}"``) before xtrace records the expanded
        # command.  Use the newest parseable value; the complete audit still
        # fails closed later if a required field never materializes.
        for candidate in reversed(matches):
            try:
                result[key] = convert(candidate.strip("\"'"))
                break
            except (TypeError, ValueError):
                continue
    if all(key in result for key in (
        "num_epochs", "rollout_batch_size", "global_batch_size"
    )):
        result["planned_optimizer_boundaries"] = math.ceil(
            int(result["num_epochs"]) * int(result["rollout_batch_size"]) /
            int(result["global_batch_size"])
        )
    return result


def executor_merge_sanity(job: str) -> dict[str, Any] | None:
    path = SLURM_LOG / f"wv-merge-{job}.log"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.findall(r"\[sanity\].*sum\|B\|=([0-9.eE+-]+)", text)
    if not match:
        return None
    adapter_l1 = float(match[-1])
    return {
        "job": job,
        "log": str(path),
        "lora_b_l1": adapter_l1,
        "nonzero": adapter_l1 > 0.0 and "[sanity] OK" in text,
    }


def executor_job_ledger(sources: list[str]) -> dict[str, list[str]]:
    curves = [Path(s) for s in sources if s.endswith("/curve.jsonl")]
    if len(curves) != 1:
        return {"eval": [], "train": [], "merge": [],
                "failed_eval": [], "failed_train": [], "failed_merge": []}
    tag_dir = curves[0].parent
    result = {"eval": [], "train": [], "merge": [],
              "failed_eval": [], "failed_train": [], "failed_merge": []}
    paths = list(tag_dir.glob("jobs_*.env"))
    paths += list(tag_dir.glob("failed_jobs_*.env"))
    for path in sorted(set(paths)):
        failed = path.name.startswith("failed_jobs_")
        text = path.read_text(errors="ignore")
        for key, bucket in (("EVAL_JOB", "eval"),
                            ("SUPP_EVAL_JOB", "eval"),
                            ("TRAIN_JOB", "train"),
                            ("MERGE_JOB", "merge")):
            match = re.search(rf"^{key}=('?)([0-9]+)\1$", text, re.MULTILINE)
            if match:
                result[bucket].append(match.group(2))
                if failed:
                    result[f"failed_{bucket}"].append(match.group(2))
    # Dynamic top-ups are submitted after the per-update env file is written.
    # Preserve every submission in the job ledger; pre-start cancellations have
    # no Slurm output and therefore contribute zero to the recovered GPU-time
    # proxy, while a materialized top-up (for example the clean SQL k8/k9 job)
    # is charged normally.
    driver = tag_dir / "driver.log"
    if driver.exists():
        result["eval"].extend(re.findall(
            r"eval top-up job=([0-9]+)",
            driver.read_text(errors="ignore"),
        ))
    return {key: sorted(set(value), key=int) for key, value in result.items()}


def audit_ahc039_k32_sensitivity() -> dict[str, Any]:
    """Preserve the stronger, non-canonical large-batch AHC039 counterexample.

    This run predates the canonical K=8/common-warm-start protocol.  Excluding
    it from the main curve is methodologically necessary, but omitting its
    stronger score entirely would make the K=8 choice look cherry-picked.
    """
    root = Path(
        "/lustre/fsw/portfolios/av/users/yingzim/runs/self_adapt_harness/"
        "ttt_discover_clean20/ahc039"
    )
    curve_path = root / "curve.jsonl"
    state_path = root / "state.json"
    assert curve_path.is_file() and state_path.is_file(), \
        "missing AHC039 K~=32 executor sensitivity provenance"
    rows = [json.loads(line) for line in curve_path.read_text().splitlines()
            if line.strip()]
    assert len(rows) >= 2 and [int(row["step"]) for row in rows[:2]] == [0, 1]
    audited = rows[:2]
    assert [int(row["cum_rollouts"]) for row in audited] == [44, 83]

    state = json.loads(state_path.read_text())
    initial_hash = hashlib.sha256(
        INITIAL_PROGRAMS["eft__ahc_simpletes__ahc039"].read_bytes()
    ).hexdigest()
    assert state["archive"]["root"]["program_hash"] == initial_hash, \
        "AHC039 K~=32 sensitivity did not start from task.initial_program"
    assert not state.get("common_warm_start"), \
        "legacy sensitivity unexpectedly claims canonical warm provenance"

    jobs_path = root / "jobs_ttt20_u1.env"
    jobs_text = jobs_path.read_text()
    jobs = {
        key.lower(): re.search(rf"^{key}=([0-9]+)$", jobs_text, re.MULTILINE).group(1)
        for key in ("TRAIN_JOB", "MERGE_JOB", "EVAL_JOB", "SUPP_EVAL_JOB")
    }
    train = executor_train_config(jobs["train_job"])
    merge = executor_merge_sanity(jobs["merge_job"])
    assert train is not None and merge is not None and merge["nonzero"], \
        "AHC039 K~=32 sensitivity lacks a nonzero executor update"
    assert int(train.get("global_batch_size", -1)) == 8
    assert int(train.get("num_epochs", -1)) == 1
    assert int(train.get("lora_rank", -1)) == 32
    assert int(train.get("lora_alpha", -1)) == 64
    assert abs(float(train.get("learning_rate", -1)) - 4e-5) < 1e-12
    assert abs(float(train.get("kl_coefficient", -1)) - 0.1) < 1e-12

    return {
        "name": "older AHC039 K~=32 executor sensitivity",
        "included_in_main_figure": False,
        "curve": str(curve_path),
        "state": str(state_path),
        "audited_completed_rows": audited,
        "endpoint_launched_trajectories": int(audited[-1]["cum_rollouts"]),
        "endpoint_x_if_the_shared_h2_anchor_were_added": int(
            audited[-1]["cum_rollouts"]
        ) + 1,
        "endpoint_score": float(audited[-1]["best"]),
        "initial_program_sha256": initial_hash,
        "training": train,
        "merge_sanity": merge,
        "jobs": jobs,
        "exclusion_reasons": [
            "batch cadence is K~=32 plus AHC top-ups, not canonical K=8",
            "legacy state has no standardized shared-H2 provenance record",
            "legacy PUCT root lacks explicit score-semantics metadata",
            "AHC KL=0.1 and GBS=8 differ from canonical KL=0.01 and GBS=4",
        ],
        "interpretation": (
            "The larger-batch executor can be stronger than the canonical K=8 "
            "curve; it is disclosed as a sensitivity counterexample, not mixed "
            "into the cadence-matched primary comparison."
        ),
    }


def audit_endpoint_validation(
    path: Path, source_plot: Path, *, require_complete: bool
) -> dict[str, Any]:
    """Verify repeated evaluator-only validation of all three route endpoints."""
    if not path.is_file():
        assert not require_complete, f"missing endpoint validation {path}"
        return {"status": "pending", "path": str(path)}
    payload = json.loads(path.read_text())
    cases_path = Path(payload["source_cases"])
    cases = json.loads(cases_path.read_text())
    assert Path(cases["source_plot_data"]).resolve() == source_plot.resolve(), \
        "endpoint cases were collected from a different plot manifest"
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    assert digest == payload.get("source_cases_sha256"), \
        "endpoint case manifest changed after validation"
    expected = {
        f"{task}::{method}"
        for task in TASKS
        for method in ("proposer", "context", "executor")
    }
    case_results = payload.get("case_results") or {}
    if require_complete:
        assert payload.get("status") == "complete", \
            "endpoint validation is not complete"
        assert payload.get("all_runs_valid") is True, \
            "at least one endpoint re-evaluation was invalid"
        assert int(payload.get("requested_runs") or 0) >= 5, \
            "endpoint validation requires at least five repeats"
        assert set(case_results) == expected, \
            f"endpoint validation does not cover all {len(expected)} route endpoints"
    duplicate_program_groups: dict[str, list[list[str]]] = {}
    outside_range: list[str] = []
    for task in TASKS:
        by_hash: dict[str, list[str]] = {}
        for method in ("proposer", "context", "executor"):
            case_id = f"{task}::{method}"
            row = case_results.get(case_id)
            if not row:
                continue
            by_hash.setdefault(str(row["program_sha256"]), []).append(method)
            if not row.get("reported_endpoint_inside_revalidation_range"):
                outside_range.append(case_id)
        groups = [methods for methods in by_hash.values() if len(methods) > 1]
        if groups:
            duplicate_program_groups[task] = groups
    return {
        "status": payload.get("status"),
        "path": str(path),
        "source_cases": str(cases_path),
        "requested_runs": payload.get("requested_runs"),
        "all_runs_valid": payload.get("all_runs_valid"),
        "cases_expected": len(expected),
        "cases_present": len(case_results),
        "reported_endpoints_outside_revalidation_range": outside_range,
        "same_program_shared_by_routes": duplicate_program_groups,
        "cost_treatment": (
            "evaluator-only CPU common measurement overhead after adaptation; "
            "excluded from all three route adaptation GPU-cost ledgers"
        ),
        "interpretation": (
            "shared program hashes expose route ties and stochastic winner's-"
            "curse effects; online best-so-far points are retained as trajectory "
            "evidence, while endpoint statistics must be reported alongside them"
        ),
    }


def audit_txn_executor_timeout_recovery(*, require_complete: bool) -> dict[str, Any]:
    """Verify that the timed-out Txn update is topped up, never erased.

    Update1 launched eight trajectories in job 2814442 but only four reached a
    usable terminal summary before the two-hour Slurm limit.  The canonical
    route must charge those launches and add new, uniquely indexed trajectories
    until K=8 usable replay rows exist.  Re-running the original k0--k7 indices
    or counting the four-row batch as a curve point would both bias the
    executor reference downward.
    """
    state_dir = (
        RUN_ROOT / "ttt_discover_sota7_extra_k8" / "txnsched"
    )
    recovery_path = state_dir / "timeout_u1_recovery_manifest.json"
    submission_path = REPO / "results/txn_timeout_recovery_submission.env"
    if not recovery_path.is_file():
        assert not require_complete, (
            "missing Transaction executor timeout-recovery manifest"
        )
        return {
            "status": "pending",
            "recovery_manifest": str(recovery_path),
            "submission_ledger": str(submission_path),
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == TXN_TASK
    assert int(recovery.get("step") or -1) == 1
    assert int(recovery.get("timed_out_eval_job") or 0) == 2814442
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    assert int(recovery.get("usable_before_timeout") or 0) == 4
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    eval_dir = state_dir / "eval_ttts7k8_u1"
    partials = sorted(eval_dir.glob("eval_manifest.partial_l8_u4.json*"))
    assert len(partials) == 1, (
        f"expected one preserved Txn 8-launch/4-usable manifest, got {partials}"
    )
    partial = json.loads(partials[0].read_text())
    assert partial.get("partial") is True
    assert int(partial.get("target") or 0) == 8
    assert int(partial.get("launched") or 0) == 8
    assert int(partial.get("usable") or 0) == 4
    assert int(partial.get("worker_rc") or 0) == 124

    wrapper_job: int | None = None
    if submission_path.is_file():
        match = re.search(
            r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
        )
        assert match is not None, "Txn recovery submission ledger lacks JOB"
        wrapper_job = int(match.group(1))
    elif require_complete:
        raise AssertionError("missing Transaction recovery submission ledger")

    driver_path = state_dir / "driver.log"
    driver_text = driver_path.read_text(errors="ignore") if driver_path.is_file() else ""
    topup_jobs = [
        int(job) for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    ]
    final_path = eval_dir / "eval_manifest.json"
    final: dict[str, Any] | None = None
    prepared = (state_dir / "prepare_step01.json").is_file()
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        assert final.get("partial") is False
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 12, (
            "Txn final u1 manifest erased one or more timeout/top-up launches"
        )
        assert topup_jobs, "Txn final u1 manifest has no top-up job provenance"
        assert 2818874 in topup_jobs, (
            "Txn final u1 manifest is not tied to the recorded k8--k11 top-up"
        )

    if require_complete:
        assert final is not None, "Transaction executor top-up is incomplete"
        assert prepared, "Transaction executor update1 was not prepared"

    return {
        "status": (
            "complete_and_prepared" if final is not None and prepared else
            "topup_collected_awaiting_prepare" if final is not None else
            "topup_in_progress"
        ),
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "timed_out_eval_job": 2814442,
        "preserved_partial_manifest": str(partials[0]),
        "original_launched": 8,
        "original_usable": 4,
        "topup_jobs": topup_jobs,
        "final_manifest": str(final_path) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all original and uniquely indexed top-up launches are charged on "
            "the executor curve; only eight usable rows enter the replay"
        ),
    }


def audit_txn_executor_u2_timeout_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Verify the independently charged recovery of Txn update2.

    Update2 launched all eight trajectories in job 2820262.  Six terminal
    summaries survived before the two-hour limit; the recovery must retain
    them, reuse the same merged checkpoint, and add fresh indices until eight
    usable replay rows exist.  The CPU wrapper itself is not GPU work, while
    the timed-out allocation and every top-up allocation remain charged.
    """
    state_dir = RUN_ROOT / "ttt_discover_sota7_extra_k8" / "txnsched"
    recovery_path = state_dir / "timeout_u2_recovery_manifest.json"
    submission_path = REPO / "results/txn_u2_timeout_recovery_submission.env"
    if not recovery_path.is_file():
        assert not require_complete, (
            "missing Transaction update2 timeout-recovery manifest"
        )
        return {
            "status": "pending",
            "recovery_manifest": str(recovery_path),
            "submission_ledger": str(submission_path),
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == TXN_TASK
    assert int(recovery.get("step") or -1) == 2
    assert int(recovery.get("timed_out_eval_job") or 0) == 2820262
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    assert int(recovery.get("usable_before_timeout") or 0) == 6
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    eval_dir = state_dir / "eval_ttts7k8_u2"
    partials = sorted(eval_dir.glob("eval_manifest.partial_l8_u6.json*"))
    # Before the resumable driver starts, the partial temporarily occupies the
    # canonical manifest name.  It is archived exactly once before top-up.
    canonical_manifest = eval_dir / "eval_manifest.json"
    canonical_payload = (
        json.loads(canonical_manifest.read_text())
        if canonical_manifest.is_file() else None
    )
    if partials:
        assert len(partials) == 1, (
            f"expected one preserved Txn u2 partial manifest, got {partials}"
        )
        partial_path = partials[0]
        partial = json.loads(partial_path.read_text())
    else:
        assert canonical_payload is not None and canonical_payload.get(
            "partial"
        ) is True, "Txn u2 timeout evidence is neither staged nor archived"
        partial_path = canonical_manifest
        partial = canonical_payload
    assert partial.get("partial") is True
    assert int(partial.get("target") or 0) == 8
    assert int(partial.get("launched") or 0) == 8
    assert int(partial.get("usable") or 0) == 6
    assert int(partial.get("worker_rc") or 0) == 124

    first_wrapper_job: int | None = None
    wrapper_job: int | None = None
    if submission_path.is_file():
        submission_text = submission_path.read_text()
        match = re.search(
            r"^JOB=([0-9]+)$", submission_text, re.MULTILINE
        )
        assert match is not None, "Txn u2 recovery ledger lacks JOB"
        first_wrapper_job = int(match.group(1))
        retry_match = re.search(
            r"^RETRY_JOB=([0-9]+)$", submission_text, re.MULTILINE
        )
        wrapper_job = (
            int(retry_match.group(1))
            if retry_match is not None else first_wrapper_job
        )
        if retry_match is not None:
            assert "FIRST_JOB_RESULT=controller_rc1_before_topup" in (
                submission_text
            )
            assert "RETRY_MODE=unsandboxed_exact_sbatch_after_step_specific_guard_fix" in (
                submission_text
            )
    elif require_complete:
        raise AssertionError("missing Transaction u2 recovery submission ledger")

    # Slurm step logs inherit the wrapper's TMPDIR, giving an immutable link
    # from the CPU recovery controller to every GPU top-up it submitted.
    wrapper_eval_jobs: set[int] = set()
    first_wrapper_eval_jobs: set[int] = set()
    if first_wrapper_job is not None and first_wrapper_job != wrapper_job:
        first_marker = f"/tmp/yingzim/{first_wrapper_job}"
        for path in SLURM_LOG.glob("ttt12-eval-*.err"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if first_marker not in text:
                continue
            match = re.search(r"ttt12-eval-([0-9]+)\.err$", path.name)
            if match:
                first_wrapper_eval_jobs.add(int(match.group(1)))
        assert not first_wrapper_eval_jobs, (
            "the recorded pre-fix Txn u2 wrapper unexpectedly launched GPU work"
        )
    if wrapper_job is not None:
        marker = f"/tmp/yingzim/{wrapper_job}"
        for path in SLURM_LOG.glob("ttt12-eval-*.err"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if marker not in text:
                continue
            match = re.search(r"ttt12-eval-([0-9]+)\.err$", path.name)
            if match:
                wrapper_eval_jobs.add(int(match.group(1)))
    driver_path = state_dir / "driver.log"
    driver_text = (
        driver_path.read_text(errors="ignore")
        if driver_path.is_file() else ""
    )
    logged_topups = {
        int(job)
        for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    }
    topup_jobs = sorted(wrapper_eval_jobs.intersection(logged_topups))

    final: dict[str, Any] | None = None
    if canonical_payload is not None and canonical_payload.get("partial") is False:
        final = canonical_payload
        assert partials, "Txn u2 final manifest replaced an unarchived partial"
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 10, (
            "Txn u2 final manifest erased timeout or top-up launches"
        )
        assert topup_jobs, "Txn u2 final manifest has no wrapper-linked top-up"
    prepared = (state_dir / "prepare_step02.json").is_file()
    if require_complete:
        assert final is not None, "Transaction executor update2 top-up is incomplete"
        assert prepared, "Transaction executor update2 was not prepared"

    return {
        "status": (
            "complete_and_prepared" if final is not None and prepared else
            "topup_collected_awaiting_prepare" if final is not None else
            "topup_in_progress"
        ),
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "failed_cpu_wrapper_job_excluded_from_gpu_hours": (
            first_wrapper_job
            if first_wrapper_job != wrapper_job else None
        ),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "timed_out_eval_job": 2820262,
        "preserved_partial_manifest": str(partial_path),
        "original_launched": 8,
        "original_usable": 6,
        "topup_jobs": topup_jobs,
        "final_manifest": str(canonical_manifest) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all eight original and all uniquely indexed top-up launches are "
            "charged; exactly eight usable rows enter the next replay"
        ),
    }


def audit_hadamard_executor_timeout_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Verify that Hadamard update5 charges its timed-out launch batch.

    Job 2819085 launched eight trajectories and produced seven usable terminal
    summaries before its two-hour limit.  The recovery must preserve that
    partial manifest, reuse the exact update5 checkpoint, and launch a new
    trajectory index until K=8 usable rows exist.  Thus the curve point charges
    at least nine launches even though only eight rows enter replay training.
    """
    state_dir = RUN_ROOT / "ttt_discover_sota7_extra_k8" / "hadamard"
    recovery_path = state_dir / "timeout_u5_recovery_manifest.json"
    submission_path = REPO / "results/hadamard_timeout_recovery_submission.env"
    if not recovery_path.is_file():
        assert not require_complete, (
            "missing Hadamard executor timeout-recovery manifest"
        )
        return {
            "status": "pending",
            "recovery_manifest": str(recovery_path),
            "submission_ledger": str(submission_path),
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == HADAMARD_TASK
    assert int(recovery.get("step") or -1) == 5
    assert int(recovery.get("timed_out_eval_job") or 0) == 2819085
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    assert int(recovery.get("usable_before_timeout") or 0) == 7
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    eval_dir = state_dir / "eval_ttts7k8_u5"
    partials = sorted(eval_dir.glob("eval_manifest.partial_l8_u7.json*"))
    assert len(partials) == 1, (
        "expected one preserved Hadamard 8-launch/7-usable manifest, "
        f"got {partials}"
    )
    partial = json.loads(partials[0].read_text())
    assert partial.get("partial") is True
    assert partial.get("task") == HADAMARD_TASK
    assert int(partial.get("target") or 0) == 8
    assert int(partial.get("launched") or 0) == 8
    assert int(partial.get("usable") or 0) == 7
    assert int(partial.get("worker_rc") or 0) == 124
    checkpoint = str(partial.get("checkpoint") or "")
    fixed_harness_sha256 = str(partial.get("fixed_harness_sha256") or "")
    assert checkpoint.endswith("/ttts7k8_hadamard_u5")
    assert fixed_harness_sha256

    wrapper_job: int | None = None
    if submission_path.is_file():
        match = re.search(
            r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
        )
        assert match is not None, "Hadamard recovery submission ledger lacks JOB"
        wrapper_job = int(match.group(1))
    elif require_complete:
        raise AssertionError("missing Hadamard recovery submission ledger")

    driver_path = state_dir / "driver.log"
    driver_text = driver_path.read_text(errors="ignore") if driver_path.is_file() else ""
    topup_jobs = [
        int(job) for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    ]
    final_path = eval_dir / "eval_manifest.json"
    final: dict[str, Any] | None = None
    prepared_path = state_dir / "prepare_step05.json"
    prepared: dict[str, Any] | None = None
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        assert final.get("partial") is False
        assert final.get("task") == HADAMARD_TASK
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 9, (
            "Hadamard final u5 manifest erased a timeout/top-up launch"
        )
        assert str(final.get("checkpoint") or "") == checkpoint
        assert str(final.get("fixed_harness_sha256") or "") == fixed_harness_sha256
        assert topup_jobs, "Hadamard final u5 manifest has no top-up job provenance"
        assert 2821177 in topup_jobs, (
            "Hadamard final u5 manifest is not tied to the recorded k8 top-up"
        )
    if prepared_path.is_file():
        prepared = json.loads(prepared_path.read_text())
        assert int(prepared.get("step") or -1) == 5
        assert prepared.get("task") == HADAMARD_TASK
        if final is not None:
            assert int(prepared.get("launched") or 0) == int(final["launched"])
            assert int(prepared.get("usable") or 0) == int(final["usable"])
            assert str(prepared.get("checkpoint") or "") == checkpoint

    if require_complete:
        assert final is not None, "Hadamard executor top-up is incomplete"
        assert prepared is not None, "Hadamard executor update5 was not prepared"

    return {
        "status": (
            "complete_and_prepared" if final is not None and prepared is not None else
            "topup_collected_awaiting_prepare" if final is not None else
            "topup_in_progress"
        ),
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "timed_out_eval_job": 2819085,
        "preserved_partial_manifest": str(partials[0]),
        "original_launched": 8,
        "original_usable": 7,
        "topup_jobs": topup_jobs,
        "final_manifest": str(final_path) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all eight timeout-job launches and every uniquely indexed top-up "
            "are charged on the executor curve; only eight usable rows enter replay"
        ),
    }


def audit_hadamard_executor_u6_timeout_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Fail closed if Hadamard update6 needs the same partial-batch repair."""
    state_dir = RUN_ROOT / "ttt_discover_sota7_extra_k8" / "hadamard"
    eval_dir = state_dir / "eval_ttts7k8_u6"
    final_path = eval_dir / "eval_manifest.json"
    prepared_path = state_dir / "prepare_step06.json"
    timeout_job = 2822098
    timeout_log = SLURM_LOG / f"ttt12-eval-{timeout_job}.err"
    timed_out = (
        timeout_log.is_file() and
        "DUE TO TIME LIMIT" in timeout_log.read_text(errors="ignore")
    )
    if not timed_out:
        final = json.loads(final_path.read_text()) if final_path.is_file() else None
        prepared = (
            json.loads(prepared_path.read_text())
            if prepared_path.is_file() else None
        )
        if final is not None:
            assert final.get("partial") is False
            assert final.get("task") == HADAMARD_TASK
            assert int(final.get("target") or 0) == 8
            assert int(final.get("launched") or 0) == 8
            assert int(final.get("usable") or 0) >= 8
        if prepared is not None:
            assert int(prepared.get("step") or -1) == 6
            assert prepared.get("task") == HADAMARD_TASK
        if require_complete:
            assert final is not None and prepared is not None, (
                "Hadamard update6 is neither complete nor timeout-recovered"
            )
        return {
            "status": (
                "not_needed_eval_completed" if final is not None and
                prepared is not None else "eval_in_progress"
            ),
            "timed_out_eval_job": None,
            "eval_job": timeout_job,
            "final_manifest": str(final_path) if final is not None else None,
        }

    recovery_path = state_dir / "timeout_u6_recovery_manifest.json"
    submission_path = REPO / "results/hadamard_u6_timeout_recovery_submission.env"
    if not recovery_path.is_file():
        assert not require_complete, (
            "missing Hadamard update6 timeout-recovery manifest"
        )
        return {
            "status": "timeout_observed_recovery_pending",
            "recovery_manifest": str(recovery_path),
            "submission_ledger": str(submission_path),
            "timed_out_eval_job": timeout_job,
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == HADAMARD_TASK
    assert int(recovery.get("step") or -1) == 6
    assert int(recovery.get("timed_out_eval_job") or 0) == timeout_job
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    assert int(recovery.get("usable_before_timeout") or 0) == 7
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    partials = sorted(eval_dir.glob("eval_manifest.partial_l8_u7.json*"))
    canonical_payload = (
        json.loads(final_path.read_text()) if final_path.is_file() else None
    )
    if partials:
        assert len(partials) == 1, (
            f"expected one preserved Hadamard u6 partial, got {partials}"
        )
        partial_path = partials[0]
        partial = json.loads(partial_path.read_text())
    else:
        assert canonical_payload is not None and canonical_payload.get(
            "partial"
        ) is True, "Hadamard u6 partial is neither staged nor archived"
        partial_path = final_path
        partial = canonical_payload
    assert partial.get("partial") is True
    assert partial.get("task") == HADAMARD_TASK
    assert int(partial.get("target") or 0) == 8
    assert int(partial.get("launched") or 0) == 8
    assert int(partial.get("usable") or 0) == 7
    assert int(partial.get("worker_rc") or 0) == 124
    checkpoint = str(partial.get("checkpoint") or "")
    harness_hash = str(partial.get("fixed_harness_sha256") or "")
    assert checkpoint.endswith("/ttts7k8_hadamard_u6")
    assert harness_hash

    wrapper_job: int | None = None
    if submission_path.is_file():
        match = re.search(
            r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
        )
        assert match is not None, "Hadamard u6 recovery ledger lacks JOB"
        wrapper_job = int(match.group(1))
    elif require_complete:
        raise AssertionError("missing Hadamard u6 recovery submission ledger")

    wrapper_eval_jobs: set[int] = set()
    if wrapper_job is not None:
        marker = f"/tmp/yingzim/{wrapper_job}"
        for path in SLURM_LOG.glob("ttt12-eval-*.err"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if marker not in text:
                continue
            match = re.search(r"ttt12-eval-([0-9]+)\.err$", path.name)
            if match:
                wrapper_eval_jobs.add(int(match.group(1)))
    driver_text = (
        (state_dir / "driver.log").read_text(errors="ignore")
        if (state_dir / "driver.log").is_file() else ""
    )
    logged_topups = {
        int(job)
        for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    }
    topup_jobs = sorted(wrapper_eval_jobs.intersection(logged_topups))

    final: dict[str, Any] | None = None
    if canonical_payload is not None and canonical_payload.get("partial") is False:
        final = canonical_payload
        assert partials, "Hadamard u6 final replaced an unarchived partial"
        assert final.get("task") == HADAMARD_TASK
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 9
        assert str(final.get("checkpoint") or "") == checkpoint
        assert str(final.get("fixed_harness_sha256") or "") == harness_hash
        assert topup_jobs, "Hadamard u6 final has no wrapper-linked top-up"
    prepared = (
        json.loads(prepared_path.read_text())
        if prepared_path.is_file() else None
    )
    if prepared is not None:
        assert int(prepared.get("step") or -1) == 6
        assert prepared.get("task") == HADAMARD_TASK
        if final is not None:
            assert int(prepared.get("launched") or 0) == int(final["launched"])
            assert int(prepared.get("usable") or 0) == int(final["usable"])
            assert str(prepared.get("checkpoint") or "") == checkpoint
    if require_complete:
        assert final is not None, "Hadamard update6 top-up is incomplete"
        assert prepared is not None, "Hadamard update6 was not prepared"

    return {
        "status": (
            "complete_and_prepared" if final is not None and
            prepared is not None else
            "topup_collected_awaiting_prepare" if final is not None else
            "topup_in_progress"
        ),
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "timed_out_eval_job": timeout_job,
        "preserved_partial_manifest": str(partial_path),
        "original_launched": 8,
        "original_usable": 7,
        "topup_jobs": topup_jobs,
        "final_manifest": str(final_path) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all eight original launches and every unique top-up are charged; "
            "exactly eight usable rows enter replay"
        ),
    }


def audit_hadamard_executor_u7_timeout_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Audit the conditional continuation behind final Hadamard update7.

    The continuation was installed before job 2823649 terminated, so its
    observed usable count is intentionally not hard-coded.  If the job reaches
    K=8, the wrapper is a no-op.  If it times out, the wrapper must bind the
    exact terminal count, preserve all eight original launches, reuse the u7
    checkpoint and fixed harness, and add only fresh trajectory indices.
    """
    state_dir = RUN_ROOT / "ttt_discover_sota7_extra_k8" / "hadamard"
    eval_dir = state_dir / "eval_ttts7k8_u7"
    final_path = eval_dir / "eval_manifest.json"
    prepared_path = state_dir / "prepare_step07.json"
    timeout_job = 2823649
    timeout_log = SLURM_LOG / f"ttt12-eval-{timeout_job}.err"
    timed_out = (
        timeout_log.is_file() and
        "DUE TO TIME LIMIT" in timeout_log.read_text(errors="ignore")
    )
    canonical_payload = (
        json.loads(final_path.read_text()) if final_path.is_file() else None
    )
    prepared = (
        json.loads(prepared_path.read_text())
        if prepared_path.is_file() else None
    )

    if not timed_out:
        final = (
            canonical_payload
            if canonical_payload is not None and
            canonical_payload.get("partial") is False else None
        )
        if final is not None:
            assert final.get("task") == HADAMARD_TASK
            assert int(final.get("target") or 0) == 8
            assert int(final.get("launched") or 0) == 8
            assert int(final.get("usable") or 0) >= 8
            assert str(final.get("checkpoint") or "").endswith(
                "/ttts7k8_hadamard_u7"
            )
        if prepared is not None:
            assert int(prepared.get("step") or -1) == 7
            assert prepared.get("task") == HADAMARD_TASK
        if require_complete:
            assert final is not None and prepared is not None, (
                "Hadamard update7 is neither complete nor timeout-recovered"
            )
        return {
            "status": (
                "not_needed_eval_completed" if final is not None and
                prepared is not None else "eval_in_progress"
            ),
            "timed_out_eval_job": None,
            "eval_job": timeout_job,
            "final_manifest": str(final_path) if final is not None else None,
        }

    recovery_path = state_dir / "timeout_u7_recovery_manifest.json"
    submission_path = REPO / "results/hadamard_u7_timeout_recovery_submission.env"

    # Slurm can mark the batch TIMEOUT while the worker's TERM cleanup is still
    # atomically collecting its last trajectory.  In that case the canonical
    # 8/8 manifest and prepare_step07.json are stronger completion evidence
    # than the batch state.  The conditional wrapper must observe those files
    # and exit without launching a duplicate top-up; the GPU job is still
    # charged in full.
    terminal_complete = (
        canonical_payload is not None and
        canonical_payload.get("partial") is False and
        int(canonical_payload.get("target") or 0) == 8 and
        int(canonical_payload.get("launched") or 0) == 8 and
        int(canonical_payload.get("usable") or 0) >= 8
    )
    terminal_prepared = (
        prepared is not None and
        int(prepared.get("step") or -1) == 7 and
        prepared.get("task") == HADAMARD_TASK and
        int(prepared.get("launched") or 0) == 8 and
        int(prepared.get("usable") or 0) >= 8
    )
    if terminal_complete and terminal_prepared:
        assert canonical_payload.get("task") == HADAMARD_TASK
        checkpoint = str(canonical_payload.get("checkpoint") or "")
        assert checkpoint.endswith("/ttts7k8_hadamard_u7")
        assert str(canonical_payload.get("fixed_harness_sha256") or "")
        assert str(prepared.get("checkpoint") or "") == checkpoint

        wrapper_job: int | None = None
        if submission_path.is_file():
            match = re.search(
                r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
            )
            assert match is not None, "Hadamard u7 recovery ledger lacks JOB"
            wrapper_job = int(match.group(1))
            wrapper_log = (
                SLURM_LOG / f"sah-ttt-had-recover-{wrapper_job}.out"
            )
            assert wrapper_log.is_file(), (
                "missing Hadamard u7 conditional-wrapper stdout"
            )
            assert "already prepared; no recovery needed" in (
                wrapper_log.read_text(errors="ignore")
            ), "Hadamard u7 timeout wrapper did not prove its no-op path"
        elif require_complete:
            raise AssertionError("missing Hadamard u7 recovery submission ledger")

        return {
            "status": "timeout_cleanup_completed_and_prepared_no_topup",
            "submission_ledger": str(submission_path),
            "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
            "timed_out_eval_job": timeout_job,
            "final_manifest": str(final_path),
            "final_launched": int(canonical_payload["launched"]),
            "final_usable": int(canonical_payload["usable"]),
            "prepared_manifest": str(prepared_path),
            "topup_jobs": [],
            "logical_cost_treatment": (
                "the timed-out GPU job is charged in full; its atomic TERM "
                "cleanup produced all eight usable rows, so the conditional "
                "CPU wrapper correctly launched no duplicate trajectory"
            ),
        }

    if not recovery_path.is_file():
        assert not require_complete, (
            "missing Hadamard update7 timeout-recovery manifest"
        )
        return {
            "status": "timeout_observed_recovery_pending",
            "recovery_manifest": str(recovery_path),
            "submission_ledger": str(submission_path),
            "timed_out_eval_job": timeout_job,
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == HADAMARD_TASK
    assert int(recovery.get("step") or -1) == 7
    assert int(recovery.get("timed_out_eval_job") or 0) == timeout_job
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    original_usable = int(recovery.get("usable_before_timeout") or 0)
    assert 1 <= original_usable <= 8
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    partials = sorted(eval_dir.glob(
        f"eval_manifest.partial_l8_u{original_usable}.json*"
    ))
    partial: dict[str, Any] | None = None
    partial_path: Path | None = None
    if original_usable < 8:
        if partials:
            assert len(partials) == 1, (
                f"expected one preserved Hadamard u7 partial, got {partials}"
            )
            partial_path = partials[0]
            partial = json.loads(partial_path.read_text())
        else:
            assert canonical_payload is not None and canonical_payload.get(
                "partial"
            ) is True, "Hadamard u7 partial is neither staged nor archived"
            partial_path = final_path
            partial = canonical_payload
        assert partial.get("task") == HADAMARD_TASK
        assert int(partial.get("target") or 0) == 8
        assert int(partial.get("launched") or 0) == 8
        assert int(partial.get("usable") or 0) == original_usable
        assert int(partial.get("worker_rc") or 0) == 124

    source_payload = partial or canonical_payload
    assert source_payload is not None
    checkpoint = str(source_payload.get("checkpoint") or "")
    harness_hash = str(source_payload.get("fixed_harness_sha256") or "")
    assert checkpoint.endswith("/ttts7k8_hadamard_u7")
    assert harness_hash

    wrapper_job: int | None = None
    if submission_path.is_file():
        match = re.search(
            r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
        )
        assert match is not None, "Hadamard u7 recovery ledger lacks JOB"
        wrapper_job = int(match.group(1))
    elif require_complete:
        raise AssertionError("missing Hadamard u7 recovery submission ledger")

    wrapper_eval_jobs: set[int] = set()
    if wrapper_job is not None:
        marker = f"/tmp/yingzim/{wrapper_job}"
        for path in SLURM_LOG.glob("ttt12-eval-*.err"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if marker not in text:
                continue
            match = re.search(r"ttt12-eval-([0-9]+)\.err$", path.name)
            if match:
                wrapper_eval_jobs.add(int(match.group(1)))
    driver_text = (
        (state_dir / "driver.log").read_text(errors="ignore")
        if (state_dir / "driver.log").is_file() else ""
    )
    logged_topups = {
        int(job)
        for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    }
    topup_jobs = sorted(wrapper_eval_jobs.intersection(logged_topups))

    canonical_payload = (
        json.loads(final_path.read_text()) if final_path.is_file() else None
    )
    final = (
        canonical_payload
        if canonical_payload is not None and
        canonical_payload.get("partial") is False else None
    )
    if final is not None:
        assert final.get("task") == HADAMARD_TASK
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 16 - original_usable
        assert str(final.get("checkpoint") or "") == checkpoint
        assert str(final.get("fixed_harness_sha256") or "") == harness_hash
        if original_usable < 8:
            assert partials, "Hadamard u7 final replaced an unarchived partial"
            assert topup_jobs, "Hadamard u7 final has no wrapper-linked top-up"
    prepared = (
        json.loads(prepared_path.read_text())
        if prepared_path.is_file() else None
    )
    if prepared is not None:
        assert int(prepared.get("step") or -1) == 7
        assert prepared.get("task") == HADAMARD_TASK
        if final is not None:
            assert int(prepared.get("launched") or 0) == int(final["launched"])
            assert int(prepared.get("usable") or 0) == int(final["usable"])
            assert str(prepared.get("checkpoint") or "") == checkpoint
    if require_complete:
        assert final is not None, "Hadamard update7 top-up is incomplete"
        assert prepared is not None, "Hadamard update7 was not prepared"

    return {
        "status": (
            "complete_and_prepared" if final is not None and
            prepared is not None else
            "topup_collected_awaiting_prepare" if final is not None else
            "topup_in_progress"
        ),
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "timed_out_eval_job": timeout_job,
        "preserved_partial_manifest": (
            str(partial_path) if partial_path is not None else None
        ),
        "original_launched": 8,
        "original_usable": original_usable,
        "topup_jobs": topup_jobs,
        "final_manifest": str(final_path) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all eight original launches and every unique top-up are charged; "
            "exactly eight usable rows determine the observed update7 endpoint"
        ),
    }


def audit_hadamard_executor_u8_timeout_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Audit the conditional continuation behind short-lease Hadamard u8.

    Update 8 is the final evaluation that was already submitted before the
    Hadamard lease correction took effect.  The CPU wrapper is installed
    before its outcome is known: it must no-op after a complete K=8 batch, or
    bind the terminal partial count and top up only fresh indices under the
    exact u8 checkpoint, parent, and fixed harness.
    """
    step, eval_job = 8, 2825671
    state_dir = RUN_ROOT / "ttt_discover_sota7_extra_k8" / "hadamard"
    eval_dir = state_dir / "eval_ttts7k8_u8"
    final_path = eval_dir / "eval_manifest.json"
    prepared_path = state_dir / "prepare_step08.json"
    recovery_path = state_dir / "timeout_u8_recovery_manifest.json"
    submission_path = REPO / "results/hadamard_u8_timeout_recovery_submission.env"
    timeout_log = SLURM_LOG / f"ttt12-eval-{eval_job}.err"
    timed_out = (
        timeout_log.is_file()
        and "DUE TO TIME LIMIT" in timeout_log.read_text(errors="ignore")
    )
    canonical = (
        json.loads(final_path.read_text()) if final_path.is_file() else None
    )
    prepared = (
        json.loads(prepared_path.read_text())
        if prepared_path.is_file() else None
    )

    wrapper_job: int | None = None
    if submission_path.is_file():
        text = submission_path.read_text()
        match = re.search(r"^JOB=([0-9]+)$", text, re.MULTILINE)
        assert match is not None, "Hadamard u8 recovery ledger lacks JOB"
        wrapper_job = int(match.group(1))
        assert re.search(
            rf"^DEPENDENCY=afterany:{eval_job}$", text, re.MULTILINE
        ), "Hadamard u8 wrapper is not bound afterany to the canonical eval"
        assert re.search(r"^RECOVERY_STEP=8$", text, re.MULTILINE)
    elif require_complete:
        raise AssertionError("missing Hadamard u8 recovery submission ledger")

    final = (
        canonical
        if canonical is not None and canonical.get("partial") is False
        else None
    )
    if final is not None:
        assert final.get("task") == HADAMARD_TASK
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert str(final.get("checkpoint") or "").endswith(
            "/ttts7k8_hadamard_u8"
        )
        assert str(final.get("fixed_harness_sha256") or "")
    if prepared is not None:
        assert int(prepared.get("step") or -1) == step
        assert prepared.get("task") == HADAMARD_TASK
        if final is not None:
            assert int(prepared.get("launched") or 0) == int(final["launched"])
            assert int(prepared.get("usable") or 0) == int(final["usable"])
            assert str(prepared.get("checkpoint") or "") == str(
                final.get("checkpoint") or ""
            )

    if not timed_out:
        if require_complete:
            assert final is not None and prepared is not None, (
                "Hadamard update8 is neither complete nor timeout-recovered"
            )
        return {
            "status": (
                "not_needed_eval_completed"
                if final is not None and prepared is not None
                else "eval_in_progress"
            ),
            "eval_job": eval_job,
            "timed_out_eval_job": None,
            "submission_ledger": str(submission_path),
            "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
            "final_manifest": str(final_path) if final is not None else None,
        }

    # A timeout can still finish all eight rows during TERM cleanup, as u7 did.
    if (
        final is not None
        and prepared is not None
        and int(final.get("launched") or 0) == 8
    ):
        if wrapper_job is not None:
            wrapper_log = SLURM_LOG / f"sah-ttt-had-recover-{wrapper_job}.out"
            if require_complete:
                assert wrapper_log.is_file()
                assert "already prepared; no recovery needed" in (
                    wrapper_log.read_text(errors="ignore")
                )
        return {
            "status": "timeout_cleanup_completed_and_prepared_no_topup",
            "eval_job": eval_job,
            "timed_out_eval_job": eval_job,
            "submission_ledger": str(submission_path),
            "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
            "final_manifest": str(final_path),
            "final_launched": 8,
            "final_usable": int(final["usable"]),
            "topup_jobs": [],
            "logical_cost_treatment": (
                "the timed-out GPU allocation is fully charged; TERM cleanup "
                "completed K=8 and the CPU wrapper launched no duplicate"
            ),
        }

    if not recovery_path.is_file():
        assert not require_complete, "Hadamard u8 timeout recovery is pending"
        return {
            "status": "timeout_observed_recovery_pending",
            "eval_job": eval_job,
            "timed_out_eval_job": eval_job,
            "submission_ledger": str(submission_path),
            "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
            "recovery_manifest": str(recovery_path),
        }

    recovery = json.loads(recovery_path.read_text())
    assert recovery.get("task") == HADAMARD_TASK
    assert int(recovery.get("step") or -1) == step
    assert int(recovery.get("timed_out_eval_job") or 0) == eval_job
    assert int(recovery.get("launched_before_timeout") or 0) == 8
    original_usable = int(recovery.get("usable_before_timeout") or 0)
    assert 1 <= original_usable <= 8
    protocol = recovery.get("protocol_unchanged") or {}
    assert int(protocol.get("K_usable") or 0) == 8
    assert int(protocol.get("max_evals_per_trajectory") or 0) == 20
    assert protocol.get("fixed_harness") is True
    assert protocol.get("checkpoint_reused") is True
    assert protocol.get("optimizer_cadence_reused") is True

    partials = sorted(eval_dir.glob(
        f"eval_manifest.partial_l8_u{original_usable}.json*"
    ))
    partial: dict[str, Any] | None = None
    partial_path: Path | None = None
    if original_usable < 8:
        if partials:
            assert len(partials) == 1
            partial_path = partials[0]
            partial = json.loads(partial_path.read_text())
        else:
            assert canonical is not None and canonical.get("partial") is True
            partial_path, partial = final_path, canonical
        assert partial.get("task") == HADAMARD_TASK
        assert int(partial.get("target") or 0) == 8
        assert int(partial.get("launched") or 0) == 8
        assert int(partial.get("usable") or 0) == original_usable
        assert int(partial.get("worker_rc") or 0) == 124

    source = partial or canonical
    assert source is not None
    checkpoint = str(source.get("checkpoint") or "")
    harness_hash = str(source.get("fixed_harness_sha256") or "")
    assert checkpoint.endswith("/ttts7k8_hadamard_u8")
    assert harness_hash

    wrapper_eval_jobs: set[int] = set()
    if wrapper_job is not None:
        marker = f"/tmp/yingzim/{wrapper_job}"
        for path in SLURM_LOG.glob("ttt12-eval-*.err"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if marker not in text:
                continue
            match = re.search(r"ttt12-eval-([0-9]+)\.err$", path.name)
            if match:
                wrapper_eval_jobs.add(int(match.group(1)))
    driver_text = (
        (state_dir / "driver.log").read_text(errors="ignore")
        if (state_dir / "driver.log").is_file() else ""
    )
    logged_topups = {
        int(job)
        for job in re.findall(r"eval top-up job=([0-9]+)", driver_text)
    }
    topup_jobs = sorted(wrapper_eval_jobs.intersection(logged_topups))

    canonical = (
        json.loads(final_path.read_text()) if final_path.is_file() else None
    )
    final = (
        canonical
        if canonical is not None and canonical.get("partial") is False
        else None
    )
    if final is not None:
        assert final.get("task") == HADAMARD_TASK
        assert int(final.get("target") or 0) == 8
        assert int(final.get("usable") or 0) >= 8
        assert int(final.get("launched") or 0) >= 16 - original_usable
        assert str(final.get("checkpoint") or "") == checkpoint
        assert str(final.get("fixed_harness_sha256") or "") == harness_hash
        if original_usable < 8:
            assert partials, "Hadamard u8 final replaced an unarchived partial"
            assert topup_jobs, "Hadamard u8 final has no wrapper-linked top-up"
    prepared = (
        json.loads(prepared_path.read_text())
        if prepared_path.is_file() else None
    )
    if prepared is not None:
        assert int(prepared.get("step") or -1) == step
        assert prepared.get("task") == HADAMARD_TASK
        if final is not None:
            assert int(prepared.get("launched") or 0) == int(final["launched"])
            assert int(prepared.get("usable") or 0) == int(final["usable"])
            assert str(prepared.get("checkpoint") or "") == checkpoint
    if require_complete:
        assert final is not None, "Hadamard update8 top-up is incomplete"
        assert prepared is not None, "Hadamard update8 was not prepared"

    return {
        "status": (
            "complete_and_prepared"
            if final is not None and prepared is not None
            else "topup_collected_awaiting_prepare"
            if final is not None else "topup_in_progress"
        ),
        "eval_job": eval_job,
        "timed_out_eval_job": eval_job,
        "recovery_manifest": str(recovery_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "preserved_partial_manifest": (
            str(partial_path) if partial_path is not None else None
        ),
        "original_launched": 8,
        "original_usable": original_usable,
        "topup_jobs": topup_jobs,
        "final_manifest": str(final_path) if final is not None else None,
        "final_launched": int(final["launched"]) if final is not None else None,
        "final_usable": int(final["usable"]) if final is not None else None,
        "logical_cost_treatment": (
            "all eight original launches and every fresh-index top-up are "
            "charged; exactly eight usable rows determine the u8 replay"
        ),
    }


def audit_hadamard_executor_eval_lease_policy() -> dict[str, Any]:
    """Separate scheduler-lease changes from the logical executor protocol."""
    ledger_path = REPO / "results/hadamard_executor_eval_lease_policy_correction.env"
    assert ledger_path.is_file(), "missing Hadamard eval-lease policy ledger"
    fields: dict[str, str] = {}
    for line in ledger_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    assert fields.get("METHOD") == (
        "scaled_TTT_Discover_style_executor_update_reference"
    )
    assert fields.get("TASK") == HADAMARD_TASK
    assert fields.get("LOGICAL_PROTOCOL_CHANGED") == "no"
    assert fields.get("K_USABLE") == "8"
    assert fields.get("MAX_EVALS_PER_TRAJECTORY") == "20"
    assert fields.get("CHECKPOINT_HARNESS_OPTIMIZER_CHANGED") == "no"
    assert fields.get("NEW_SUBMISSION_POLICY_QOS") == "normal"
    assert fields.get("NEW_SUBMISSION_POLICY_TIME") == "04:00:00"
    assert fields.get("ACTUAL_U8_JOB") == "2825671"
    assert fields.get("ACTUAL_U8_QOS") == "short"
    assert fields.get("ACTUAL_U8_TIME") == "02:00:00"
    assert fields.get("ACTUAL_U8_STATE_MUTATED") == "no"

    driver_path = REPO / "scripts/drive_ttt_executor_12h.sh"
    driver_text = driver_path.read_text()
    policy_clause = '"$task" == "eft__math__hadamard_maximal_det"'
    assert driver_text.count(policy_clause) >= 2, (
        "Hadamard is missing from initial-eval or top-up normal-QoS policy"
    )
    u8_jobs = (
        RUN_ROOT / "ttt_discover_sota7_extra_k8" / "hadamard" /
        "jobs_ttts7k8_u8.env"
    )
    assert u8_jobs.is_file(), "missing Hadamard u8 immutable job ledger"
    assert re.search(r"^EVAL_JOB=2825671$", u8_jobs.read_text(), re.MULTILINE)

    return {
        "status": "future_submissions_normal_4h_u8_as_run_short_2h",
        "ledger": str(ledger_path),
        "driver": str(driver_path),
        "actual_u8_job": 2825671,
        "logical_protocol_changed": False,
        "cost_treatment": fields["COST_TREATMENT"],
        "active_segment_caveat": fields["ACTIVE_SEGMENT_CAVEAT"],
    }


def audit_rewardfix_context_round_namespace_correction(
    *, require_complete: bool
) -> dict[str, Any]:
    """Bind the reward-fix context plateau controller to rounds 1109--1118."""
    correction_path = (
        REPO / "results/rewardfix_context_controller_round_namespace_correction.env"
    )
    controller_path = (
        REPO / "results/commit_gated_plateau_controller_submissions.env"
    )
    assert correction_path.is_file(), "missing reward-fix controller correction"
    assert controller_path.is_file(), "missing commit-gated controller ledger"

    def fields(path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
        return result

    correction = fields(correction_path)
    controllers = fields(controller_path)
    assert correction.get("RETIRED_JOB") == "2824271"
    assert correction.get("GPU_WORK_AFFECTED") == "no"
    assert correction.get("ACTIVE_OUTER_AT_CORRECTION") == "2824519"
    assert correction.get("REPLACEMENT_JOB") == "2825248"
    assert correction.get("START_REVIEW_ROUND") == "1109"
    assert correction.get("MAX_ROUND") == "1118"
    assert controllers.get("REWARDFIX_CONTEXT_JOB") == "2825248"
    assert controllers.get("REWARDFIX_CONTEXT_RETIRED_JOB") == "2824271"

    workspace = RUN_ROOT / "context_sota7_rewardfix_v1"
    review_path = workspace / "plateau_review.json"
    complete_path = workspace / "CANONICAL_CONTEXT_COMPLETE"
    review = json.loads(review_path.read_text()) if review_path.is_file() else None
    if review is not None:
        assert review.get("method") == (
            "context/analyzer; proposer and executor weights frozen"
        )
        assert int(review.get("max_round") or 0) == 1118
        completed_round = int(review.get("completed_round") or 0)
        assert 1109 <= completed_round <= 1118
        rounds = [int(row.get("round") or 0) for row in review.get("rows") or []]
        assert rounds and min(rounds) >= 1100 and max(rounds) == completed_round
    if require_complete:
        assert review is not None, "reward-fix context plateau review is incomplete"
        assert complete_path.is_file(), (
            "reward-fix context completion marker is absent"
        )
    return {
        "status": (
            "complete" if review is not None and complete_path.is_file()
            else "corrected_controller_waiting_for_round1109"
        ),
        "correction_ledger": str(correction_path),
        "retired_cpu_controller": 2824271,
        "replacement_cpu_controller": 2825248,
        "active_gpu_outer_untouched": 2824519,
        "start_review_round": 1109,
        "max_round": 1118,
        "review": str(review_path) if review is not None else None,
        "cpu_controller_cost_treatment": (
            "both one-core waiters are orchestration only and excluded from "
            "GPU-hour accounting; every outer GPU allocation remains charged"
        ),
    }


def audit_sys_context_controller_recovery(
    *, require_complete: bool
) -> dict[str, Any]:
    """Verify round011 survived a CPU-controller script-read failure once."""
    workspace = RUN_ROOT / "context_sota5_sys_guarded"
    outer = RUN_ROOT / "outer-context-sota5-sys-guarded"
    manifest_path = workspace / "round011_controller_recovery.json"
    submission_path = REPO / "results/sys_context_recovery_submission.env"
    if not manifest_path.is_file():
        assert not require_complete, (
            "missing system-context round011 controller-recovery manifest"
        )
        return {
            "status": "round011_or_recovery_pending",
            "recovery_manifest": str(manifest_path),
            "submission_ledger": str(submission_path),
        }

    payload = json.loads(manifest_path.read_text())
    assert payload.get("method") == (
        "context/analyzer; proposer and executor weights frozen"
    )
    assert int(payload.get("failed_cpu_controller_job") or 0) == 2812925
    assert int(payload.get("already_launched_canonical_outer_job") or 0) == 2823686
    assert int(payload.get("recovered_round") or -1) == 11
    assert str(payload.get("treatment") or "").startswith(
        "retain canonical round011"
    )
    summary = Path(str(payload.get("round_summary") or ""))
    assert summary == outer / "round011" / "round_summary.json"
    assert summary.is_file()
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == payload.get(
        "round_summary_sha256"
    )

    driver = workspace / "driver.log"
    driver_text = driver.read_text(errors="ignore")
    assert "syntax error near unexpected token" in driver_text
    # The recovery deliberately post-processes the already launched round011;
    # a second round011 GPU job would make the logical/as-run treatment stale.
    round11_jobs = re.findall(
        r"step\s+1/3:\s+round11\b[\s\S]{0,300}?\bjob\s+([0-9]+)",
        driver_text,
    )
    assert round11_jobs == ["2823686"], (
        f"unexpected system-context round011 job chain: {round11_jobs}"
    )

    wrapper_job: int | None = None
    if submission_path.is_file():
        match = re.search(
            r"^JOB=([0-9]+)$", submission_path.read_text(), re.MULTILINE
        )
        assert match is not None, "system-context recovery ledger lacks JOB"
        wrapper_job = int(match.group(1))
    elif require_complete:
        raise AssertionError("missing system-context recovery submission ledger")

    return {
        "status": "round011_bookkeeping_recovered_without_duplicate_launch",
        "recovery_manifest": str(manifest_path),
        "submission_ledger": str(submission_path),
        "cpu_wrapper_job_excluded_from_gpu_hours": wrapper_job,
        "failed_cpu_controller_job_excluded_from_gpu_hours": 2812925,
        "canonical_round011_outer_job": 2823686,
        "round011_summary_sha256": payload["round_summary_sha256"],
        "logical_cost_treatment": (
            "round011 remains one ordinary context batch; both controller "
            "jobs are CPU-only and excluded from adaptation GPU-hours"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="papers/figures/score_compute_curves_sota7_data.json")
    parser.add_argument(
        "--out", default="results/score_compute_curves_sota7_audit.json")
    parser.add_argument(
        "--endpoint-validation",
        default=str(RUN_ROOT / "sota7_endpoint_validation/results.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    source = Path(args.data).resolve()
    payload = json.loads(source.read_text())
    assert tuple(payload["tasks"]) == TASKS, \
        f"unexpected task order: {tuple(payload['tasks'])}"
    clean_core_route_active = {
        AHC039_TASK: Path(str(payload["tasks"][AHC039_TASK].get(
            "proposer_workspace") or "")).name ==
        "proposer_sota5_ahc039_clean_v1",
        EPLB_TASK: Path(str(payload["tasks"][EPLB_TASK].get(
            "proposer_workspace") or "")).name ==
        "proposer_sota5_eplb_clean_v1",
    }
    assert "divided by task's Best Human value" in payload["y"], payload["y"]
    reference_standard = payload.get("reference_standard") or {}
    assert reference_standard.get("name") == "Best Human"
    assert reference_standard.get("y_equals_one") is True
    assert reference_standard.get("normalization") == (
        "combined_score / human_best_combined_score"
    )
    assert Path(str(reference_standard.get("source") or "")).resolve() == (
        HUMAN_BEST_REFERENCE.resolve()
    )
    assert reference_standard.get("source_sha256") == hashlib.sha256(
        HUMAN_BEST_REFERENCE.read_bytes()
    ).hexdigest()
    guards = payload.get("protocol_validity_guards") or {}
    assert "topology" in str(guards.get("eplb", "")), \
        "missing EPLB topology validity guard"
    assert "success_rate" in str(guards.get("prism", "")), \
        "missing PRISM complete-success validity guard"
    assert "exact permutation" in str(guards.get("txn", "")), \
        "missing Txn exact-permutation validity guard"
    guard_proof = payload.get("protocol_guard_proof") or {}
    legacy_guard_proof = payload.get("legacy_semantic_guard_proof") or {}
    guard_fields = guard_proof.get("fields") or {}
    if not args.allow_incomplete:
        assert guard_fields.get("eplb_topology_guard") == "ok", \
            "EPLB guard test proof is missing"
        assert guard_fields.get("prism_success_guard") == "ok", \
            "PRISM guard test proof is missing"
        assert guard_fields.get("txn_legality_guard") == "ok", \
            "Txn guard test proof is missing"
        assert guard_proof.get("matches_current_worker"), \
            "guard test did not validate the current _eval_worker.py"
        assert legacy_guard_proof.get("matches_current_worker"), \
            "legacy semantic audit did not use the current _eval_worker.py"
        legacy_counts = legacy_guard_proof.get("counts") or {}
        for guarded_task in (
            "adrs__eplb", "adrs__prism", "adrs__txn_scheduling"
        ):
            counts = legacy_counts.get(guarded_task) or {}
            assert int(counts.get("checked") or 0) > 0, \
                f"legacy semantic audit is missing {guarded_task}"
            assert int(counts.get("checked") or 0) == (
                int(counts.get("valid") or 0) + int(counts.get("invalid") or 0)
            ), f"legacy semantic audit counts do not close for {guarded_task}"
            assert (legacy_guard_proof.get("warm_start_valid") or {}).get(
                guarded_task) is True, \
                f"shared H2 warm-start failed {guarded_task} semantic guard"
        for guarded_task in ("adrs__eplb", "adrs__prism"):
            scope = (legacy_guard_proof.get("scope") or {}).get(
                guarded_task
            ) or {}
            expected_context_root = {
                "adrs__eplb": "outer-context-sota5-sys-guarded",
                "adrs__prism": "outer-context-sota7-rewardfix-v1",
            }[guarded_task]
            assert Path(scope.get("isolated_context_root", "")).name == (
                expected_context_root
            ), f"{guarded_task} semantic replay used the wrong context lineage"
            assert scope.get("isolated_context_rounds_replayed"), \
                f"{guarded_task} context rounds lack current-guard replay"
            assert int(scope.get("isolated_context_records_checked") or 0) > 0
            assert int(scope.get("isolated_context_records_invalid") or 0) == 0, \
                f"a {guarded_task} context row failed semantic replay"
            assert scope.get(
                "isolated_context_all_accepted_rows_replayed"
            ) is True, f"{guarded_task} context replay does not cover all rows"
            executor_scope = scope.get("isolated_executor") or {}
            assert int(executor_scope.get("batches_in_state") or 0) > 0
            assert int(executor_scope.get("records_checked") or 0) > 0
            assert int(executor_scope.get("records_invalid") or 0) == 0, \
                f"a {guarded_task} executor training row failed semantic replay"
            assert executor_scope.get("all_training_rows_replayed") is True, \
                f"{guarded_task} executor replay does not cover all training rows"
            assert executor_scope.get("archive_nonroot_unmatched") == [], \
                f"{guarded_task} executor archive contains unreplayed nodes"
            assert executor_scope.get("archive_nonroot_invalid") == [], \
                f"{guarded_task} executor archive contains invalid nodes"
            assert executor_scope.get("selected_parent_valid") is True, \
                f"{guarded_task} executor selected parent is not replay-valid"
        txn_warm_replay = (legacy_guard_proof.get(
            "warm_start_current_guard_scores"
        ) or {}).get("adrs__txn_scheduling")
        assert txn_warm_replay is not None and float(txn_warm_replay) > 0.0, \
            "Txn H2 warm start lacks a valid current-guard replay score"
    executor_batch_k = int(payload["executor_baseline"][
        "target_usable_trajectories_per_update"])
    official_sources = payload["executor_baseline"].get("official_sources") or {}
    assert official_sources.get("repository_commit_audited") == (
        "6c40e82dab9d5de7416ac873ad5cd3106084aaed"
    ), "TTT-Discover reference implementation is not pinned to the audited commit"
    assert "8 groups x 64" in str(official_sources.get("official_batch", "")), \
        "TTT-Discover official batch provenance is missing"
    if not args.allow_incomplete:
        assert executor_batch_k == 8, (
            f"canonical comparison requires cadence-matched K=8, got K={executor_batch_k}"
        )

    baselines = json.loads(
        (REPO / "results/baseline_h2_20ev.json").read_text())["baseline"]
    warm_provenance = payload.get("shared_h2_warm_start_provenance") or {}
    warm_tasks = warm_provenance.get("tasks") or {}
    assert Path(warm_provenance.get("path", "")).is_file(), \
        "missing H2 warm-start provenance file"
    result: dict[str, Any] = {
        "schema": 1,
        "source": str(source),
        "metric": "combined_score / task-specific Best Human value",
        "sql_proposer_clean_lineage": audit_sql_proposer_lineage(
            payload, require_complete=not args.allow_incomplete),
        "txn_proposer_clean_lineage": audit_txn_proposer_lineage(
            payload, require_complete=not args.allow_incomplete),
        "prism_proposer_clean_lineage": audit_prism_proposer_lineage(
            payload, require_complete=not args.allow_incomplete),
        "ahc039_proposer_clean_lineage": audit_core_clean_proposer_lineage(
            payload, AHC039_TASK, "ahc039",
            require_complete=not args.allow_incomplete,
        ),
        "eplb_proposer_clean_lineage": audit_core_clean_proposer_lineage(
            payload, EPLB_TASK, "eplb",
            require_complete=not args.allow_incomplete,
        ),
        "hadamard_proposer_rewardfix_lineage": (
            audit_rewardfix_proposer_lineage(
                payload, "eft__math__hadamard_maximal_det", "hadamard",
                require_complete=not args.allow_incomplete,
            )
        ),
        "ahc058_proposer_rewardfix_lineage": (
            audit_rewardfix_proposer_lineage(
                payload, "eft__ahc_simpletes__ahc058", "ahc058",
                require_complete=not args.allow_incomplete,
            )
        ),
        "prism_context_rewardfix_lineage": (
            audit_rewardfix_context_lineage(
                payload, require_complete=not args.allow_incomplete
            )
        ),
        "ahc058_context_analysis_required_lineage": (
            audit_ahc058_analysis_required_context_lineage(
                payload, require_complete=not args.allow_incomplete
            )
        ),
        "sys_context_controller_recovery": (
            audit_sys_context_controller_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "rewardfix_context_round_namespace_correction": (
            audit_rewardfix_context_round_namespace_correction(
                require_complete=not args.allow_incomplete
            )
        ),
        "all_plotted_reward_attribution": (
            audit_all_plotted_reward_attribution(
                payload, require_complete=not args.allow_incomplete
            )
        ),
        "reported_condition_alignment": audit_reporting_condition_alignment(
            payload, require_complete=not args.allow_incomplete),
        "human_best_reference_alignment": audit_human_best_reference(payload),
        "paper_reward_route_alignment": audit_paper_reward_route_alignment(
            require_complete=not args.allow_incomplete
        ),
        "endpoint_revalidation": audit_endpoint_validation(
            Path(args.endpoint_validation).resolve(), source,
            require_complete=not args.allow_incomplete,
        ),
        "txn_executor_timeout_recovery": (
            audit_txn_executor_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "txn_executor_u2_timeout_recovery": (
            audit_txn_executor_u2_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "hadamard_executor_timeout_recovery": (
            audit_hadamard_executor_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "hadamard_executor_u6_timeout_recovery": (
            audit_hadamard_executor_u6_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "hadamard_executor_u7_timeout_recovery": (
            audit_hadamard_executor_u7_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "hadamard_executor_u8_timeout_recovery": (
            audit_hadamard_executor_u8_timeout_recovery(
                require_complete=not args.allow_incomplete
            )
        ),
        "hadamard_executor_eval_lease_policy": (
            audit_hadamard_executor_eval_lease_policy()
        ),
        "proposer_prompt_integrity": audit_proposer_prompt_integrity(payload),
        "tasks": {},
        "aggregate": {},
        "common_fixed_warm_start_cost": {
            "scope": "one shared pre-adaptation trajectory per task, not per method",
            "executor_trajectories": len(TASKS),
            "max_evaluator_call_budget": 20 * len(TASKS),
            "recorded_evaluator_calls": sum(
                int(((warm_tasks.get(task) or {}).get("ledger") or {}).get(
                    "evaluator_calls") or 0)
                for task in TASKS
            ),
            "recorded_executor_model_calls": sum(
                int(((warm_tasks.get(task) or {}).get("ledger") or {}).get(
                    "llm_calls") or 0)
                for task in TASKS
            ),
            "recorded_sandbox_seconds": sum(
                float(((warm_tasks.get(task) or {}).get("ledger") or {}).get(
                    "sandbox_seconds") or 0.0)
                for task in TASKS
            ),
            "excluded_from_each_method_incremental_total": True,
            "from_scratch_accounting": (
                "add these identical seven-task costs to each method; rankings "
                "and pairwise budget differences are unchanged"
            ),
        },
        "suite_costs": {
            "proposer": {"executor_trajectories": 0, "harness_proposals": 0,
                         "proposer_model_calls": 0,
                         "reviewer_model_calls_lower_bound": 0,
                         "charged_evaluator_call_budget": 0,
                         "recorded_evaluator_calls_lower_bound": 0,
                         "recorded_executor_model_calls_lower_bound": 0,
                         "recorded_total_model_calls_lower_bound": 0,
                         "recorded_sandbox_seconds_lower_bound": 0.0,
                         "analyzer_calls": 0, "weight_updates": 0,
                         "planned_optimizer_boundaries": 0,
                         "outer_jobs": [], "train_jobs": [], "merge_jobs": [],
                         "unmapped_weight_update_tags": []},
            "context": {"executor_trajectories": 0, "harness_proposals": 0,
                        "proposer_model_calls": 0,
                        "reviewer_model_calls_lower_bound": 0,
                        "charged_evaluator_call_budget": 0,
                        "recorded_evaluator_calls_lower_bound": 0,
                        "recorded_executor_model_calls_lower_bound": 0,
                        "recorded_total_model_calls_lower_bound": 0,
                        "recorded_sandbox_seconds_lower_bound": 0.0,
                        "analyzer_calls": 0, "analyzer_briefs": 0,
                        "weight_updates": 0, "planned_optimizer_boundaries": 0,
                        "outer_jobs": []},
            "executor": {"executor_trajectories": 0, "harness_proposals": 0,
                         "charged_evaluator_call_budget": 0,
                         "recorded_evaluator_calls_lower_bound": 0,
                         "recorded_executor_model_calls_lower_bound": 0,
                         "recorded_total_model_calls_lower_bound": 0,
                         "recorded_sandbox_seconds_lower_bound": 0.0,
                         "analyzer_calls": 0, "weight_updates": 0,
                         "planned_optimizer_boundaries": 0,
                         "eval_jobs": [], "train_jobs": [], "merge_jobs": []},
        },
        "fairness": {
            "matched": [
                "same Qwen3.5-9B base family",
                "same task evaluators and higher-is-better combined_score",
                "same EPLB topology, PRISM full-success, and Txn exact-permutation semantic validity guards",
                "legacy proposer programs are hash-indexed and replayed through the current semantic guards",
                "same fixed-H2 20-evaluator-call warm-start measurement at x=1",
                "same task.initial_program at the start of each fresh arm campaign; later batches use only that arm's task-local ratchet, and the H2-discovered program is not inherited",
                "proposer/context H1 prompts share the incumbent plus bounded per-candidate scored feedback; only context adds two frozen-model analyst calls and only proposer changes proposer weights",
                "every plotted proposer-weight H1 prompt is serialized and audited to contain no curated ANALYST NOTE; the leaked historical SQL lineage is excluded",
                "terminal summary rows are authoritative for reward attribution; an explicit best_score=null cannot inherit the seed checkpoint, and affected historical Hadamard/AHC058 proposer plus AHC058/PRISM context lineages are excluded",
                "the historical PRISM round410--417 proposer lineage is excluded because its selected round410 harness was rewarded by a success_rate=0.98 program; the replacement campaign enforces success_rate=1.0 online",
                "executor PUCT root carries the true initial-program seed score, not the H2 harness score",
                (
                    f"all three routes use nominal K={executor_batch_k} on the "
                    "two formerly mismatched core tasks; every curve x uses "
                    "actual materialized executor launches"
                    if all(clean_core_route_active.values()) else
                    f"context and executor, plus six of seven proposer lineages, "
                    f"use nominal K={executor_batch_k}; the live AHC039 fallback "
                    "retains its recorded K=16 cadence until the clean replacement "
                    "materializes"
                ),
                "invalid/non-materialized H1 proposals are excluded from executor-trajectory x but charged in the proposal ledger and in the recorded-model-call lower bound when assistant turns were serialized",
                "same task evaluator; secondary budget charges launched x per-round max_evals",
                "x counts actual launched executor trajectories, including failures",
                "recorded model/evaluator calls are read from trajectory ledgers; missing crash summaries are labeled lower bounds",
            ],
            "not_matched": [
                "after the shared fixed-H2 anchor, proposer/context synthesize a per-candidate harness through H1 while the TTT-style executor arm keeps the initial harness fixed and has no H1; this is intrinsic to the compared systems but prevents attributing every score gap solely to which weights receive the update",
                "x excludes harness-proposal and analyzer inference",
                "x excludes LoRA-training and checkpoint-merge GPU time",
                "proposer/context outer jobs reserve four GPUs; proposer-weight H1 currently uses one trained-phi replica while frozen context H1 can use up to four replicas, so logical model-call counts are comparable but wall-clock concurrency is not; authoritative allocated-GPU-hours charge all reserved GPUs, including idle capacity",
                "the official TTT-Discover paper calls its 25,600-rollout comparisons sampling-budget matched; it does not establish equal total training compute, so this audit reports weight-update compute separately",
                *([] if clean_core_route_active[AHC039_TASK] else [
                    "the live AHC039 proposer fallback uses K=16/max_evals=30 "
                    "rather than canonical K=8/max_evals=20; it is forbidden "
                    "from the strict final artifact"
                ]),
                "historical proposer campaigns are not independent reruns of the new arms",
                *([] if clean_core_route_active[AHC039_TASK] else [
                    "the live historical AHC039 proposer curve aggregates multiple "
                    "task-local restarts rather than one uninterrupted optimizer "
                    "lineage; all launches remain charged"
                ]),
                "context jobs batch two or three tasks behind one shared model server",
                "AHC/system-context jobs co-batch older AHC058/PRISM branches that are excluded from the primary curves and replaced by rewardfix-v1; suite GPU allocation charges the whole shared jobs, while task-local logical-call metrics include only plotted lineages",
                "five accepted proposer/context outer jobs exited nonzero only after their complete launched trajectory sets and atomic collector outputs were materialized; their score evidence is retained and their complete allocations are charged, with a machine-audited anomaly registry",
                f"local budget-scaled TTT-Discover-style reference uses one group of {executor_batch_k} rather than the official 8x64 per update",
                "TTT replay teacher-forces the final edit/program instead of the full sampled token trajectory",
                "local executor optimizer uses Adam beta2=0.98 and weight decay 0.1 rather than the official Tinker optimizer",
                "local FSDP scheduler starts at zero learning rate; canonical K=8 uses GBS=4 to obtain two optimizer boundaries in one replay epoch",
                "the first local GBS=4 boundary changes Adam moments but not weights; the second boundary makes the nonzero weight update, unlike the official single full-batch optimizer step",
                "both local weight-updating routes carry accumulated LoRA weights forward but reinitialize Adam and the scheduler for every training job; the official Tinker training client keeps optimizer state",
                "proposer LoRA is r64/a128 for 3 epochs; executor LoRA is r32/a64 for 1 epoch",
                "weight-update totals count attempted training jobs, including failed attempts; successful conditioned batches are reported separately",
                "the earlier separate-team AHC039 559,534 endpoint is excluded because it lacks a local program/rollout ledger",
                "historical proposer and newly run context/executor campaigns do not use paired sampling seeds",
                "one campaign per arm; no uncertainty interval",
                "H1 request counts come from serialized assistant turns; transport-level retries, timed-out requests, and server work orphaned after a client timeout are not completely recoverable and therefore make recorded proposer-model calls a lower bound",
                "local-vLLM trajectory ledgers count model requests but do not recover prompt/completion token totals, so model-call budgets are a sensitivity analysis rather than FLOP equivalence",
                "final allocated GPU hours come from a frozen authoritative sacct snapshot "
                "(live logs provide only a provisional proxy); neither is FLOP accounting",
            ],
            "supported_claim": (
                "executor-trajectory and charged-evaluator-budget sample efficiency, "
                "recorded-model-call sensitivity, plus observed in-budget endpoints; an empirical plateau is claimed only when the final three batch transitions show no gain"
            ),
            "unsupported_claim": (
                "equal-total-compute superiority or an absolute asymptotic limit"
            ),
            "batch_cadence": {
                "proposer": (
                    "all seven plotted proposer lineages use K=8/max_evals=20"
                    if all(clean_core_route_active.values()) else
                    "six tasks use K=8/max_evals=20; the allow-incomplete "
                    "AHC039 fallback uses K=16/max_evals=30"
                ),
                "context": "nominal K=8 candidates per outer round",
                "executor": f"K={executor_batch_k} trajectories per weight update",
                "implication": (
                    "nominal adaptation cadence is matched at K=8 for all three "
                    "routes on AHC039 and EPLB"
                    if all(clean_core_route_active.values()) else
                    "the clean AHC039/EPLB proposer replacements are still "
                    "pending; no strong cadence-matched claim is allowed"
                ),
            },
            "warm_start_cost_treatment": (
                "the one identical H2 trajectory is shown in x and reported as a "
                "common fixed cost; per-arm logical adaptation-cost totals exclude it"
            ),
        },
    }

    for task in TASKS:
        task_data = payload["tasks"][task]
        reference = float(payload["anchors"][task][1])
        baseline = float(payload["anchors"][task][0])
        expected_h2 = float(baselines[task]["h2_best"])
        assert abs(baseline - expected_h2) <= 1e-12, \
            f"{task}: figure baseline {baseline} != H2 {expected_h2}"
        warm_record = warm_tasks.get(task) or {}
        warm_ledger = warm_record.get("ledger") or {}
        assert Path(warm_record.get("source_summary_all", "")).is_file(), \
            f"{task}: original H2 summary provenance missing"
        assert abs(float(warm_record.get("h2_best_score_exact")) - baseline) <= 1e-6, \
            f"{task}: recovered H2 score mismatch"
        assert abs(float(warm_record.get("seed_score_exact")) -
                   float(baselines[task]["seed"])) <= 1e-6, \
            f"{task}: recovered initial-program score mismatch"
        assert int(warm_record.get("evaluations") or 0) == 20, \
            f"{task}: H2 warm-start did not use 20 evaluations"
        assert int(warm_ledger.get("max_evaluator_calls") or 0) == 20, \
            f"{task}: H2 warm-start cap mismatch"
        assert warm_record.get("program_inherited_by_comparison_arms") is False, \
            f"{task}: provenance claims H2 program inheritance"

        series = task_data["series"]
        curves = {
            "proposer": series["proposer_full"]["points"],
            "context": series["context"]["points"],
            "executor": series["executor"]["points"],
        }
        if not args.allow_incomplete:
            assert math.isclose(
                float(curves["proposer"][-1]["score"]),
                float(task_data["reported_proposer_combined_score"]),
                abs_tol=1e-12,
            ), f"{task}: proposer curve endpoint is not aligned with report"
        for method, points in curves.items():
            xs = [int(p["x"]) for p in points]
            scores = [float(p["score"]) for p in points]
            assert xs == sorted(set(xs)), f"{task}/{method}: x not strictly increasing"
            assert all(b + 1e-12 >= a for a, b in zip(scores, scores[1:])), \
                f"{task}/{method}: best-so-far decreased"
            assert abs(scores[0] - baseline) <= 1e-12, \
                f"{task}/{method}: wrong H2 anchor"
            assert xs[0] == 1 and points[0].get("source") == "shared_h2_warm_start", \
                f"{task}/{method}: warm-start provenance missing"
            assert points[0].get("program_inherited") is False, \
                f"{task}/{method}: H2 program must not be inherited"
            cumulative = 1
            for point in non_anchor(points):
                launched = int(point.get("launched") or 0)
                max_evals = int(point.get("max_evals_per_trajectory") or 0)
                charged = int(point.get("charged_evaluator_call_budget") or 0)
                assert launched > 0 and max_evals > 0, \
                    f"{task}/{method}: missing rollout/evaluator budget ledger"
                assert charged == launched * max_evals, \
                    f"{task}/{method}: inconsistent evaluator budget charge"
                cumulative += launched
                assert int(point["x"]) == cumulative, \
                    f"{task}/{method}: x does not charge warm-start + launches"

        context_points = non_anchor(curves["context"])
        executor_points = non_anchor(curves["executor"])
        updated_executor = [p for p in executor_points if int(p["step"]) > 0]
        context_review: dict[str, Any] = {}
        executor_review: dict[str, Any] = {}
        required_updates = REQUIRED_EXECUTOR_UPDATES[task]
        if not args.allow_incomplete:
            assert len(context_points) >= REQUIRED_CONTEXT_ROUNDS, \
                f"{task}: only {len(context_points)} clean context rounds"
            assert len(updated_executor) >= required_updates, \
                f"{task}: only {len(updated_executor)}/{required_updates} executor updates"

        workspace = task_data["context_workspace"]
        context_manifest = RUN_ROOT / workspace / "run_manifest.json"
        context_first_seed_sha256: str | None = None
        context_continuity: dict[str, Any] = {}
        if context_manifest.exists():
            manifest = json.loads(context_manifest.read_text())
            assert manifest["isolated_feedback"], \
                f"{task}: context feedback is not isolated"
            context_driver = context_manifest.with_name("driver.log")
            if context_driver.is_file():
                context_text = context_driver.read_text(errors="ignore")
            else:
                assert args.allow_incomplete, (
                    f"{task}: context manifest exists but accepted driver "
                    "evidence is missing"
                )
                context_text = ""
            assert not re.search(
                r"^\[[^]]+\]\s+\[ctx\]\s+trained ->",
                context_text,
                flags=re.MULTILINE,
            ), f"{task}: context trained weights"
            context_bases = json.loads(
                context_manifest.with_name("round000_bases.json").read_text())
            context_base = context_bases[task]
            assert Path(context_base["package"]).resolve() == (
                REPO / "src/inner/harness").resolve(), \
                f"{task}: context did not start from fixed H2 harness"
            assert abs(float(context_base["score"]) - baseline) <= 1e-12, \
                f"{task}: context warm-start score mismatch"
            assert abs(float(context_base["seed_score"]) -
                       float(baselines[task]["seed"])) <= 1e-12, \
                f"{task}: context initial-program score mismatch"
            if context_points:
                first_round = int(context_points[0]["round"])
                first_prompt_path = (
                    Path(manifest["outer_root"]) / f"round{first_round:03d}" /
                    "prompts.json"
                )
                prompts = json.loads(first_prompt_path.read_text())
                seed_excerpt = prompt_seed_excerpt(str(prompts[task]))
                expected_excerpt = INITIAL_PROGRAMS[task].read_text().strip()[:5000]
                assert seed_excerpt == expected_excerpt, \
                    f"{task}: context first batch did not show task.initial_program"
                context_first_seed_sha256 = hashlib.sha256(
                    seed_excerpt.encode()
                ).hexdigest()
            context_continuity = audit_context_round_continuity(
                task,
                context_points,
                Path(manifest["outer_root"]),
                context_bases,
            )
        elif args.allow_incomplete:
            context_text = ""
        else:
            raise AssertionError(f"{task}: missing context manifest {context_manifest}")
        if task in ("adrs__eplb", "adrs__prism", "adrs__txn_scheduling"):
            expected_workspace = {
                "adrs__eplb": "context_sota5_sys_guarded",
                # The earlier system-context PRISM lineage inherited rewards
                # through the terminal-null checkpoint fallback.  It is
                # intentionally replaced by the post-fix, task-isolated
                # AHC058/PRISM campaign.
                "adrs__prism": "context_sota7_rewardfix_v1",
                "adrs__txn_scheduling": "context_sota7_extra_guarded",
            }[task]
            assert workspace == expected_workspace, \
                f"{task}: paper curve is not using the post-proof guarded context run"
            proof_job = int(guard_fields.get("job") or 0)
            assert proof_job > 0, "protocol-guard proof job is missing"
            context_jobs = [
                int(point["job"]) for point in context_points if point.get("job")
            ]
            assert len(context_jobs) == len(context_points), \
                f"{task}: context round-to-job provenance is incomplete"
            if task == "adrs__txn_scheduling":
                assert all(job > proof_job for job in context_jobs), \
                    f"{task}: a context round predates the Txn guard proof"
            elif task == "adrs__eplb":
                preproof_rounds = {
                    int(point["round"])
                    for point in context_points
                    if point.get("job") and int(point["job"]) <= proof_job
                }
                replayed = set((
                    (legacy_guard_proof.get("scope") or {}).get(
                        "adrs__eplb", {}
                    ).get("isolated_context_rounds_replayed") or []
                ))
                assert preproof_rounds <= replayed, \
                    "an EPLB pre-current-proof context round lacks hash replay"
        if not args.allow_incomplete:
            assert context_text.count(
                "phi UNCHANGED (context-only ablation)") >= REQUIRED_CONTEXT_ROUNDS
            context_workspace = context_manifest.parent
            assert (context_workspace / "CANONICAL_CONTEXT_COMPLETE").is_file(), \
                f"{task}: context campaign has not passed plateau/budget review"
            context_review_path = context_workspace / "plateau_review.json"
            assert context_review_path.is_file(), \
                f"{task}: context plateau review is missing"
            context_review = json.loads(context_review_path.read_text())
            assert context_review.get("status") in (
                "three_transition_empirical_plateau",
                "budget_limited_at_explicit_cap",
            ), f"{task}: invalid context completion status"
            assert task in (context_review.get("tasks") or []), \
                f"{task}: absent from context plateau review"
            assert int(context_review.get("completed_round") or -1) == int(
                context_points[-1]["round"]
            ), f"{task}: plotted context endpoint is not the reviewed endpoint"
        else:
            context_review_path = context_manifest.parent / "plateau_review.json"
            if context_review_path.is_file():
                context_review = json.loads(context_review_path.read_text())

        if executor_points:
            assert executor_points[0]["checkpoint"] == BASE, \
                f"{task}: executor step0 is not base"
            state_sources = [Path(path) for path in series["executor"]["sources"]
                             if path.endswith("/state.json")]
            assert len(state_sources) == 1 and state_sources[0].is_file(), \
                f"{task}: executor state provenance missing"
            executor_state = json.loads(state_sources[0].read_text())
            executor_state_dir = state_sources[0].parent
            executor_review_path = executor_state_dir / "plateau_review.json"
            if not args.allow_incomplete:
                assert (executor_state_dir / "CANONICAL_EXECUTOR_COMPLETE").is_file(), \
                    f"{task}: executor campaign has not passed plateau/budget review"
                assert executor_review_path.is_file(), \
                    f"{task}: executor plateau review is missing"
                executor_review = json.loads(executor_review_path.read_text())
                assert executor_review.get("status") in (
                    "three_transition_empirical_plateau",
                    "budget_limited_at_explicit_cap",
                ), f"{task}: invalid executor completion status"
                assert int(executor_review.get("completed_update") or -1) == int(
                    executor_points[-1]["step"]
                ), f"{task}: plotted executor endpoint is not the reviewed endpoint"
            elif executor_review_path.is_file():
                executor_review = json.loads(executor_review_path.read_text())
            state_batches = {
                int(batch["step"]): batch
                for batch in executor_state.get("batches") or []
            }
            root = executor_state["archive"]["root"]
            # AHC baseline JSON rounds the native score to six decimals while
            # the recovered summary/root registry retains the full value.
            assert abs(float(root["score"]) -
                       float(baselines[task]["seed"])) <= 1e-6, \
                f"{task}: PUCT root was assigned a non-seed score"
            assert root.get("score_semantics") == "score of task.initial_program", \
                f"{task}: PUCT root score semantics missing"
            warm = executor_state.get("common_warm_start") or {}
            assert abs(float(warm.get("score", float("nan"))) - baseline) <= 1e-12, \
                f"{task}: executor shared H2 warm-start mismatch"
            assert warm.get("program_inherited") is False, \
                f"{task}: executor improperly inherited H2 program"
            for point in executor_points:
                step = int(point["step"])
                batch = state_batches.get(step)
                assert batch is not None, \
                    f"{task}: executor step{step} missing from persistent state"
                assert int(point.get("usable") or 0) >= executor_batch_k, \
                    f"{task}: executor step{step} is a partial usable batch"
                assert int(point.get("train_rows") or 0) >= executor_batch_k, \
                    f"{task}: executor step{step} has an undersized replay"
                assert int(batch.get("usable") or 0) == int(point["usable"]), \
                    f"{task}: executor step{step} usable count disagrees with state"
                assert int(batch.get("train_rows") or 0) == int(point["train_rows"]), \
                    f"{task}: executor step{step} replay count disagrees with state"
                assert int(batch.get("launched") or 0) == int(point["launched"]), \
                    f"{task}: executor step{step} launch count disagrees with state"
        for point in updated_executor:
            checkpoint = Path(point["checkpoint"])
            assert str(checkpoint) != BASE and (checkpoint / "config.json").is_file(), \
                f"{task}: invalid merged executor checkpoint {checkpoint}"

        common_budget = min(int(points[-1]["x"]) for points in curves.values())
        metrics = {}
        for method, points in curves.items():
            point = at_budget(points, common_budget)
            measured = [
                candidate for candidate in non_anchor(points)
                if candidate.get("status") != "failed_without_round_summary"
            ]
            tail_window = measured[-4:]
            plateau_last_three_batches = (
                len(tail_window) == 4 and
                float(tail_window[-1]["score"]) <=
                float(tail_window[0]["score"]) + 1e-12
            )
            improvement_points = [
                candidate for previous, candidate in zip(points, points[1:])
                if float(candidate["score"]) > float(previous["score"]) + 1e-12
            ]
            metrics[method] = {
                "endpoint_x": int(points[-1]["x"]),
                "endpoint_score": float(points[-1]["score"]),
                "endpoint_ratio": ratio(points[-1]["score"], reference),
                "score_at_common_budget": float(point["score"]),
                "ratio_at_common_budget": ratio(point["score"], reference),
                "log_auc_to_common_budget": log_step_auc(points, common_budget, reference),
                "first_human_best_crossing_x": first_crossing(points, reference),
                "first_human_best_match_within_reported_precision_x": (
                    first_crossing(
                        points,
                        reference - REFERENCE_REPORTING_HALF_UNIT[task],
                    )
                ),
                "has_non_anchor_batch_by_common_budget": any(
                    int(p["x"]) <= common_budget for p in non_anchor(points)),
                "last_improvement_x": (
                    int(improvement_points[-1]["x"])
                    if improvement_points else None
                ),
                "plateau_last_three_batches": plateau_last_three_batches,
                "observed_ceiling_status": (
                    "three-transition empirical plateau"
                    if plateau_last_three_batches else
                    "budget-limited endpoint; not a ceiling"
                ),
            }

        if not args.allow_incomplete:
            for method, review in (
                ("context", context_review), ("executor", executor_review)
            ):
                reviewed_plateau = review.get("status") == (
                    "three_transition_empirical_plateau"
                )
                assert bool(metrics[method]["plateau_last_three_batches"]) == \
                    reviewed_plateau, (
                        f"{task}/{method}: plotted tail disagrees with completion review"
                    )

        evaluator_curves = {
            method: evaluator_budget_curve(points)
            for method, points in curves.items()
        }
        common_evaluator_budget = min(
            int(points[-1]["budget"]) for points in evaluator_curves.values()
        )
        evaluator_budget_metrics = {}
        for method, cost_curve in evaluator_curves.items():
            endpoint_score = float(cost_curve[-1]["score"])
            common_score = eval_score_at_budget(
                cost_curve, common_evaluator_budget)
            evaluator_budget_metrics[method] = {
                "endpoint_charged_evaluator_calls": int(cost_curve[-1]["budget"]),
                "endpoint_score": endpoint_score,
                "endpoint_ratio": ratio(endpoint_score, reference),
                "score_at_common_evaluator_budget": common_score,
                "ratio_at_common_evaluator_budget": ratio(common_score, reference),
                "log_auc_to_common_evaluator_budget": eval_log_step_auc(
                    cost_curve, common_evaluator_budget, reference),
                "first_human_best_crossing_charged_evaluator_calls": (
                    eval_first_crossing(cost_curve, reference)
                ),
                "first_human_best_match_within_reported_precision_charged_evaluator_calls": (
                    eval_first_crossing(
                        cost_curve,
                        reference - REFERENCE_REPORTING_HALF_UNIT[task],
                    )
                ),
                "curve": cost_curve,
            }

        model_call_curves = {
            method: model_call_budget_curve(method, points)
            for method, points in curves.items()
        }
        common_model_call_budget = min(
            int(points[-1]["budget"]) for points in model_call_curves.values()
        )
        model_call_budget_metrics = {}
        for method, cost_curve in model_call_curves.items():
            common_score = eval_score_at_budget(cost_curve, common_model_call_budget)
            model_call_budget_metrics[method] = {
                "endpoint_recorded_model_calls_lower_bound": int(
                    cost_curve[-1]["budget"]),
                "endpoint_score": float(cost_curve[-1]["score"]),
                "endpoint_ratio": ratio(cost_curve[-1]["score"], reference),
                "score_at_common_recorded_model_call_budget": common_score,
                "ratio_at_common_recorded_model_call_budget": ratio(
                    common_score, reference),
                "log_auc_to_common_recorded_model_call_budget": eval_log_step_auc(
                    cost_curve, common_model_call_budget, reference),
                "first_human_best_crossing_recorded_model_calls": (
                    eval_first_crossing(cost_curve, reference)
                ),
                "first_human_best_match_within_reported_precision_recorded_model_calls": (
                    eval_first_crossing(
                        cost_curve,
                        reference - REFERENCE_REPORTING_HALF_UNIT[task],
                    )
                ),
                "curve": cost_curve,
            }

        proposer_update_tags, proposer_drivers = driver_update_count(
            series["proposer_full"]["sources"])
        proposer_weight_jobs = proposer_weight_job_ledger(proposer_update_tags)
        # The same immutable LoRA logs used for the executor audit expose the
        # proposer hyperparameters.  Recover them explicitly so the paper does
        # not imply that the rollout x-axis is also an equal-training-compute
        # comparison.
        proposer_train_configs = [
            executor_train_config(job)
            for job in proposer_weight_jobs["train"]
        ]
        proposer_train_configs = [
            config for config in proposer_train_configs
            if config is not None
        ]
        proposer_points = non_anchor(curves["proposer"])
        proposer_rollouts = sum(int(p.get("launched") or 0) for p in proposer_points)
        proposer_eval_budget = sum(
            int(p.get("charged_evaluator_call_budget") or 0)
            for p in proposer_points)
        proposer_recorded_evals = sum(
            int(p.get("recorded_evaluator_calls") or 0) for p in proposer_points)
        proposer_executor_calls = sum(
            int(p.get("recorded_executor_model_calls") or 0) for p in proposer_points)
        proposer_sandbox_seconds = sum(
            float(p.get("recorded_sandbox_seconds") or 0.0) for p in proposer_points)
        proposer_proposals = sum(int(p.get("proposed") or 0) for p in proposer_points)
        proposer_model_calls = sum(
            int(p.get("h1_model_calls") or 0) for p in proposer_points)
        proposer_reviewer_calls = sum(
            int(p.get("reviewer_model_calls_lower_bound") or 0)
            for p in proposer_points
        )
        proposer_zero_launch_failures = list(
            task_data.get("proposer_failed_outer_jobs") or [])
        proposer_zero_launch_job_ids = {
            str(row["job"]) for row in proposer_zero_launch_failures
            if row.get("job") is not None
        }
        context_rollouts = sum(int(p.get("launched") or 0) for p in context_points)
        context_eval_budget = sum(
            int(p.get("charged_evaluator_call_budget") or 0)
            for p in context_points)
        context_recorded_evals = sum(
            int(p.get("recorded_evaluator_calls") or 0) for p in context_points)
        context_executor_calls = sum(
            int(p.get("recorded_executor_model_calls") or 0) for p in context_points)
        context_sandbox_seconds = sum(
            float(p.get("recorded_sandbox_seconds") or 0.0) for p in context_points)
        context_proposals = sum(int(p.get("proposed") or 0) for p in context_points)
        context_model_calls = sum(
            int(p.get("h1_model_calls") or 0) for p in context_points)
        context_reviewer_calls = sum(
            int(p.get("reviewer_model_calls_lower_bound") or 0)
            for p in context_points
        )
        analyzer_briefs = sum(int(p.get("analyst_briefs") or 0)
                              for p in context_points)
        analyzer_calls = sum(int(p.get("analyzer_model_calls") or 0)
                             for p in context_points)
        executor_rollouts = sum(int(p.get("launched") or 0) for p in executor_points)
        executor_eval_budget = sum(
            int(p.get("charged_evaluator_call_budget") or 0)
            for p in executor_points)
        executor_recorded_evals = sum(
            int(p.get("recorded_evaluator_calls") or 0) for p in executor_points)
        executor_model_calls = sum(
            int(p.get("recorded_executor_model_calls") or 0) for p in executor_points)
        executor_sandbox_seconds = sum(
            float(p.get("recorded_sandbox_seconds") or 0.0) for p in executor_points)
        jobs = executor_job_ledger(series["executor"]["sources"])
        train_configs = [executor_train_config(job) for job in jobs["train"]]
        train_configs = [config for config in train_configs if config is not None]
        failed_train_jobs = set(jobs.get("failed_train") or [])
        for config in train_configs:
            config["failed_attempt"] = str(config["job"]) in failed_train_jobs
        successful_train_configs = [
            config for config in train_configs if not config["failed_attempt"]
        ]
        merge_sanity = [executor_merge_sanity(job) for job in jobs["merge"]]
        merge_sanity = [row for row in merge_sanity if row is not None]
        failed_merge_jobs = set(jobs.get("failed_merge") or [])
        for row in merge_sanity:
            row["failed_attempt"] = str(row["job"]) in failed_merge_jobs
        successful_merge_sanity = [
            row for row in merge_sanity if not row["failed_attempt"]
        ]
        if not args.allow_incomplete and jobs["train"]:
            assert len(train_configs) == len(jobs["train"]), \
                f"{task}: missing executor train configuration log"
            expected_kl = 0.01 if "ahc" in task else 0.1
            expected_lr = 2e-5 if task == "eft__ahc_simpletes__ahc058" else 4e-5
            assert all(abs(float(config.get("kl_coefficient", -1)) - expected_kl) < 1e-12
                       for config in train_configs), \
                f"{task}: executor KL does not match published task-category setting"
            assert len(successful_train_configs) >= len(updated_executor), \
                f"{task}: updated checkpoints lack successful train jobs"
            assert all(int(config.get("global_batch_size", -1)) == 4
                       for config in successful_train_configs), \
                f"{task}: canonical executor update did not use GBS=4"
            assert all(int(config.get("num_epochs", -1)) == 1
                       for config in successful_train_configs), \
                f"{task}: canonical executor update replayed more than one epoch"
            assert all(int(config.get("lora_rank", -1)) == 32 and
                       int(config.get("lora_alpha", -1)) == 64
                       for config in successful_train_configs), \
                f"{task}: canonical executor LoRA shape mismatch"
            assert all(abs(float(config.get("learning_rate", -1)) - expected_lr) < 1e-12
                       for config in successful_train_configs), \
                f"{task}: canonical executor learning rate mismatch"
            assert all(abs(float(config.get("adam_beta2", -1)) - 0.98) < 1e-12
                       for config in successful_train_configs), \
                f"{task}: local executor Adam beta2 changed without disclosure"
            assert all(abs(float(config.get("weight_decay", -1)) - 0.1) < 1e-12
                       for config in successful_train_configs), \
                f"{task}: local executor weight decay changed without disclosure"
            assert len(successful_merge_sanity) >= len(updated_executor), \
                f"{task}: updated checkpoints lack merge sanity evidence"
            assert all(row["nonzero"] for row in successful_merge_sanity), \
                f"{task}: a canonical executor adapter is all-zero"
        if not args.allow_incomplete and proposer_weight_jobs["train"]:
            assert len(proposer_train_configs) == len(
                proposer_weight_jobs["train"]
            ), f"{task}: missing proposer train configuration log"
            assert all(
                int(config.get("global_batch_size", -1)) == 8
                and int(config.get("num_epochs", -1)) == 3
                and int(config.get("lora_rank", -1)) == 64
                and int(config.get("lora_alpha", -1)) == 128
                and abs(float(config.get("learning_rate", -1)) - 3e-5) < 1e-12
                and abs(float(config.get("kl_coefficient", -1)) - 0.05) < 1e-12
                and abs(float(config.get("adam_beta2", -1)) - 0.98) < 1e-12
                and abs(float(config.get("weight_decay", -1)) - 0.1) < 1e-12
                for config in proposer_train_configs
            ), f"{task}: proposer LoRA training configuration changed"

        task_costs = {
            "proposer": {
                "executor_trajectories": proposer_rollouts,
                "charged_evaluator_call_budget": proposer_eval_budget,
                "recorded_evaluator_calls_lower_bound": proposer_recorded_evals,
                "recorded_executor_model_calls_lower_bound": proposer_executor_calls,
                "recorded_total_model_calls_lower_bound": (
                    proposer_executor_calls + proposer_model_calls
                    + proposer_reviewer_calls),
                "recorded_sandbox_seconds_lower_bound": proposer_sandbox_seconds,
                "harness_proposals": proposer_proposals,
                "proposer_model_calls": proposer_model_calls,
                "reviewer_model_calls_lower_bound": proposer_reviewer_calls,
                "analyzer_calls": 0,
                "successful_weight_conditioned_batches": sum(
                    1 for point in proposer_points
                    if point.get("phi") and
                    str(point.get("phi")) != BASE.rsplit("/", 1)[-1]
                ),
                "weight_conditioning_events_logged": len(proposer_update_tags),
                "distinct_weight_checkpoint_tags": len(set(proposer_update_tags)),
                "weight_update_attempt_jobs": len(proposer_weight_jobs["train"]),
                "planned_optimizer_boundaries": sum(
                    int(config.get("planned_optimizer_boundaries") or 0)
                    for config in proposer_train_configs
                ),
                "weight_update_tags": proposer_update_tags,
                "outer_jobs": sorted(
                    {str(p["job"]) for p in proposer_points if p.get("job")}
                    | proposer_zero_launch_job_ids,
                    key=int),
                "failed_zero_trajectory_outer_jobs": proposer_zero_launch_failures,
                "train_jobs": proposer_weight_jobs["train"],
                "merge_jobs": proposer_weight_jobs["merge"],
                "train_configs": proposer_train_configs,
                "unmapped_weight_update_tags": proposer_weight_jobs["unmapped_tags"],
                "drivers": proposer_drivers,
            },
            "context": {
                "executor_trajectories": context_rollouts,
                "charged_evaluator_call_budget": context_eval_budget,
                "recorded_evaluator_calls_lower_bound": context_recorded_evals,
                "recorded_executor_model_calls_lower_bound": context_executor_calls,
                "recorded_total_model_calls_lower_bound": (
                    context_executor_calls + context_model_calls
                    + context_reviewer_calls + analyzer_calls),
                "recorded_sandbox_seconds_lower_bound": context_sandbox_seconds,
                "harness_proposals": context_proposals,
                "proposer_model_calls": context_model_calls,
                "reviewer_model_calls_lower_bound": context_reviewer_calls,
                "analyzer_calls": analyzer_calls,
                "analyzer_briefs": analyzer_briefs,
                "analyzer_calls_per_brief": 2,
                "weight_updates": 0,
                "planned_optimizer_boundaries": 0,
                "outer_jobs": sorted({str(p["job"]) for p in context_points if p.get("job")}, key=int),
            },
            "executor": {
                "executor_trajectories": executor_rollouts,
                "charged_evaluator_call_budget": executor_eval_budget,
                "recorded_evaluator_calls_lower_bound": executor_recorded_evals,
                "recorded_executor_model_calls_lower_bound": executor_model_calls,
                "recorded_total_model_calls_lower_bound": executor_model_calls,
                "recorded_sandbox_seconds_lower_bound": executor_sandbox_seconds,
                "harness_proposals": 0,
                "analyzer_calls": 0,
                "successful_weight_updates": len(updated_executor),
                "weight_update_attempt_jobs": len(jobs["train"]),
                "planned_optimizer_boundaries_attempted": sum(
                    int(config.get("planned_optimizer_boundaries") or 0)
                    for config in train_configs
                ),
                "planned_optimizer_boundaries_successful": sum(
                    int(config.get("planned_optimizer_boundaries") or 0)
                    for config in successful_train_configs
                ),
                "jobs": jobs,
                "train_configs": train_configs,
                "merge_sanity": merge_sanity,
            },
        }
        common_scores = {
            method: float(metrics[method]["score_at_common_budget"])
            for method in ("proposer", "context", "executor")
        }
        endpoint_scores = {
            method: float(metrics[method]["endpoint_score"])
            for method in ("proposer", "context", "executor")
        }
        common_best = max(common_scores.values())
        endpoint_best = max(endpoint_scores.values())
        result["tasks"][task] = {
            "human_best_reference": reference,
            "reference": reference,
            "reference_caveat": REFERENCE_CAVEATS.get(task),
            "reference_valid_under_current_protocol": True,
            "reference_reporting_half_unit": REFERENCE_REPORTING_HALF_UNIT[task],
            "fixed_h2": baseline,
            "initial_program_score": float(baselines[task]["seed"]),
            "shared_warm_start_cost": {
                "executor_trajectories": 1,
                "max_evaluator_calls": 20,
                "recorded_evaluator_calls": int(
                    warm_ledger.get("evaluator_calls") or 0),
                "recorded_executor_model_calls": int(
                    warm_ledger.get("llm_calls") or 0),
                "recorded_sandbox_seconds": float(
                    warm_ledger.get("sandbox_seconds") or 0.0),
                "source_summary_all": warm_record["source_summary_all"],
                "best_program_sha256": warm_record[
                    "h2_best_program_sha256"],
                "current_guard_replay_score": (
                    (legacy_guard_proof.get(
                        "warm_start_current_guard_scores"
                    ) or {}).get(task)
                ),
                "included_in_display_x": True,
                "excluded_from_incremental_method_costs": True,
            },
            "common_budget": common_budget,
            "common_incremental_rollout_budget": common_budget - 1,
            "common_charged_evaluator_call_budget": common_evaluator_budget,
            "common_recorded_model_call_budget_lower_bound": common_model_call_budget,
            "metrics": metrics,
            "completion_reviews": {
                "context": context_review,
                "executor": executor_review,
            },
            "context_first_adaptive_batch": {
                "program": "task.initial_program",
                "seed_excerpt_sha256": context_first_seed_sha256,
                "verified": context_first_seed_sha256 is not None,
            },
            "context_task_local_continuity": context_continuity,
            "common_budget_ranking": {
                "scores": common_scores,
                "best_methods_including_ties": [
                    method for method, score in common_scores.items()
                    if abs(score - common_best) <= 1e-12
                ],
            },
            "observed_endpoint_ranking": {
                "scores": endpoint_scores,
                "best_methods_including_ties": [
                    method for method, score in endpoint_scores.items()
                    if abs(score - endpoint_best) <= 1e-12
                ],
                "caveat": "endpoint budgets differ; this is not an efficiency ranking",
            },
            "evaluator_call_budget_metrics": evaluator_budget_metrics,
            "model_call_budget_metrics": model_call_budget_metrics,
            "logical_cost_to_common_budget": {
                method: logical_cost_to_budget(method, points, common_budget)
                for method, points in curves.items()
            },
            "costs": task_costs,
            "reported_proposer_score": task_data["reported_proposer_combined_score"],
            "reported_endpoint_has_local_compute_ledger": task_data[
                "reported_endpoint_has_local_compute_ledger"],
        }

        for method in ("proposer", "context"):
            suite = result["suite_costs"][method]
            task_cost = task_costs[method]
            suite["executor_trajectories"] += task_cost["executor_trajectories"]
            suite["charged_evaluator_call_budget"] += task_cost[
                "charged_evaluator_call_budget"]
            for key in (
                "recorded_evaluator_calls_lower_bound",
                "recorded_executor_model_calls_lower_bound",
                "recorded_total_model_calls_lower_bound",
                "recorded_sandbox_seconds_lower_bound",
            ):
                suite[key] += task_cost[key]
            suite["harness_proposals"] += task_cost["harness_proposals"]
            suite["proposer_model_calls"] += task_cost["proposer_model_calls"]
            suite["reviewer_model_calls_lower_bound"] += task_cost[
                "reviewer_model_calls_lower_bound"
            ]
            suite["analyzer_calls"] += task_cost["analyzer_calls"]
            if method == "context":
                suite["analyzer_briefs"] += task_cost["analyzer_briefs"]
            suite["weight_updates"] += task_cost.get(
                "weight_update_attempt_jobs", task_cost.get("weight_updates", 0))
            suite["planned_optimizer_boundaries"] += task_cost.get(
                "planned_optimizer_boundaries", 0
            )
            suite["outer_jobs"].extend(task_cost["outer_jobs"])
            if method == "proposer":
                suite["train_jobs"].extend(task_cost["train_jobs"])
                suite["merge_jobs"].extend(task_cost["merge_jobs"])
                suite["unmapped_weight_update_tags"].extend(
                    task_cost["unmapped_weight_update_tags"])
        suite = result["suite_costs"]["executor"]
        suite["executor_trajectories"] += executor_rollouts
        suite["charged_evaluator_call_budget"] += executor_eval_budget
        suite["recorded_evaluator_calls_lower_bound"] += executor_recorded_evals
        suite["recorded_executor_model_calls_lower_bound"] += executor_model_calls
        suite["recorded_total_model_calls_lower_bound"] += executor_model_calls
        suite["recorded_sandbox_seconds_lower_bound"] += executor_sandbox_seconds
        suite["weight_updates"] += len(jobs["train"])
        suite["planned_optimizer_boundaries"] += sum(
            int(config.get("planned_optimizer_boundaries") or 0)
            for config in train_configs
        )
        for source, dest in (("eval", "eval_jobs"), ("train", "train_jobs"),
                             ("merge", "merge_jobs")):
            suite[dest].extend(jobs[source])

    for costs in result["suite_costs"].values():
        for key, value in list(costs.items()):
            if key.endswith("jobs"):
                costs[key] = sorted(set(value), key=int)
            elif key == "unmapped_weight_update_tags":
                costs[key] = sorted(set(value))

    proposer_shape_configs = [
        executor_train_config(job)
        for job in result["suite_costs"]["proposer"]["train_jobs"]
    ]
    executor_shape_configs = [
        executor_train_config(job)
        for job in result["suite_costs"]["executor"]["train_jobs"]
    ]
    proposer_shape_configs = [row for row in proposer_shape_configs if row]
    executor_shape_configs = [row for row in executor_shape_configs if row]
    result["weight_update_compute_shape"] = {
        "proposer": {
            "trainable_parameters_observed": sorted({
                int(row["trainable_parameters"])
                for row in proposer_shape_configs
                if row.get("trainable_parameters") is not None
            }),
            "lora_rank": 64,
            "epochs_per_attempt": 3,
            "global_batch_size": 8,
            "planned_optimizer_boundaries_per_job_observed": sorted({
                int(row["planned_optimizer_boundaries"])
                for row in proposer_shape_configs
                if row.get("planned_optimizer_boundaries") is not None
            }),
            "attempted_training_jobs": len(
                result["suite_costs"]["proposer"]["train_jobs"]),
            "planned_optimizer_boundaries": result["suite_costs"][
                "proposer"]["planned_optimizer_boundaries"],
        },
        "executor": {
            "trainable_parameters_observed": sorted({
                int(row["trainable_parameters"])
                for row in executor_shape_configs
                if row.get("trainable_parameters") is not None
            }),
            "lora_rank": 32,
            "epochs_per_attempt": 1,
            "global_batch_size": 4,
            "planned_optimizer_boundaries_per_job_observed": sorted({
                int(row["planned_optimizer_boundaries"])
                for row in executor_shape_configs
                if row.get("planned_optimizer_boundaries") is not None
            }),
            "attempted_training_jobs": len(
                result["suite_costs"]["executor"]["train_jobs"]),
            "planned_optimizer_boundaries": result["suite_costs"][
                "executor"]["planned_optimizer_boundaries"],
        },
        "interpretation": (
            "the proposer update has twice the trainable LoRA parameters and "
            "three replay epochs; the local executor update has half-rank LoRA, "
            "one epoch, and two K=8/GBS=4 optimizer boundaries. Rollout matching "
            "therefore does not match training compute."
        ),
    }
    if not args.allow_incomplete:
        assert not result["suite_costs"]["proposer"]["unmapped_weight_update_tags"], \
            "some historical proposer updates lack train/merge job provenance"
        assert result["weight_update_compute_shape"]["proposer"][
            "trainable_parameters_observed"] == [116391936], \
            "proposer trainable-parameter count changed"
        assert result["weight_update_compute_shape"]["executor"][
            "trainable_parameters_observed"] == [58195968], \
            "executor trainable-parameter count changed"

    result["compute_timing_proxy"] = {
        "proposer": timing_ledger({
            "outer": result["suite_costs"]["proposer"]["outer_jobs"],
            "train": result["suite_costs"]["proposer"]["train_jobs"],
            "merge": result["suite_costs"]["proposer"]["merge_jobs"],
        }),
        "context": timing_ledger({
            "outer": result["suite_costs"]["context"]["outer_jobs"],
        }),
        "executor": timing_ledger({
            "eval": result["suite_costs"]["executor"]["eval_jobs"],
            "train": result["suite_costs"]["executor"]["train_jobs"],
            "merge": result["suite_costs"]["executor"]["merge_jobs"],
        }),
        "interpretation": (
            "Suite totals deduplicate shared jobs. Plotted proposer and executor "
            "campaigns are task-local; context jobs jointly serve two or three of "
            "the plotted tasks, so only their deduplicated suite total—not a per-task "
            "split—is meaningful. Allocation time is a cross-check, not a FLOP "
            "estimate; logical operation counts remain the exact task-local ledger."
        ),
    }
    if not args.allow_incomplete:
        for method, ledger in result["compute_timing_proxy"].items():
            if method == "interpretation":
                continue
            assert not ledger["sacct_jobs_missing"], (
                f"{method}: authoritative sacct rows are missing for "
                f"{ledger['sacct_jobs_missing']}"
            )
            assert all(
                int((row.get("slurm_accounting") or {}).get(
                    "allocated_gpus_sacct", 0
                )) > 0
                for row in ledger["jobs"]
            ), f"{method}: a charged GPU job has no GPU allocation in sacct"
            assert ledger["sacct_snapshot"] == str(SACCT_SNAPSHOT), (
                f"{method}: strict audit requires a frozen sacct snapshot"
            )
            assert int(ledger["sacct_jobs_from_frozen_snapshot"]) == len(
                ledger["jobs"]
            ), f"{method}: not every accepted job is frozen in the sacct snapshot"
    for method in ("proposer", "context", "executor"):
        timing = result["compute_timing_proxy"][method]
        trajectories = int(result["suite_costs"][method][
            "executor_trajectories"])
        timing["seven_task_ledgered_executor_trajectories"] = trajectories
        timing["allocated_gpu_hours_proxy_per_ledgered_trajectory"] = (
            float(timing["allocated_gpu_hours_proxy"]) / trajectories
            if trajectories else None
        )
        timing["allocated_gpu_hours_sacct_per_ledgered_trajectory"] = (
            float(timing["allocated_gpu_hours_sacct"]) / trajectories
            if trajectories and timing["sacct_jobs_recovered"] else None
        )
        timing["normalization_caveat"] = (
            "whole task-local jobs; AHC039 includes multiple disclosed restarts; "
            "still not a FLOP estimate"
            if method == "proposer" else (
                "deduplicated multi-task suite allocation divided by suite trajectories; "
                "no per-task allocation is implied"
                if method == "context" else
                "whole task-local allocation divided by task-local trajectories; "
                "still not a FLOP estimate"
            )
        )

    # A live shell-tail edit caused a small, fixed set of outer batch steps to
    # exit nonzero after all requested trajectories and both atomic collector
    # artifacts had already been written.  These are accepted protocol jobs,
    # not zero-work infrastructure retries: retain their terminal evidence and
    # charge their complete allocation.  Keeping the set in a checked registry
    # makes that exception explicit and prevents a partially materialized job
    # from receiving the same treatment later.
    anomaly_registry = json.loads(ACCEPTED_JOB_ANOMALIES.read_text())
    assert int(anomaly_registry.get("schema") or 0) == 1
    anomaly_entries = list(anomaly_registry.get("entries") or [])
    anomaly_records = []
    seen_anomaly_jobs: set[str] = set()
    for entry in anomaly_entries:
        route = str(entry.get("route") or "")
        assert route in ("proposer", "context"), (
            f"unknown accepted-anomaly route: {route}"
        )
        outer_job = str(entry.get("outer_job") or "")
        assert outer_job.isdigit(), (
            f"accepted anomaly lacks an outer job: {entry}"
        )
        assert outer_job not in seen_anomaly_jobs, (
            f"duplicate accepted anomaly job: {outer_job}"
        )
        seen_anomaly_jobs.add(outer_job)
        assert outer_job in result["suite_costs"][route]["outer_jobs"], (
            f"accepted anomaly {outer_job} is absent from the {route} "
            "accepted-cost ledger"
        )

        round_dir = Path(str(entry.get("round_dir") or ""))
        assert round_dir.is_dir(), (
            f"accepted anomaly round is missing: {round_dir}"
        )
        expected = int(entry.get("expected_launched") or 0)
        assert expected > 0
        launch_logs = sorted((round_dir / "rollout_logs").glob("*.log"))
        terminal_summaries = sorted(
            (round_dir / "rollouts").rglob("summary.json")
        )
        assert len(launch_logs) == expected, (
            f"accepted anomaly {outer_job}: {len(launch_logs)} launch logs "
            f"!= expected {expected}"
        )
        assert len(terminal_summaries) == expected, (
            f"accepted anomaly {outer_job}: {len(terminal_summaries)} "
            f"terminal summaries != expected {expected}"
        )
        assert (round_dir / "round_summary.json").is_file(), (
            f"accepted anomaly {outer_job} lacks round_summary.json"
        )
        assert (round_dir / "next_bases.json").is_file(), (
            f"accepted anomaly {outer_job} lacks next_bases.json"
        )

        error_log = SLURM_LOG / f"sah-outer-{outer_job}.err"
        assert error_log.is_file(), (
            f"accepted anomaly {outer_job} lacks its Slurm error log"
        )
        error_text = error_log.read_text(errors="ignore").lower()
        assert "syntax error" in error_text or "command not found" in error_text, (
            f"accepted anomaly {outer_job} does not contain the registered "
            "post-launch shell failure"
        )
        timing_row = next(
            row for row in result["compute_timing_proxy"][route]["jobs"]
            if str(row.get("job")) == outer_job
        )
        accounting = timing_row.get("slurm_accounting")
        if not args.allow_incomplete:
            assert accounting is not None, (
                f"accepted anomaly {outer_job} lacks authoritative sacct data"
            )
            assert int(accounting.get("allocated_gpus_sacct") or 0) > 0, (
                f"accepted anomaly {outer_job} has no charged GPU allocation"
            )
        anomaly_records.append({
            **entry,
            "launch_logs_verified": len(launch_logs),
            "terminal_summaries_verified": len(terminal_summaries),
            "atomic_collector_outputs_verified": [
                str(round_dir / "round_summary.json"),
                str(round_dir / "next_bases.json"),
            ],
            "slurm_error_log": str(error_log),
            "sacct_state": accounting.get("state") if accounting else None,
            "cost_treatment_verified": (
                "included in accepted protocol allocation; not duplicated in "
                "operational-retry overhead"
            ),
        })
    result["accepted_job_anomalies"] = {
        "registry": str(ACCEPTED_JOB_ANOMALIES),
        "entries": anomaly_records,
        "jobs_verified": len(anomaly_records),
        "interpretation": (
            "complete terminal-score evidence is accepted despite a post-work "
            "batch-step failure; each job's full allocation remains in the "
            "accepted protocol cost ledger"
        ),
    }

    # Keep orchestration/infrastructure failures out of the logical method
    # budget while still charging their real allocation in a separate as-run
    # ledger.  This prevents both possible distortions: treating a failed PyPI
    # fetch as algorithmic work, or silently erasing GPU time that was billed.
    retry_registry = json.loads(OPERATIONAL_RETRIES.read_text())
    assert int(retry_registry.get("schema") or 0) == 1
    retry_entries = list(retry_registry.get("entries") or [])
    retry_jobs: dict[str, list[str]] = {
        "proposer": [], "context": [], "executor": []
    }
    for entry in retry_entries:
        route = str(entry.get("route") or "")
        assert route in retry_jobs, f"unknown operational-retry route: {route}"
        assert entry.get("accepted_for_curve") is False
        assert int(entry.get("executor_trajectories_launched") or 0) == 0
        outer_job = str(entry.get("outer_job") or "")
        assert outer_job.isdigit(), f"retry entry lacks outer_job: {entry}"
        retry_jobs[route].append(outer_job)
        quarantine = entry.get("quarantine")
        if quarantine:
            qpath = Path(str(quarantine))
            assert qpath.is_dir(), f"missing operational quarantine: {qpath}"
            assert not list(qpath.rglob("summary.json")), \
                f"quarantined retry unexpectedly launched executor work: {qpath}"
    retry_timing = {
        route: timing_ledger({"outer": sorted(set(jobs), key=int)})
        for route, jobs in retry_jobs.items()
    }
    if not args.allow_incomplete:
        for route, ledger in retry_timing.items():
            assert not ledger["sacct_jobs_missing"], (
                f"{route}: operational-retry sacct rows are missing for "
                f"{ledger['sacct_jobs_missing']}"
            )
            assert all(
                int((row.get("slurm_accounting") or {}).get(
                    "allocated_gpus_sacct", 0
                )) > 0
                for row in ledger["jobs"]
            ), f"{route}: an operational retry has no GPU allocation in sacct"
            assert ledger["sacct_snapshot"] == str(SACCT_SNAPSHOT), (
                f"{route}: strict retry audit requires a frozen sacct snapshot"
            )
            assert int(ledger["sacct_jobs_from_frozen_snapshot"]) == len(
                ledger["jobs"]
            ), f"{route}: not every retry job is frozen in the sacct snapshot"

    # Analyzer-required AHC058 retries are distinct from the zero-trajectory
    # infrastructure failures above.  A rejected attempt may have completed
    # all K=8 executor trajectories, but its missing analyzer marker makes it
    # inadmissible as context-method score evidence.  Keep those trajectories
    # out of the logical curve while charging the complete outer allocation.
    analysis_retry_ledger = (
        RUN_ROOT / "context_sota7_ahc058_analysis_required_v1" /
        "analysis_required_retries.jsonl"
    )
    analysis_rejections: list[dict[str, Any]] = []
    analysis_rejection_jobs: list[str] = []
    if analysis_retry_ledger.is_file():
        for line in analysis_retry_ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row.get("accepted_score_evidence") is False
            assert row.get("charge_full_allocation_as_run") is True
            archive = Path(str(row.get("archive") or ""))
            assert archive.is_dir(), (
                f"missing analyzer-rejection archive: {archive}"
            )
            round_dir = archive / "round"
            launch_logs = list((round_dir / "rollout_logs").glob("*.log"))
            terminal_summaries = list(
                (round_dir / "rollouts").rglob("summary.json")
            )
            job_value = row.get("job")
            job = str(job_value) if job_value is not None else None
            if job is not None:
                assert job.isdigit(), f"invalid rejected analyzer job: {job}"
                analysis_rejection_jobs.append(job)
            elif launch_logs or terminal_summaries:
                raise AssertionError(
                    "an analyzer-rejected attempt launched executor work but "
                    "has no outer job for allocation accounting"
                )
            analysis_rejections.append({
                **row,
                "job": job,
                "discarded_executor_trajectories": len(launch_logs),
                "discarded_terminal_summaries": len(terminal_summaries),
                "score_treatment": (
                    "excluded from logical context curve and endpoint; full "
                    "outer allocation charged as-run"
                ),
            })
    analysis_rejection_jobs = sorted(
        set(analysis_rejection_jobs), key=int
    )
    accepted_context_jobs = set(
        result["suite_costs"]["context"]["outer_jobs"]
    )
    assert not accepted_context_jobs.intersection(analysis_rejection_jobs), (
        "an analyzer-rejected outer job also appears in accepted context cost"
    )
    analysis_rejection_timing = timing_ledger({
        "outer": analysis_rejection_jobs
    })
    if not args.allow_incomplete and analysis_rejection_jobs:
        assert not analysis_rejection_timing["sacct_jobs_missing"], (
            "analyzer-rejection sacct rows are missing for "
            f"{analysis_rejection_timing['sacct_jobs_missing']}"
        )
        assert all(
            int((row.get("slurm_accounting") or {}).get(
                "allocated_gpus_sacct", 0
            )) > 0
            for row in analysis_rejection_timing["jobs"]
        ), "an analyzer-rejected attempt has no charged GPU allocation"
        assert analysis_rejection_timing["sacct_snapshot"] == str(
            SACCT_SNAPSHOT
        ), "strict analyzer-rejection audit requires frozen sacct data"
        assert int(analysis_rejection_timing[
            "sacct_jobs_from_frozen_snapshot"
        ]) == len(analysis_rejection_timing["jobs"]), (
            "not every analyzer-rejected job is frozen in the sacct snapshot"
        )
    result["analysis_required_rejection_costs"] = {
        "ledger": str(analysis_retry_ledger),
        "entries": analysis_rejections,
        "timing": analysis_rejection_timing,
        "discarded_executor_trajectories": sum(
            int(row["discarded_executor_trajectories"])
            for row in analysis_rejections
        ),
        "discarded_terminal_summaries": sum(
            int(row["discarded_terminal_summaries"])
            for row in analysis_rejections
        ),
        "interpretation": (
            "protocol-invalid fail-open analyzer attempts contribute no score "
            "or logical method budget, but their full GPU allocation is "
            "reported in as-run cost"
        ),
    }

    # A whole executed lineage can become inadmissible after a provenance
    # incident even when most of its individual jobs completed normally.  This
    # is distinct from both a zero-work retry and an analyzer-rejected batch:
    # none of its logical work is allowed into the replacement campaign, but
    # every outer/train/merge allocation remains real as-run overhead.
    excluded_registry = json.loads(EXCLUDED_CAMPAIGNS.read_text())
    assert int(excluded_registry.get("schema") or 0) == 1
    excluded_entries = list(excluded_registry.get("entries") or [])
    excluded_job_roles: dict[str, dict[str, list[str]]] = {
        "proposer": {"outer": [], "train": [], "merge": []},
        "context": {"outer": [], "train": [], "merge": []},
        "executor": {"outer": [], "train": [], "merge": []},
    }
    excluded_records: list[dict[str, Any]] = []
    all_excluded_jobs: set[str] = set()
    for entry in excluded_entries:
        route = str(entry.get("route") or "")
        assert route in excluded_job_roles, (
            f"unknown excluded-campaign route: {route}"
        )
        entry_jobs: set[str] = set()
        for key, role in (("outer_jobs", "outer"),
                          ("train_jobs", "train"),
                          ("merge_jobs", "merge")):
            jobs = [str(job) for job in entry.get(key) or []]
            assert all(job.isdigit() for job in jobs), (
                f"excluded campaign has an invalid {key}: {entry}"
            )
            assert len(jobs) == len(set(jobs)), (
                f"excluded campaign duplicates a job within {key}: {entry}"
            )
            excluded_job_roles[route][role].extend(jobs)
            entry_jobs.update(jobs)
        assert entry_jobs, f"excluded campaign has no GPU jobs: {entry}"
        assert not all_excluded_jobs.intersection(entry_jobs), (
            "a GPU job appears in more than one excluded campaign"
        )
        all_excluded_jobs.update(entry_jobs)
        accepted_jobs = {
            str(job) for key in ("outer_jobs", "train_jobs", "merge_jobs",
                                 "eval_jobs")
            for job in result["suite_costs"][route].get(key, [])
        }
        assert not accepted_jobs.intersection(entry_jobs), (
            f"{route}: excluded campaign jobs also appear in accepted costs: "
            f"{sorted(accepted_jobs.intersection(entry_jobs), key=int)}"
        )
        assert not set(retry_jobs[route]).intersection(entry_jobs), (
            f"{route}: excluded campaign jobs also appear in retry costs"
        )
        forensic = Path(str(entry.get("forensic_audit") or ""))
        if not forensic.is_absolute():
            forensic = REPO / forensic
        assert forensic.is_file(), (
            f"excluded campaign lacks its forensic audit: {forensic}"
        )
        forensic_payload = json.loads(forensic.read_text())
        assert forensic_payload.get("curve_treatment", "").startswith("exclude")
        # Fail closed if either stranded v1 CPU controller submits more GPU
        # work after this registry was written.  Slurm step logs preserve the
        # parent controller in the inherited TMPDIR warning even across PID
        # namespaces, so the scan is independent of mutable driver bookkeeping.
        if entry.get("campaign") == "proposer_sota5_sql_clean_v1":
            controllers = {
                str(job) for job in entry.get("cpu_controller_jobs") or []
            }
            assert controllers == {"2809343", "2812928"}

            def children(pattern: str) -> set[str]:
                found: set[str] = set()
                for log_path in SLURM_LOG.glob(pattern):
                    try:
                        log_text = log_path.read_text(errors="ignore")
                    except OSError:
                        continue
                    if not any(
                        f"/tmp/yingzim/{controller}" in log_text
                        for controller in controllers
                    ):
                        continue
                    match = re.search(r"-([0-9]+)\.(?:out|err|log)$",
                                      log_path.name)
                    if match:
                        found.add(match.group(1))
                return found

            discovered = {
                "outer_jobs": children("sah-outer-*.err"),
                "train_jobs": children("wv-lora-*.log"),
                "merge_jobs": children("wv-merge-*.log"),
            }
            for key, found in discovered.items():
                registered = {str(job) for job in entry.get(key) or []}
                assert registered == found, (
                    f"SQL-v1 {key} registry is stale; registered="
                    f"{sorted(registered, key=int)}, discovered="
                    f"{sorted(found, key=int)}"
                )
        elif entry.get("campaign") == "prism_stale_phi_round1030_attempt":
            assert set(entry.get("task_ids") or []) == {PRISM_TASK}
            assert entry_jobs == {"2823642"}
            assert int(forensic_payload.get("outer_job") or 0) == 2823642
            assert forensic_payload.get("task") == PRISM_TASK
            assert forensic_payload.get("observed_stale_phi") == (
                "mphi_sota7_prism_clean_v1_08"
            )
            assert forensic_payload.get("expected_parent_phi") == (
                "mphi_sota7_prism_clean_v1_09"
            )
        excluded_records.append({
            **entry,
            "forensic_audit": str(forensic.resolve()),
            "gpu_jobs_charged": sorted(entry_jobs, key=int),
        })

    excluded_timing = {
        route: timing_ledger({
            role: sorted(set(jobs), key=int)
            for role, jobs in roles.items()
        })
        for route, roles in excluded_job_roles.items()
    }
    if not args.allow_incomplete:
        for route, ledger in excluded_timing.items():
            assert not ledger["sacct_jobs_missing"], (
                f"{route}: excluded-campaign sacct rows are missing for "
                f"{ledger['sacct_jobs_missing']}"
            )
            assert all(
                int((row.get("slurm_accounting") or {}).get(
                    "allocated_gpus_sacct", 0
                )) > 0
                for row in ledger["jobs"]
            ), f"{route}: an excluded-campaign job has no GPU allocation"
            assert ledger["sacct_snapshot"] == str(SACCT_SNAPSHOT), (
                f"{route}: strict excluded-campaign audit requires frozen sacct"
            )
            assert int(ledger["sacct_jobs_from_frozen_snapshot"]) == len(
                ledger["jobs"]
            ), f"{route}: not every excluded-campaign job is frozen in sacct"
    result["excluded_campaign_costs"] = {
        "registry": str(EXCLUDED_CAMPAIGNS),
        "entries": excluded_records,
        "timing": excluded_timing,
        "interpretation": (
            "whole inadmissible lineages contribute no score or logical method "
            "budget; all listed GPU allocations are charged as-run"
        ),
    }

    as_run = {}
    for route in ("proposer", "context", "executor"):
        accepted = result["compute_timing_proxy"][route]
        overhead = retry_timing[route]
        analysis_overhead = (
            analysis_rejection_timing["allocated_gpu_hours_sacct"]
            if route == "context" else 0.0
        )
        excluded_overhead = excluded_timing[route][
            "allocated_gpu_hours_sacct"
        ]
        as_run[route] = {
            "accepted_protocol_allocated_gpu_hours_sacct": accepted[
                "allocated_gpu_hours_sacct"
            ],
            "operational_retry_allocated_gpu_hours_sacct": overhead[
                "allocated_gpu_hours_sacct"
            ],
            "analysis_rejection_allocated_gpu_hours_sacct": (
                analysis_overhead
            ),
            "excluded_campaign_allocated_gpu_hours_sacct": (
                excluded_overhead
            ),
            "total_allocated_gpu_hours_sacct": (
                accepted["allocated_gpu_hours_sacct"]
                + overhead["allocated_gpu_hours_sacct"]
                + analysis_overhead
                + excluded_overhead
            ),
            "interpretation": (
                "accepted protocol jobs plus zero-work infrastructure retries "
                "plus analyzer-rejected attempts and whole excluded campaigns; "
                "logical curve metrics exclude every overhead class"
            ),
        }
    result["operational_retry_costs"] = {
        "registry": str(OPERATIONAL_RETRIES),
        "entries": retry_entries,
        "timing": retry_timing,
        "as_run_totals": as_run,
        "cpu_wrapper_jobs_recorded_but_not_in_gpu_hours": sorted({
            str(entry["wrapper_job"])
            for entry in retry_entries if entry.get("wrapper_job")
        }, key=int),
    }
    result["as_run_costs"] = {
        "totals_by_route": as_run,
        "includes": [
            "accepted protocol allocation",
            "zero-trajectory infrastructure/orchestration retries",
            "analysis-required retries rejected from score evidence",
            "whole executed campaigns excluded after provenance failure",
        ],
    }

    sensitivity = audit_ahc039_k32_sensitivity()
    ahc039 = result["tasks"]["eft__ahc_simpletes__ahc039"]
    sensitivity["comparison"] = {
        "historical_proposer_local_ledger_endpoint": ahc039["metrics"][
            "proposer"
        ]["endpoint_score"],
        "reported_proposer_endpoint_with_local_compute_ledger": ahc039[
            "reported_proposer_score"
        ],
        "beats_historical_proposer_local_ledger_endpoint": (
            sensitivity["endpoint_score"] >
            ahc039["metrics"]["proposer"]["endpoint_score"]
        ),
        "beats_reported_proposer_endpoint": (
            sensitivity["endpoint_score"] > ahc039["reported_proposer_score"]
        ),
    }
    result["excluded_executor_batch_size_sensitivities"] = [sensitivity]

    methods = ("proposer", "context", "executor")
    aggregate: dict[str, Any] = {
        "macro_ratio_at_task_specific_common_budget": {},
        "macro_log_auc_to_task_specific_common_budget": {},
        "macro_ratio_at_observed_endpoint": {},
        "macro_ratio_at_task_specific_common_charged_evaluator_budget": {},
        "macro_log_auc_to_task_specific_common_charged_evaluator_budget": {},
        "macro_ratio_at_task_specific_common_recorded_model_call_budget": {},
        "macro_log_auc_to_task_specific_common_recorded_model_call_budget": {},
        "human_best_crossings_at_observed_endpoint": {},
        "human_best_matches_within_reported_precision_at_observed_endpoint": {},
        "human_best_crossings_excluding_tasks_already_above_human_at_h2": {},
        "human_best_matches_excluding_tasks_already_matching_human_at_h2": {},
        "common_budget_pairwise_wins": {},
        "common_charged_evaluator_budget_pairwise_wins": {},
        "common_recorded_model_call_budget_pairwise_wins": {},
        "observed_endpoint_pairwise_wins": {},
        "three_transition_empirical_plateaus": {},
        "budget_limited_endpoints": {},
        "tasks_best_at_common_budget_including_ties": {},
        "tasks_best_at_observed_endpoint_including_ties": {},
        "reported_proposer_endpoint": {
            "human_best_crossings": 0,
            "wins_vs_context_observed_endpoint": 0,
            "wins_vs_executor_observed_endpoint": 0,
            "caveat": (
                "endpoint budgets differ; every table endpoint is locally ledgered"
            ),
        },
    }
    for method in methods:
        task_metrics = [result["tasks"][task]["metrics"][method] for task in TASKS]
        aggregate["macro_ratio_at_task_specific_common_budget"][method] = sum(
            row["ratio_at_common_budget"] for row in task_metrics) / len(TASKS)
        aggregate["macro_log_auc_to_task_specific_common_budget"][method] = sum(
            row["log_auc_to_common_budget"] for row in task_metrics) / len(TASKS)
        aggregate["macro_ratio_at_observed_endpoint"][method] = sum(
            row["endpoint_ratio"] for row in task_metrics
        ) / len(TASKS)
        eval_metrics = [
            result["tasks"][task]["evaluator_call_budget_metrics"][method]
            for task in TASKS
        ]
        aggregate[
            "macro_ratio_at_task_specific_common_charged_evaluator_budget"
        ][method] = sum(
            row["ratio_at_common_evaluator_budget"] for row in eval_metrics
        ) / len(TASKS)
        aggregate[
            "macro_log_auc_to_task_specific_common_charged_evaluator_budget"
        ][method] = sum(
            row["log_auc_to_common_evaluator_budget"] for row in eval_metrics
        ) / len(TASKS)
        model_metrics = [
            result["tasks"][task]["model_call_budget_metrics"][method]
            for task in TASKS
        ]
        aggregate[
            "macro_ratio_at_task_specific_common_recorded_model_call_budget"
        ][method] = sum(
            row["ratio_at_common_recorded_model_call_budget"]
            for row in model_metrics
        ) / len(TASKS)
        aggregate[
            "macro_log_auc_to_task_specific_common_recorded_model_call_budget"
        ][method] = sum(
            row["log_auc_to_common_recorded_model_call_budget"]
            for row in model_metrics
        ) / len(TASKS)
        aggregate["human_best_crossings_at_observed_endpoint"][method] = sum(
            row["first_human_best_crossing_x"] is not None
            for row in task_metrics
        )
        aggregate[
            "human_best_matches_within_reported_precision_at_observed_endpoint"
        ][method] = sum(
            row["first_human_best_match_within_reported_precision_x"] is not None
            for row in task_metrics
        )
        aggregate[
            "human_best_crossings_excluding_tasks_already_above_human_at_h2"
        ][method] = sum(
            result["tasks"][task]["fixed_h2"] < result["tasks"][task]["reference"]
            and result["tasks"][task]["metrics"][method][
                "first_human_best_crossing_x"] is not None
            for task in TASKS
        )
        aggregate[
            "human_best_matches_excluding_tasks_already_matching_human_at_h2"
        ][method] = sum(
            result["tasks"][task]["fixed_h2"] < (
                result["tasks"][task]["reference"]
                - result["tasks"][task]["reference_reporting_half_unit"]
            )
            and result["tasks"][task]["metrics"][method][
                "first_human_best_match_within_reported_precision_x"
            ] is not None
            for task in TASKS
        )
        aggregate["three_transition_empirical_plateaus"][method] = sum(
            row["plateau_last_three_batches"] for row in task_metrics
        )
        aggregate["budget_limited_endpoints"][method] = sum(
            not row["plateau_last_three_batches"] for row in task_metrics
        )
        aggregate["tasks_best_at_common_budget_including_ties"][method] = sum(
            method in result["tasks"][task]["common_budget_ranking"][
                "best_methods_including_ties"
            ]
            for task in TASKS
        )
        aggregate["tasks_best_at_observed_endpoint_including_ties"][method] = sum(
            method in result["tasks"][task]["observed_endpoint_ranking"][
                "best_methods_including_ties"
            ]
            for task in TASKS
        )

    for left, right in (("proposer", "context"), ("proposer", "executor"),
                        ("context", "executor")):
        key = f"{left}_over_{right}"
        aggregate["common_budget_pairwise_wins"][key] = sum(
            result["tasks"][task]["metrics"][left]["ratio_at_common_budget"]
            > result["tasks"][task]["metrics"][right]["ratio_at_common_budget"] + 1e-12
            for task in TASKS
        )
        aggregate["common_charged_evaluator_budget_pairwise_wins"][key] = sum(
            result["tasks"][task]["evaluator_call_budget_metrics"][left][
                "ratio_at_common_evaluator_budget"]
            > result["tasks"][task]["evaluator_call_budget_metrics"][right][
                "ratio_at_common_evaluator_budget"] + 1e-12
            for task in TASKS
        )
        aggregate["common_recorded_model_call_budget_pairwise_wins"][key] = sum(
            result["tasks"][task]["model_call_budget_metrics"][left][
                "ratio_at_common_recorded_model_call_budget"]
            > result["tasks"][task]["model_call_budget_metrics"][right][
                "ratio_at_common_recorded_model_call_budget"] + 1e-12
            for task in TASKS
        )
        aggregate["observed_endpoint_pairwise_wins"][key] = sum(
            result["tasks"][task]["metrics"][left]["endpoint_score"]
            > result["tasks"][task]["metrics"][right]["endpoint_score"] + 1e-12
            for task in TASKS
        )

    # The delete-one calculation remains in the machine-readable audit as a
    # historical sensitivity.  It gates claims only while the live figure uses
    # the old K=16/max-evals=30 AHC039 continuation.  Once the isolated K=8/20
    # replacement is selected, deleting AHC039 is no longer a cadence audit and
    # must not veto an otherwise valid aggregate result.
    cadence_metric_specs = {
        "common_trajectory_score_ratio": (
            "metrics", "ratio_at_common_budget"
        ),
        "common_trajectory_log_auc": (
            "metrics", "log_auc_to_common_budget"
        ),
        "common_evaluator_budget_score_ratio": (
            "evaluator_call_budget_metrics",
            "ratio_at_common_evaluator_budget",
        ),
        "common_evaluator_budget_log_auc": (
            "evaluator_call_budget_metrics",
            "log_auc_to_common_evaluator_budget",
        ),
        "common_recorded_model_call_score_ratio": (
            "model_call_budget_metrics",
            "ratio_at_common_recorded_model_call_budget",
        ),
        "common_recorded_model_call_log_auc": (
            "model_call_budget_metrics",
            "log_auc_to_common_recorded_model_call_budget",
        ),
    }

    def cadence_subset_summary(task_subset: tuple[str, ...]) -> dict[str, Any]:
        metric_rows: dict[str, Any] = {}
        for metric_name, (section, field) in cadence_metric_specs.items():
            macro = {
                method: sum(
                    float(result["tasks"][task][section][method][field])
                    for task in task_subset
                ) / len(task_subset)
                for method in methods
            }
            best = max(macro.values())
            metric_rows[metric_name] = {
                "macro": macro,
                "best_methods_including_ties": [
                    method for method, value in macro.items()
                    if abs(value - best) <= 1e-12
                ],
                "proposer_pairwise_stance": {
                    other: (
                        1 if macro["proposer"] > macro[other] + 1e-12 else
                        -1 if macro["proposer"] < macro[other] - 1e-12 else
                        0
                    )
                    for other in ("context", "executor")
                },
            }
        return {
            "tasks": list(task_subset),
            "task_count": len(task_subset),
            "metrics": metric_rows,
        }

    ahc039_task = "eft__ahc_simpletes__ahc039"
    cadence_all = cadence_subset_summary(TASKS)
    cadence_without = cadence_subset_summary(tuple(
        task for task in TASKS if task != ahc039_task
    ))
    primary_cadence_metrics = (
        "common_trajectory_score_ratio",
        "common_trajectory_log_auc",
    )
    secondary_cadence_metrics = tuple(
        metric for metric in cadence_metric_specs
        if metric not in primary_cadence_metrics
    )

    def cadence_stance_changed(metric_name: str) -> bool:
        full = cadence_all["metrics"][metric_name]
        reduced = cadence_without["metrics"][metric_name]
        return (
            full["best_methods_including_ties"] !=
            reduced["best_methods_including_ties"] or
            full["proposer_pairwise_stance"] !=
            reduced["proposer_pairwise_stance"]
        )

    delete_one_primary_changed = [
        metric for metric in primary_cadence_metrics
        if cadence_stance_changed(metric)
    ]
    delete_one_secondary_changed = [
        metric for metric in secondary_cadence_metrics
        if cadence_stance_changed(metric)
    ]
    if clean_core_route_active[AHC039_TASK]:
        primary_changed: list[str] = []
        secondary_changed: list[str] = []
        result["ahc039_proposer_cadence_delete_one_sensitivity"] = {
            "status": "not_applicable_clean_k8_max_evals20_selected",
            "mismatch": None,
            "all_seven_tasks": cadence_all,
            "without_ahc039": cadence_without,
            "delete_one_diagnostic_primary_changes": (
                delete_one_primary_changed
            ),
            "delete_one_diagnostic_secondary_changes": (
                delete_one_secondary_changed
            ),
            "primary_metrics_with_changed_order_or_pairwise_stance": [],
            "secondary_metrics_with_changed_order_or_pairwise_stance": [],
            "primary_aggregate_is_cadence_sensitive": False,
            "action_rule": (
                "none: the plotted AHC039 proposer is already an isolated "
                "K=8/max-evals=20 lineage"
            ),
            "interpretation_limit": (
                "the retained delete-one numbers are an influence diagnostic, "
                "not evidence of a cadence mismatch"
            ),
        }
    else:
        primary_changed = delete_one_primary_changed
        secondary_changed = delete_one_secondary_changed
        result["ahc039_proposer_cadence_delete_one_sensitivity"] = {
            "status": "active_live_historical_cadence_mismatch",
            "mismatch": (
                "plotted historical proposer continuation uses "
                "K=16/max_evals=30; canonical context/executor and other "
                "proposer lineages use K=8/max_evals=20"
            ),
            "all_seven_tasks": cadence_all,
            "without_ahc039": cadence_without,
            "primary_metrics_with_changed_order_or_pairwise_stance": (
                primary_changed
            ),
            "secondary_metrics_with_changed_order_or_pairwise_stance": (
                secondary_changed
            ),
            "primary_aggregate_is_cadence_sensitive": bool(primary_changed),
            "action_rule": (
                "select the fresh AHC039 K=8/max-evals=20 proposer arm before "
                "making a strong aggregate reward-routing claim"
            ),
            "interpretation_limit": (
                "delete-one sensitivity detects dependence on the mismatched "
                "task; it does not impute the missing canonical-cadence score"
            ),
        }

    all_routes_plateau_tasks = [
        task for task in TASKS
        if all(
            bool(result["tasks"][task]["metrics"][method][
                "plateau_last_three_batches"
            ])
            for method in methods
        )
    ]
    plateau_pairwise_wins = {}
    for left, right in (("proposer", "context"), ("proposer", "executor"),
                        ("context", "executor")):
        key = f"{left}_over_{right}"
        plateau_pairwise_wins[key] = sum(
            result["tasks"][task]["metrics"][left]["endpoint_score"]
            > result["tasks"][task]["metrics"][right]["endpoint_score"] + 1e-12
            for task in all_routes_plateau_tasks
        )
    aggregate["all_routes_empirical_plateau_tasks"] = all_routes_plateau_tasks
    aggregate[
        "observed_endpoint_pairwise_wins_on_all_routes_plateau_tasks"
    ] = plateau_pairwise_wins

    proposer_leads_common_score = all(
        aggregate["macro_ratio_at_task_specific_common_budget"]["proposer"]
        > aggregate["macro_ratio_at_task_specific_common_budget"][other] + 1e-12
        for other in ("context", "executor")
    )
    proposer_leads_common_auc = all(
        aggregate["macro_log_auc_to_task_specific_common_budget"]["proposer"]
        > aggregate["macro_log_auc_to_task_specific_common_budget"][other] + 1e-12
        for other in ("context", "executor")
    )
    proposer_leads_endpoint_macro = all(
        aggregate["macro_ratio_at_observed_endpoint"]["proposer"]
        > aggregate["macro_ratio_at_observed_endpoint"][other] + 1e-12
        for other in ("context", "executor")
    )
    proposer_endpoint_majority = all(
        aggregate["observed_endpoint_pairwise_wins"][
            f"proposer_over_{other}"
        ] > len(TASKS) / 2
        for other in ("context", "executor")
    )
    primary_cadence_sensitive = bool(primary_changed)
    result["claim_gate"] = {
        "status": "preliminary" if args.allow_incomplete else "final",
        "declared_task_scope": (
            "seven preselected priority/strength tasks; not an "
            "unbiased benchmark-population estimate"
        ),
        "executor_trajectory_sample_efficiency": {
            "proposer_leads_common_budget_macro_score": (
                proposer_leads_common_score
            ),
            "proposer_leads_common_budget_macro_log_auc": (
                proposer_leads_common_auc
            ),
            "ahc039_k16_delete_one_changes_primary_stance": (
                primary_cadence_sensitive
            ),
            "strong_aggregate_claim_supported": (
                proposer_leads_common_score and proposer_leads_common_auc and
                not primary_cadence_sensitive
            ),
            "required_wording": (
                "executor-trajectory sample efficiency; not equal FLOPs, "
                "model calls, training compute, or GPU time"
            ),
            "cadence_action": (
                "none; the clean AHC039 K=8/max-evals=20 route is selected"
                if clean_core_route_active[AHC039_TASK] else
                "wait for and select the clean AHC039 K=8/max-evals=20 route "
                "before a strong aggregate claim"
            ),
        },
        "observed_endpoint_strength": {
            "proposer_leads_macro_endpoint_ratio": (
                proposer_leads_endpoint_macro
            ),
            "proposer_wins_pairwise_majority_vs_both_routes": (
                proposer_endpoint_majority
            ),
            "strong_aggregate_claim_supported": (
                proposer_leads_endpoint_macro and proposer_endpoint_majority
            ),
            "all_routes_empirical_plateau_tasks": all_routes_plateau_tasks,
            "required_wording": (
                "observed plateau/cap endpoints with unequal endpoint budgets; "
                "never absolute or asymptotic limits"
            ),
        },
        "absolute_limit_claim_allowed": False,
        "fallback_if_mixed": (
            "report task-level wins/losses and each predeclared aggregate; do "
            "not replace a failed aggregate with endpoint pixel height or a "
            "post-hoc task subset"
        ),
    }

    endpoint_summary = aggregate["reported_proposer_endpoint"]
    for task in TASKS:
        row = result["tasks"][task]
        proposer_score = float(row["reported_proposer_score"])
        endpoint_summary["human_best_crossings"] += (
            proposer_score >= float(row["reference"])
        )
        endpoint_summary["wins_vs_context_observed_endpoint"] += (
            proposer_score > row["metrics"]["context"]["endpoint_score"] + 1e-12
        )
        endpoint_summary["wins_vs_executor_observed_endpoint"] += (
            proposer_score > row["metrics"]["executor"]["endpoint_score"] + 1e-12
        )
    result["aggregate"] = aggregate

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

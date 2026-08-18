#!/usr/bin/env python3
"""Fail-closed publication audit for the five-task reward-routing artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.analysis.collect import clean5_sacct as clean5_cost  # noqa: E402
RUN_ROOT = Path(
    "/lustre/fsw/portfolios/av/users/yingzim/runs/self_adapt_harness"
)
TASKS = (
    "eft__ahc_simpletes__ahc039",
    "eft__ahc_simpletes__ahc058",
    "adrs__eplb",
    "adrs__prism",
    "adrs__llm_sql",
)
ROUTES = ("proposer", "context", "executor")
SERIES = {
    "proposer": "proposer_full",
    "context": "context",
    "executor": "executor",
}
PROPOSER_WORKSPACES = {
    "eft__ahc_simpletes__ahc039": "proposer_sota5_ahc039_clean_v1",
    "eft__ahc_simpletes__ahc058": "proposer_sota7_ahc058_rewardfix_v1",
    "adrs__eplb": "proposer_sota5_eplb_clean_v1",
    "adrs__prism": "proposer_sota7_prism_clean_v1",
    "adrs__llm_sql": "proposer_sota7_sql_clean_v2",
}
CONTEXT_WORKSPACES = {
    "eft__ahc_simpletes__ahc039": "context_sota5_ahc_clean",
    "eft__ahc_simpletes__ahc058": "context_sota7_ahc058_analysis_required_v1",
    "adrs__eplb": "context_sota5_sys_guarded",
    "adrs__prism": "context_sota7_rewardfix_v1",
    "adrs__llm_sql": "context_sota5_sys_guarded",
}
EXECUTOR_DIRS = {
    "eft__ahc_simpletes__ahc039": "ttt_discover_sota5_k8_ahc039/ahc039",
    "eft__ahc_simpletes__ahc058": "ttt_discover_sota5_k8/ahc058",
    "adrs__eplb": "ttt_discover_sota5_k8/eplb",
    "adrs__prism": "ttt_discover_sota5_k8/prism",
    "adrs__llm_sql": "ttt_discover_sota5_k8/llmsql",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def at_budget(points: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    eligible = [point for point in points if int(point["x"]) <= budget]
    assert eligible
    return eligible[-1]


def log_step_auc(
    points: list[dict[str, Any]], budget: int, reference: float
) -> float:
    rows = [
        (max(1, int(point["x"])), float(point["score"]) / reference)
        for point in points if int(point["x"]) <= budget
    ]
    assert rows
    if budget <= 1:
        return rows[-1][1]
    if rows[0][0] != 1:
        rows.insert(0, (1, rows[0][1]))
    area = 0.0
    for index, (left, value) in enumerate(rows):
        right = rows[index + 1][0] if index + 1 < len(rows) else budget
        if right > left:
            area += value * (math.log(right) - math.log(left))
    return area / math.log(budget)


def audit_completion(
    view: dict[str, Any], full_audit: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in TASKS:
        task_view = view["tasks"][task]
        task_audit = full_audit["tasks"][task]
        proposer_ws = RUN_ROOT / PROPOSER_WORKSPACES[task]
        context_ws = RUN_ROOT / CONTEXT_WORKSPACES[task]
        executor_dir = RUN_ROOT / EXECUTOR_DIRS[task]
        assert task_view["proposer_workspace"] == str(proposer_ws)
        assert task_view["context_workspace"] == context_ws.name

        proposer_marker = proposer_ws / "CANONICAL_COMPLETE"
        proposer_review_path = proposer_ws / "plateau_review.json"
        assert proposer_marker.is_file() and proposer_review_path.is_file()
        proposer_review = json.loads(proposer_review_path.read_text())
        assert proposer_review.get("status") in (
            "three_transition_empirical_plateau",
            "budget_limited_at_explicit_cap",
        )
        proposer_points = task_view["series"]["proposer_full"]["points"]
        proposer_round = int(proposer_points[-1]["round"])
        assert int(proposer_review.get("completed_round") or -1) == proposer_round

        context_marker = context_ws / "CANONICAL_CONTEXT_COMPLETE"
        assert context_marker.is_file()
        context_review = task_audit["completion_reviews"]["context"]
        assert context_review.get("status") in (
            "three_transition_empirical_plateau",
            "budget_limited_at_explicit_cap",
        )
        context_round = int(task_view["series"]["context"]["points"][-1]["round"])
        assert int(context_review.get("completed_round") or -1) == context_round
        if task == "eft__ahc_simpletes__ahc058":
            assert context_round >= 1129, (
                "AHC058 context stopped before the predeclared ten-round floor"
            )

        executor_marker = executor_dir / "CANONICAL_EXECUTOR_COMPLETE"
        assert executor_marker.is_file()
        executor_review = task_audit["completion_reviews"]["executor"]
        assert executor_review.get("status") in (
            "three_transition_empirical_plateau",
            "budget_limited_at_explicit_cap",
        )
        executor_step = int(task_view["series"]["executor"]["points"][-1]["step"])
        assert int(executor_review.get("completed_update") or -1) == executor_step
        result[task] = {
            "proposer": {
                "workspace": str(proposer_ws),
                "review": str(proposer_review_path),
                "status": proposer_review["status"],
                "completed_round": proposer_round,
            },
            "context": {
                "workspace": str(context_ws),
                "status": context_review["status"],
                "completed_round": context_round,
            },
            "executor": {
                "state_dir": str(executor_dir),
                "status": executor_review["status"],
                "completed_update": executor_step,
            },
        }
    return result


def audit_lineages(full_audit: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "eft__ahc_simpletes__ahc039": (
            "ahc039_proposer_clean_lineage",
            "clean_cadence_matched_lineage_verified_complete",
        ),
        "eft__ahc_simpletes__ahc058": (
            "ahc058_proposer_rewardfix_lineage",
            "clean_terminal_attribution_lineage_verified",
        ),
        "adrs__eplb": (
            "eplb_proposer_clean_lineage",
            "clean_cadence_matched_lineage_verified_complete",
        ),
        "adrs__prism": (
            "prism_proposer_clean_lineage",
            "clean_task_local_current_guard_lineage_verified",
        ),
        "adrs__llm_sql": (
            "sql_proposer_clean_lineage",
            "clean_task_local_lineage_verified",
        ),
    }
    rows = {}
    for task, (key, status) in expected.items():
        row = full_audit[key]
        assert row.get("status") == status, f"{task}: {row.get('status')}"
        rows[task] = {"audit_key": key, "status": status}
    attribution = full_audit["all_plotted_reward_attribution"]["tasks"]
    prompts = full_audit["proposer_prompt_integrity"]
    for task in TASKS:
        for route in ("proposer", "context"):
            assert attribution[task][route]["status"] == (
                "terminal_attribution_verified"
            )
        assert prompts[task]["curated_notes_found"] == 0
        assert prompts[task]["every_plotted_round_task_isolated"] is True
        assert prompts[task]["first_batch_initial_program_verified"] is True
    assert full_audit["human_best_reference_alignment"]["status"] == (
        "all_y_equals_one_values_match_frozen_human_references"
    )
    return {
        "proposer_lineages": rows,
        "all_selected_h1_rewards_match_terminal_summaries": True,
        "all_selected_proposer_prompts_task_isolated_and_uncurated": True,
        "human_best_reference_bound_to_frozen_manifest": True,
    }


def audit_route_update_isolation(
    view: dict[str, Any], full_audit: dict[str, Any]
) -> dict[str, Any]:
    """Prove that each plotted route implements its declared update target.

    The AHC058 replacement already rejects a post-cold context round when its
    analyzer brief is missing.  Apply the same publication gate to every
    selected task so the five panels cannot silently mix analyzer/context and
    plain frozen-H1 rounds.
    """
    rows: dict[str, Any] = {}
    for task in TASKS:
        task_view = view["tasks"][task]
        context_points = [
            point
            for point in task_view["series"]["context"]["points"]
            if point.get("round") is not None
        ]
        proposer_points = [
            point
            for point in task_view["series"]["proposer_full"]["points"]
            if point.get("round") is not None
        ]
        executor_points = [
            point
            for point in task_view["series"]["executor"]["points"]
            if point.get("step") is not None
        ]
        assert context_points and proposer_points and executor_points

        context_rounds = []
        for index, point in enumerate(context_points):
            briefs = int(point.get("analyst_briefs") or 0)
            calls = int(point.get("analyzer_model_calls") or 0)
            specialists = list(point.get("analyzer_specialists") or [])
            if index == 0:
                assert briefs == 0 and calls == 0 and not specialists, (
                    f"{task}: cold context round unexpectedly used an analyzer"
                )
            else:
                assert briefs == 1, (
                    f"{task}: post-cold context round {point['round']} has "
                    f"{briefs} analyzer briefs"
                )
                assert calls == 2, (
                    f"{task}: post-cold context round {point['round']} has "
                    f"{calls} analyzer calls"
                )
                assert len(specialists) == 2 and set(specialists) == {
                    "performance", "design"
                }, (
                    f"{task}: post-cold context round {point['round']} has "
                    f"unexpected analyzer specialists {specialists}"
                )
            context_rounds.append({
                "round": int(point["round"]),
                "analyst_briefs": briefs,
                "analyzer_model_calls": calls,
                "analyzer_specialists": specialists,
            })

        for point in proposer_points:
            assert int(point.get("analyst_briefs") or 0) == 0
            assert int(point.get("analyzer_model_calls") or 0) == 0
        for point in executor_points:
            assert int(point.get("h1_model_calls") or 0) == 0
            assert int(point.get("analyst_briefs") or 0) == 0
            assert int(point.get("analyzer_model_calls") or 0) == 0

        costs = full_audit["tasks"][task]["costs"]
        proposer_cost = costs["proposer"]
        context_cost = costs["context"]
        executor_cost = costs["executor"]
        assert int(proposer_cost.get("analyzer_calls") or 0) == 0
        assert int(context_cost.get("weight_updates") or 0) == 0
        assert int(context_cost.get("planned_optimizer_boundaries") or 0) == 0
        assert int(context_cost.get("analyzer_briefs") or 0) == max(
            0, len(context_points) - 1
        )
        assert int(context_cost.get("analyzer_calls") or 0) == 2 * max(
            0, len(context_points) - 1
        )
        assert int(executor_cost.get("harness_proposals") or 0) == 0
        assert int(executor_cost.get("analyzer_calls") or 0) == 0
        rows[task] = {
            "context_rounds": context_rounds,
            "context_post_cold_rounds_all_have_two_specialists": True,
            "context_weight_updates": 0,
            "proposer_analyzer_calls": 0,
            "executor_h1_or_analyzer_calls": 0,
        }
    return {
        "status": "all_five_routes_match_declared_update_targets",
        "tasks": rows,
    }


def audit_endpoint_validation(path: Path, view_path: Path) -> dict[str, Any]:
    assert (path.parent / "CANONICAL_COMPLETE").is_file()
    payload = json.loads(path.read_text())
    assert payload.get("status") == "complete"
    assert payload.get("all_runs_valid") is True
    assert int(payload.get("requested_runs") or 0) >= 5
    cases_path = Path(payload["source_cases"]).resolve()
    cases = json.loads(cases_path.read_text())
    assert Path(cases["source_plot_data"]).resolve() == view_path.resolve()
    collection_view_sha = str(
        cases.get("source_plot_data_sha256_at_collection") or ""
    )
    assert len(collection_view_sha) == 64 and all(
        character in "0123456789abcdef" for character in collection_view_sha
    )
    assert digest(cases_path) == payload.get("source_cases_sha256")
    expected = {
        f"{task}::{route}" for task in TASKS for route in ROUTES
    }
    results = payload.get("case_results") or {}
    assert set(results) == expected
    case_rows = {
        str(row["case_id"]): row for row in cases.get("cases") or []
    }
    assert set(case_rows) == expected
    view = json.loads(view_path.read_text())
    bound_rows: dict[str, Any] = {}
    for case_id in sorted(expected):
        task, route = case_id.split("::", 1)
        case = case_rows[case_id]
        result = results[case_id]
        endpoint = float(
            view["tasks"][task]["series"][SERIES[route]]["points"][-1][
                "score"
            ]
        )
        target = float(case["reported_curve_endpoint_score"])
        assert math.isclose(endpoint, target, rel_tol=0.0, abs_tol=1e-12), (
            f"{case_id}: final plotted endpoint changed after case collection"
        )
        assert math.isclose(
            float(result["reported_curve_endpoint_score"]), target,
            rel_tol=0.0, abs_tol=1e-12,
        )
        assert result["program_sha256"] == case["program_sha256"]
        embedded = view["tasks"][task]["endpoint_revalidation"][route]
        assert isinstance(embedded, dict)
        assert embedded["program_sha256"] == case["program_sha256"]
        assert math.isclose(
            float(embedded["reported_curve_endpoint_score"]), endpoint,
            rel_tol=0.0, abs_tol=1e-12,
        )
        bound_rows[case_id] = {
            "endpoint_score": endpoint,
            "program_sha256": case["program_sha256"],
            "embedded_in_final_view": True,
        }
    outside = [
        case_id for case_id, row in results.items()
        if not row.get("reported_endpoint_inside_revalidation_range")
    ]
    assert not outside, f"online endpoints outside N>=5 range: {outside}"
    return {
        "status": "all_15_endpoints_revalidated",
        "path": str(path),
        "source_view_sha256_at_case_collection": collection_view_sha,
        "final_view_sha256": digest(view_path),
        "cases": len(results),
        "runs_per_case": int(payload["requested_runs"]),
        "all_runs_valid": True,
        "reported_endpoints_outside_observed_range": [],
        "all_final_view_endpoints_and_programs_bound_to_validation": True,
        "bindings": bound_rows,
    }


def audit_cost_snapshot(
    path: Path, full_audit_path: Path, view_path: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload.get("status") == "complete"
    assert tuple(payload.get("task_scope") or ()) == TASKS
    assert Path(payload["source_audit"]).resolve() == full_audit_path.resolve()
    assert Path(payload["source_clean5_view"]).resolve() == view_path.resolve()
    assert payload.get("source_clean5_view_sha256") == digest(view_path)
    assert not payload.get("missing_jobs")
    assert not payload.get("active_jobs")
    assert not payload.get("zero_gpu_jobs")
    zero_allocation = payload.get("zero_allocation_submissions") or {}
    current_audit = json.loads(full_audit_path.read_text())
    current_roles = clean5_cost.requested_roles(current_audit)
    assert list(payload.get("requested_jobs") or []) == list(current_roles), (
        "final audit changed the selected cost job set after sacct freeze"
    )
    rows = payload.get("rows") or {}
    for job, roles in current_roles.items():
        assert rows[job]["roles"] == roles, (
            f"{job}: final audit changed cost attribution roles after freeze"
        )
    for job, state in zero_allocation.items():
        row = rows[job]
        assert str(state).startswith("CANCELLED")
        assert int(row["allocated_gpus_sacct"]) == 0
        assert int(row["elapsed_seconds_sacct"]) == 0
        assert math.isclose(
            float(row["allocated_gpu_hours_sacct"]), 0.0, abs_tol=0.0
        )
    result: dict[str, Any] = {}
    for route in ROUTES:
        accepted_jobs = []
        overhead_jobs = []
        for job, row in rows.items():
            roles = list(row.get("roles") or [])
            accepted = any(role.startswith(f"accepted:{route}:") for role in roles)
            overhead = any(
                role.startswith(prefix)
                for role in roles
                for prefix in (
                    f"retry:{route}:",
                    f"excluded_campaign:{route}:",
                    f"superseded_campaign:{route}:",
                    f"analysis_rejection:{route}:",
                )
            )
            assert not (accepted and overhead), (
                f"{job}: accepted and overhead cost categories overlap"
            )
            if accepted:
                accepted_jobs.append(job)
            if overhead:
                overhead_jobs.append(job)
        accepted_hours = sum(
            float(rows[job]["allocated_gpu_hours_sacct"])
            for job in accepted_jobs
        )
        overhead_hours = sum(
            float(rows[job]["allocated_gpu_hours_sacct"])
            for job in overhead_jobs
        )
        result[route] = {
            "accepted_protocol_jobs": len(accepted_jobs),
            "accepted_protocol_allocated_gpu_hours_sacct": accepted_hours,
            "retry_excluded_rejected_or_superseded_jobs": len(overhead_jobs),
            "overhead_allocated_gpu_hours_sacct": overhead_hours,
            "as_run_allocated_gpu_hours_sacct": accepted_hours + overhead_hours,
        }
    return {
        "status": "final_frozen_clean5_sacct_snapshot",
        "path": str(path),
        "source_audit_snapshot_sha256": payload["source_audit_sha256"],
        "current_full_audit_sha256": digest(full_audit_path),
        "source_clean5_view_sha256": payload["source_clean5_view_sha256"],
        "final_audit_cost_job_set_matches_frozen_snapshot": True,
        "jobs": len(rows),
        "cancelled_before_allocation_submissions_retained_at_zero_cost": len(
            zero_allocation
        ),
        "routes": result,
        "interpretation": (
            "allocated GPU wall-hours, not FLOPs; shared jobs are charged once "
            "and superseded AHC039/EPLB proposer campaigns remain charged as-run"
        ),
    }


def score_summary(view: dict[str, Any]) -> dict[str, Any]:
    task_rows: dict[str, Any] = {}
    macro_score = {route: 0.0 for route in ROUTES}
    macro_auc = {route: 0.0 for route in ROUTES}
    wins = {
        "proposer_over_context_at_common_budget": 0,
        "proposer_over_executor_at_common_budget": 0,
        "proposer_over_context_at_observed_endpoint": 0,
        "proposer_over_executor_at_observed_endpoint": 0,
    }
    for task in TASKS:
        task_view = view["tasks"][task]
        reference = float(view["anchors"][task][1])
        budget = int(task_view["task_specific_common_rollout_budget"])
        common = {}
        endpoints = {}
        auc = {}
        for route in ROUTES:
            points = task_view["series"][SERIES[route]]["points"]
            point = at_budget(points, budget)
            common[route] = {
                "completed_point_x": int(point["x"]),
                "unused_budget_due_to_atomic_batching": budget - int(point["x"]),
                "score": float(point["score"]),
                "score_over_human_best": float(point["score"]) / reference,
            }
            endpoints[route] = {
                "x": int(points[-1]["x"]),
                "score": float(points[-1]["score"]),
                "score_over_human_best": float(points[-1]["score"]) / reference,
            }
            auc[route] = log_step_auc(points, budget, reference)
            macro_score[route] += common[route]["score_over_human_best"]
            macro_auc[route] += auc[route]
        for other in ("context", "executor"):
            wins[f"proposer_over_{other}_at_common_budget"] += (
                common["proposer"]["score"] > common[other]["score"] + 1e-12
            )
            wins[f"proposer_over_{other}_at_observed_endpoint"] += (
                endpoints["proposer"]["score"]
                > endpoints[other]["score"] + 1e-12
            )
        task_rows[task] = {
            "human_best_combined_score": reference,
            "common_trajectory_budget": budget,
            "common_budget": common,
            "log_auc_to_common_budget": auc,
            "observed_plateau_or_cap_endpoint": endpoints,
        }
    for values in (macro_score, macro_auc):
        for route in ROUTES:
            values[route] /= len(TASKS)
    return {
        "tasks": task_rows,
        "five_task_macro_score_over_human_at_common_budget": macro_score,
        "five_task_macro_log_auc_to_common_budget": macro_auc,
        "task_level_wins": wins,
        "proposer_leads_both_predeclared_common_budget_macros": (
            all(macro_score["proposer"] > macro_score[other] + 1e-12
                for other in ("context", "executor"))
            and all(macro_auc["proposer"] > macro_auc[other] + 1e-12
                    for other in ("context", "executor"))
        ),
        "claim_scope": (
            "executor-trajectory sample efficiency at task-specific common "
            "complete-batch budgets; not equal FLOPs, GPU time, or model calls"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view", default="papers/figures/score_compute_curves_clean5_final_data.json"
    )
    parser.add_argument(
        "--full-audit", default="results/score_compute_curves_sota7_final_audit.json"
    )
    parser.add_argument(
        "--view-audit",
        default="results/score_compute_curves_clean5_final_view_audit.json",
    )
    parser.add_argument(
        "--endpoint-validation",
        default=str(RUN_ROOT / "clean5_endpoint_validation_final/results.json"),
    )
    parser.add_argument(
        "--sacct", default="results/score_compute_curves_clean5_sacct_snapshot.json"
    )
    parser.add_argument(
        "--out", default="results/score_compute_curves_clean5_publishable_audit.json"
    )
    parser.add_argument("--allow-reporting-mismatch", action="store_true")
    args = parser.parse_args()
    paths = {
        "view": Path(args.view).resolve(),
        "full_audit": Path(args.full_audit).resolve(),
        "view_audit": Path(args.view_audit).resolve(),
        "endpoint_validation": Path(args.endpoint_validation).resolve(),
        "sacct": Path(args.sacct).resolve(),
    }
    view = json.loads(paths["view"].read_text())
    full_audit = json.loads(paths["full_audit"].read_text())
    view_audit = json.loads(paths["view_audit"].read_text())
    assert view.get("layout") == "1x5"
    assert tuple((view.get("tasks") or {}).keys()) == TASKS
    assert (view.get("reference_standard") or {}).get("name") == "Best Human"
    assert view.get("y") == "combined_score divided by task's Best Human value"
    assert Path(full_audit["source"]).name.startswith("score_compute_curves_sota7")
    assert view_audit.get("status") in (
        "exact_subview_and_reported_conditions_aligned",
        "exact_subview_but_reported_conditions_diverge",
    )
    mismatches = list((view_audit.get("reported_condition_alignment") or {}).get(
        "mismatches"
    ) or [])
    if not args.allow_reporting_mismatch:
        assert not mismatches, f"reported conditions diverge: {mismatches}"

    result = {
        "schema": 1,
        "status": (
            "ready_for_reporting_sync" if mismatches
            else "publishable_clean5_snapshot"
        ),
        "task_scope": list(TASKS),
        "artifacts": {
            key: {"path": str(path), "sha256": digest(path)}
            for key, path in paths.items()
        },
        "reference_standard": view["reference_standard"],
        "reported_condition_mismatches": mismatches,
        "completion": audit_completion(view, full_audit),
        "lineage_and_reward_attribution": audit_lineages(full_audit),
        "route_update_isolation": audit_route_update_isolation(
            view, full_audit
        ),
        "endpoint_revalidation": audit_endpoint_validation(
            paths["endpoint_validation"], paths["view"]
        ),
        "cost_accounting": audit_cost_snapshot(
            paths["sacct"], paths["full_audit"], paths["view"]
        ),
        "score_readout": score_summary(view),
        "claim_boundaries": {
            "executor_route_ownership": "reference baseline; not ours",
            "executor_reference_name": (
                "local budget-scaled TTT-Discover-style executor adaptation"
            ),
            "absolute_or_asymptotic_limit_claim_allowed": False,
            "equal_total_compute_claim_allowed": False,
            "campaign_level_significance_claim_allowed": False,
            "endpoint_term": "observed empirical plateau/cap endpoint",
        },
    }
    atomic_write(Path(args.out).resolve(), result)
    print(f"wrote {args.out}: status={result['status']}")


if __name__ == "__main__":
    main()

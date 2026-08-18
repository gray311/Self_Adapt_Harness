#!/usr/bin/env python3
"""Plot the canonical four-task inference-16 reward-routing comparison.

This script intentionally refuses partial or historical inputs.  Every method
must contribute the shared x=1 anchor followed by nineteen complete batches at
x=1+16*r, ending at x=305.  Analyzer requests are reported in the cost ledger
but are not agent trajectories and therefore are not on this axis. There is no
post-submit model reviewer in canonical runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "reward-route-inference16-v1"
ROUNDS = 19
PER_ROUND = 16
FINAL_X = 305
TASKS = (
    ("erdos", "eft__math__erdos_min_overlap", "Erdős min-overlap", 2000, 3000),
    ("ac2", "eft__math__second_autocorr_ineq", "Autocorrelation II", 2100, 3100),
    ("hadamard", "eft__math__hadamard_maximal_det", "Hadamard max-det", 2200, 3200),
    ("eplb", "adrs__eplb", "EPLB", 2300, 3300),
)
SERIES = (
    ("proposer", "Update proposer weights (ours)", "#1769aa", "o", "-"),
    ("context", "Analyzer/context only", "#6f6f6f", "^", "-."),
    ("executor", "Update executor weights (TTT-style reference)", "#e67e22", "s", "--"),
)


def expected_x(index: int) -> int:
    return 1 + (index + 1) * PER_ROUND


def load_h1_route(
    run_root: Path, method: str, tag: str, task: str, round_base: int, anchor: float
) -> list[dict[str, Any]]:
    outer = run_root / "self_adapt_harness" / f"outer-{PROTOCOL}-{method}-{tag}"
    points: list[dict[str, Any]] = [{
        "x": 1,
        "score": anchor,
        "source": str(REPO / "results" / "baseline_h2_20ev.json"),
        "kind": "shared_fixed_h2_anchor",
    }]
    for index in range(ROUNDS):
        round_dir = outer / f"round{round_base + index:03d}"
        summary_path = round_dir / "round_summary.json"
        bases_path = round_dir / "next_bases.json"
        slot_path = round_dir / "h2_slot_plan.json"
        for path in (summary_path, bases_path, slot_path):
            if not path.is_file():
                raise FileNotFoundError(f"incomplete {method}/{tag}: missing {path}")
        summary = json.loads(summary_path.read_text())
        round_meta = json.loads((round_dir / "round.json").read_text())
        assert round_meta.get("program_ratchet_mode") == "strict_single"
        ratchet_audit = json.loads((round_dir / "program_ratchet_audit.json").read_text())
        assert ratchet_audit.get("mode") == "strict_single"
        budget = summary.get("inference_trajectory_budget") or {}
        slot_plan = json.loads(slot_path.read_text())
        x = expected_x(index)
        assert budget.get("axis_unit") == "generated_agent_trajectory"
        assert budget.get("fixed_h1_plus_h2_slots") is True
        assert int(budget.get("h1_slots_per_task")) == 8
        assert int(budget.get("h2_slots_per_task")) == 8
        assert int(budget.get("nominal_total_slots_per_task")) == 16
        assert int(budget.get("logical_round_index")) == index
        assert int(budget.get("axis_x_after_round")) == x
        assert slot_plan.get("fixed_h2_slots") is True
        assert len(slot_plan.get("slots") or []) == 8
        score = float(json.loads(bases_path.read_text())[task]["score"])
        points.append({
            "x": x,
            "score": score,
            "round": round_base + index,
            "logical_round_index": index,
            "h1_trajectories": 8,
            "h2_trajectories": 8,
            "incumbent_fallbacks": sum(
                row["h2_slot_mode"] == "incumbent_fallback"
                for row in slot_plan["slots"]
            ),
            "source": str(summary_path),
        })
    assert points[-1]["x"] == FINAL_X
    return points


def load_executor_route(
    run_root: Path, tag: str, anchor: float
) -> list[dict[str, Any]]:
    curve_path = (
        run_root / "self_adapt_harness" / PROTOCOL / "executor" / tag / "curve.jsonl"
    )
    if not curve_path.is_file():
        raise FileNotFoundError(f"incomplete executor/{tag}: missing {curve_path}")
    rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
    assert len(rows) == ROUNDS, f"executor/{tag}: expected {ROUNDS} batches, got {len(rows)}"
    points: list[dict[str, Any]] = [{
        "x": 1,
        "score": anchor,
        "source": str(REPO / "results" / "baseline_h2_20ev.json"),
        "kind": "shared_fixed_h2_anchor",
    }]
    for index, row in enumerate(rows):
        x = expected_x(index)
        assert int(row["step"]) == index
        assert int(row["launched"]) == 16
        assert row.get("trajectory_axis_unit") == "generated_agent_trajectory"
        assert int(row["cum_inference_trajectories"]) == x
        points.append({
            "x": x,
            "score": float(row["best"]),
            "step": index,
            "h1_trajectories": 0,
            "h2_trajectories": 16,
            "usable": int(row["usable"]),
            "source": str(curve_path),
        })
    assert points[-1]["x"] == FINAL_X
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=os.environ.get(
            "RUN_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/runs"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(REPO / "papers" / "figures" / "reward_route_inference16_1x4"),
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    run_root = Path(args.run_root)
    baseline = json.loads((REPO / "results" / "baseline_h2_20ev.json").read_text())[
        "baseline"
    ]
    human = json.loads((REPO / "results" / "human_best_references.json").read_text())[
        "tasks"
    ]
    output: dict[str, Any] = {
        "schema": 1,
        "status": "complete",
        "protocol": PROTOCOL,
        "trajectory_axis": "cumulative generated agent trajectories (H1 + H2)",
        "analyzer_calls_excluded_from_axis": True,
        "post_submit_reviewer_model_calls": 0,
        "rounds": ROUNDS,
        "trajectories_per_round": PER_ROUND,
        "shared_anchor_x": 1,
        "common_final_x": FINAL_X,
        "normalization": "combined_score / human_best_combined_score",
        "tasks": {},
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.78, wspace=0.25)
    for axis, (tag, task, title, prop_base, ctx_base) in zip(axes, TASKS):
        anchor = float(baseline[task]["h2_best"])
        routes = {
            "proposer": load_h1_route(run_root, "proposer", tag, task, prop_base, anchor),
            "context": load_h1_route(run_root, "context", tag, task, ctx_base, anchor),
            "executor": load_executor_route(run_root, tag, anchor),
        }
        human_score = float(human[task]["human_best_combined_score"])
        normalized_values = []
        for key, label, color, marker, style in SERIES:
            points = routes[key]
            xs = [int(row["x"]) for row in points]
            ys = [float(row["score"]) / human_score for row in points]
            normalized_values.extend(ys)
            assert xs == [1] + [expected_x(index) for index in range(ROUNDS)]
            axis.plot(xs, ys, label=label, color=color, marker=marker,
                      linestyle=style, linewidth=2, markersize=4)
        low, high = min(normalized_values), max(normalized_values)
        span = high - low
        pad = max(span * 0.12, max(abs(low), abs(high), 1.0) * 0.004)
        if span < 1e-12:
            pad = max(abs(low) * 0.02, 0.01)
        view_low, view_high = low - pad, high + pad
        human_visible = (
            view_low <= 1.0 <= view_high
            or abs(1.0 - 0.5 * (low + high)) <= 1.5 * (view_high - view_low)
        )
        if human_visible:
            view_low = min(view_low, 1.0 - 0.2 * pad)
            view_high = max(view_high, 1.0 + 0.2 * pad)
            axis.axhline(1.0, color="#444444", linestyle=":", linewidth=1.2)
        elif 1.0 > view_high:
            axis.text(0.98, 0.97, "↑ human best = 1.0", ha="right", va="top",
                      transform=axis.transAxes, fontsize=8, color="#444444")
        else:
            axis.text(0.98, 0.03, "↓ human best = 1.0", ha="right", va="bottom",
                      transform=axis.transAxes, fontsize=8, color="#444444")
        axis.set_ylim(view_low, view_high)
        axis.set_xscale("log")
        axis.set_xlim(0.9, FINAL_X * 1.08)
        axis.set_title(title)
        axis.set_xlabel("generated inference trajectories (H1 + H2, log)")
        axis.grid(alpha=0.2)
        output["tasks"][task] = {
            "tag": tag,
            "human_best_combined_score": human_score,
            "y_view": {
                "low": view_low,
                "high": view_high,
                "task_specific_zoom": True,
                "human_best_line_visible": human_visible,
            },
            "series": routes,
        }
    axes[0].set_ylabel("best validated score / human best")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.98))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    prefix.with_name(prefix.name + "_data.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot in-task campaign gains together with zero-shot cross-task transfer.

The two cell types intentionally answer different questions:

* diagonal: direction-corrected improvement of the complete in-task Weight
  condition over Initial;
* off diagonal: zero-shot Best@6 change of a source-task proposer adapter over
  the untrained proposer on the same target.

Keeping the distinction explicit prevents the full campaign improvement from
being mistaken for a weights-only zero-shot intervention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
TASK_TO_SOURCE = {
    "eft__math__erdos_min_overlap": "mphi_f_erdosmin_09",
    "eft__math__first_autocorr_ineq": "mphi_f_firstaut_11",
    "eft__math__second_autocorr_ineq": "mphi_f_secondau_10",
    "eft__math__circle_packing": "mphi_f_circlepa_03",
    "eft__math__hadamard_maximal_det": "mphi_f_hadamard_10",
    "eft__ahc_simpletes__ahc039": "mphi_f_ahc039_07",
    "eft__ahc_simpletes__ahc058": "mphi_f_ahc058_07",
    "adrs__eplb": "mphi_f_eplb_04",
    "adrs__prism": "mphi_f_prism_05",
    "adrs__llm_sql": "mphi_f_llmsql_05",
    "adrs__txn_scheduling": "mphi_f_txnsched_03",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in matrix
    ]


def in_task_gain(row: dict[str, Any]) -> float:
    initial = float(row["initial"])
    weight = float(row["weight"])
    if initial == 0.0:
        raise ValueError("campaign Initial score must be non-zero")
    if row["direction"] == "higher_is_better":
        return 100.0 * (weight - initial) / abs(initial)
    if row["direction"] == "lower_is_better":
        return 100.0 * (initial - weight) / abs(initial)
    raise ValueError(f"unknown direction: {row['direction']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=os.environ.get("RUN_ROOT")
        or "/lustre/fsw/portfolios/av/users/yingzim/runs",
    )
    parser.add_argument(
        "--gain-data", default="results/cross_task_in_task_gain.json"
    )
    parser.add_argument(
        "--out-prefix", default="papers/figures/cross_task_transfer"
    )
    parser.add_argument("--color-limit", type=float, default=20.0)
    parser.add_argument(
        "--color-softness",
        type=float,
        default=0.25,
        help="signed-log soft scale in percentage points",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    outer = run_root / "self_adapt_harness" / "outer"
    workspace = run_root / "self_adapt_harness" / "cross_task"
    gain_path = (REPO / args.gain_data).resolve()
    out_prefix = (REPO / args.out_prefix).resolve()
    gain_data = json.loads(gain_path.read_text())
    task_rows: dict[str, dict[str, Any]] = gain_data["tasks"]
    tasks = list(task_rows)
    sources = [TASK_TO_SOURCE[task] for task in tasks]

    merged: dict[str, dict[str, float]] = {}
    source_rounds: dict[str, list[int]] = {}
    input_rows: list[dict[str, Any]] = []
    row_files: list[Path] = []
    for name in ("rows.txt", "rows2.txt"):
        path = workspace / name
        if not path.exists():
            continue
        row_files.append(path)
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            source, round_text, job = parts[:3]
            round_id = int(round_text)
            summary = outer / f"round{round_id:03d}" / "round_summary.json"
            row_record = {
                "source": source,
                "round": round_id,
                "job": job,
                "summary": str(summary),
                "summary_exists": summary.exists(),
            }
            input_rows.append(row_record)
            if not summary.exists():
                continue
            groups = json.loads(summary.read_text()).get("groups", {})
            source_rounds.setdefault(source, []).append(round_id)
            for task, result in groups.items():
                score = result.get("best_score")
                if score is None:
                    continue
                old = merged.setdefault(source, {}).get(task)
                merged[source][task] = (
                    float(score) if old is None else max(old, float(score))
                )

    base = merged.get("BASE", {})
    zero_shot = np.full((len(tasks), len(tasks)), np.nan)
    for i, source in enumerate(sources):
        for j, task in enumerate(tasks):
            value = merged.get(source, {}).get(task)
            baseline = base.get(task)
            # A recorded zero represents a failed rollout in this campaign.
            if value in (None, 0) or baseline in (None, 0):
                continue
            zero_shot[i, j] = 100.0 * (value - baseline) / abs(baseline)

    campaign_diagonal = np.array(
        [in_task_gain(task_rows[task]) for task in tasks], dtype=float
    )
    displayed = zero_shot.copy()
    for index, value in enumerate(campaign_diagonal):
        displayed[index, index] = value

    # The untrained-proposer Best@6 baseline for ahc058 is approximately zero,
    # so its off-diagonal ratios are not meaningful.  The diagonal is well
    # defined because it uses the non-zero campaign Initial score.
    ahc058_index = tasks.index("eft__ahc_simpletes__ahc058")
    colored = displayed.copy()
    for index in range(len(tasks)):
        if index != ahc058_index:
            colored[index, ahc058_index] = np.nan

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch

    limit = float(args.color_limit)
    if limit <= 0:
        raise ValueError("--color-limit must be positive")
    softness = float(args.color_softness)
    if softness <= 0:
        raise ValueError("--color-softness must be positive")

    # A linear ±20% map makes scientifically real sub-1% gains (AC1, a039,
    # EPLB) indistinguishable from white.  Apply one monotone signed-log color
    # transform to every cell while leaving the annotated values untouched.
    # The soft scale prevents tiny numerical noise around zero from receiving
    # a saturated color.
    def color_transform(values: np.ndarray | float) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        clipped = np.clip(array, -limit, limit)
        return np.sign(clipped) * np.log1p(np.abs(clipped) / softness)

    transformed = color_transform(colored)
    transformed_limit = float(np.log1p(limit / softness))
    norm = TwoSlopeNorm(
        vmin=-transformed_limit, vcenter=0.0, vmax=transformed_limit
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(11.4, 8.5), constrained_layout=True)
    image = ax.imshow(transformed, cmap=cmap, norm=norm, aspect="equal")
    short = [task_rows[task]["short"] for task in tasks]
    ax.set_xticks(range(len(tasks)), short, fontsize=10.5)
    ax.set_yticks(range(len(tasks)), short, fontsize=10.5)
    ax.set_xlabel("target task  $\\tau_j$", fontsize=12)
    ax.set_ylabel("source-task proposer  $\\phi_i$", fontsize=12)
    ax.set_title(
        "In-task adaptation gains vs. zero-shot cross-task transfer\n"
        "diagonal: full Weight vs. Initial  |  off-diagonal: Best@6 vs. untrained proposer",
        fontsize=13.5,
        pad=12,
    )

    for i in range(len(tasks)):
        for j in range(len(tasks)):
            is_diagonal = i == j
            is_ahc058_offdiag = j == ahc058_index and not is_diagonal
            value = displayed[i, j]
            if is_ahc058_offdiag:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#eeeeee",
                        edgecolor="none",
                        zorder=1,
                    )
                )
            if is_ahc058_offdiag:
                label = "n/a"
                text_color = "#777777"
                weight = "normal"
            elif not np.isfinite(value):
                label = "—"
                text_color = "#777777"
                weight = "normal"
            else:
                label = f"{value:+.1f}"
                transformed_value = float(color_transform(value))
                text_color = (
                    "white"
                    if abs(transformed_value) >= 0.62 * transformed_limit
                    else "black"
                )
                weight = "bold" if is_diagonal else "normal"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.4 if is_diagonal else 7.8,
                color=text_color,
                fontweight=weight,
                zorder=3,
            )
            if is_diagonal:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                        zorder=4,
                    )
                )

    colorbar = fig.colorbar(image, ax=ax, extend="both", shrink=0.88, pad=0.035)
    raw_colorbar_ticks = np.array(
        [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]
    )
    raw_colorbar_ticks = raw_colorbar_ticks[
        np.abs(raw_colorbar_ticks) <= limit
    ]
    colorbar.set_ticks(color_transform(raw_colorbar_ticks))
    colorbar.set_ticklabels([f"{value:g}" for value in raw_colorbar_ticks])
    colorbar.set_label(
        f"improvement (%) — signed-log color, clipped at ±{limit:g}",
        fontsize=11,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=2.0,
                label="full in-task Weight vs. Initial",
            ),
            Patch(
                facecolor="#eeeeee",
                edgecolor="none",
                label="ahc058 off-diagonal ratio undefined",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(out_prefix.with_suffix(".pdf"))

    diagonal_rows = {}
    for index, task in enumerate(tasks):
        row = task_rows[task]
        diagonal_rows[task] = {
            "source_adapter": sources[index],
            "initial": float(row["initial"]),
            "weight": float(row["weight"]),
            "direction": row["direction"],
            "improvement_pct": float(campaign_diagonal[index]),
            "weights_only_zero_shot_pct": (
                None
                if not np.isfinite(zero_shot[index, index])
                else float(zero_shot[index, index])
            ),
        }
    manifest = {
        "schema": 1,
        "cell_semantics": {
            "diagonal": gain_data["definition"],
            "diagonal_causal_scope": gain_data["causal_scope"],
            "off_diagonal": "Zero-shot Best@6 percentage change of the source adapter relative to the untrained proposer on the same target.",
        },
        "task_order": tasks,
        "short_labels": short,
        "color_limit_pct": limit,
        "color_scale": {
            "type": "signed_log1p",
            "formula": "sign(x) * log1p(abs(clip(x, -limit, limit)) / softness)",
            "softness_pct": softness,
            "annotations_use_untransformed_values": True,
        },
        "ahc058_off_diagonal_color_suppressed": True,
        "campaign_diagonal": diagonal_rows,
        "displayed_matrix_pct": json_matrix(displayed),
        "weights_only_zero_shot_matrix_pct": json_matrix(zero_shot),
        "source_rounds": source_rounds,
        "inputs": {
            "campaign_gain": {
                "path": str(gain_path),
                "sha256": digest(gain_path),
            },
            "row_files": [
                {"path": str(path), "sha256": digest(path)}
                for path in row_files
            ],
            "rows": input_rows,
        },
    }
    data_path = out_prefix.with_name(out_prefix.name + "_data.json")
    data_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(out_prefix.with_suffix(".png"))
    print(out_prefix.with_suffix(".pdf"))
    print(data_path)
    print("campaign diagonal:")
    for task, value in zip(short, campaign_diagonal):
        print(f"  {task:5s} {value:+.1f}%")


if __name__ == "__main__":
    main()

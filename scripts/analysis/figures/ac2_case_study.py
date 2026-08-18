#!/usr/bin/env python3
"""AC2 case-study figure: three reward routes with audited event nodes.

Data: papers/figures/reward_route_requested5_latest_data.json
(displayed_common_budget_series, eft__math__second_autocorr_ineq).
y = gap to human best in percent.  Layout: chart left, numbered node
panel right; markers on the curves carry the numbers.
"""
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("papers/figures/reward_route_requested5_latest_data.json"))
T = D["tasks"]["eft__math__second_autocorr_ineq"]
HB = T["human_best_combined_score"]
B  = T["common_trajectory_budget"]
S  = T["displayed_common_budget_series"]

def gap(s): return (s / HB - 1.0) * 100.0

ROUTES = [
    ("proposer_full", "Update proposer weights (ours)",
     dict(color="#084594", ls="-",  lw=2.4, marker="o", ms=4.5, zorder=9)),
    ("context", "Analyzer context (weights frozen)",
     dict(color="#8a8a8a", ls="-.", lw=1.9, marker="^", ms=4.5, zorder=7)),
    ("executor", "Update executor weights (fixed H2)",
     dict(color="#e07b28", ls="--", lw=1.9, marker="s", ms=4.5, zorder=8)),
]
STYLE = {k: st for k, _, st in ROUTES}

NODES = [
    ("1", 6, "proposer_full",
     "Multi-start initialization:\nproposer edits system prompt + skill,\nwidening the search space",
     (1.8, 1.05)),
    ("2", 8, "context",
     "The harness adds a new tool and advice\nbut no probing discipline: the executor\n"
     "wastes the entire budget climbing a\nsingle seed and ends below the start",
     (9.4, -1.55)),
    ("4", 14, "context",
     "The jump is large only because the\nprevious round fell.  Beyond that recovery\n"
     "the gain is small, and comes from ordinary\nprogram tuning, not the tool added in\n"
     "response to feedback: feedback locates\nproblems, it does not improve the proposer",
     (16.6, -0.20)),
    ("7", 31, "proposer_full",
     "Breakthrough: the trained proposer\nships a step-config generator plus\n"
     "a probing-budget rule, and the\nexecutor follows the workflow --\n"
     "screen with cheap probes, full\nevaluations only for finalists --\nlanding a new best",
     (27.8, 1.38)),
    ("8", 33, "executor",
     "The executor route's one real gain:\na local rewrite of the optimizer\n"
     "inside the fixed harness -- no\nprobing, no new interface.  Executor\n"
     "updates can improve the program,\nnever the process",
     (26.4, -1.5)),
    ("3", 13, "proposer_full",
     "Forced diversity: when a function family\nstalls repeatedly, switch to a new one.\n"
     "Inherited as skills and enforced as code,\nthis is machinery an executor update\ncannot author",
     (14.2, 1.22)),
]

def render(active, outputs):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for key, label, st in ROUTES:
        pts = S[key]
        ax.step([p["x"] for p in pts], [gap(p["score"]) for p in pts],
                where="post", label=label, **st)
    ax.axhline(0.0, color="#555555", lw=1.0, ls=":", zorder=2)
    ax.text(1.2, 0.06, "human best", fontsize=8.5, color="#555555",
            va="bottom")
    ax.axvline(B, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.text(B + 0.55, -2.62, f"common budget $B{{=}}{B}$", fontsize=8,
            color="#888888", rotation=90, ha="left", va="bottom")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gap to human best (%)")
    ax.set_xlim(0.5, B + 1.5)
    ax.grid(color="#e8e8e8", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, -0.005), fontsize=8.8, frameon=False,
               handlelength=2.0, ncols=3, columnspacing=1.3)

    for num, x, route, text, txy in [n for n in NODES if n[0] in active]:
        pts = S[route]
        exact = [p for p in pts if p["x"] == x]
        y = gap(exact[0]["score"]) if exact else \
            gap(max(p["score"] for p in pts if p["x"] <= x))
        c = STYLE[route]["color"]
        ax.plot(x, y, marker="o", ms=12.5, mfc="white", mec=c, mew=1.8,
                zorder=11)
        ax.text(x, y, num, ha="center", va="center", fontsize=8, color=c,
                weight="bold", zorder=12)
        ax.annotate(text, xy=(x, y), xytext=txy, fontsize=8.0,
                    color="#333333", ha="left", va="center", zorder=10,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.0,
                                    shrinkA=2, shrinkB=8,
                                    connectionstyle="arc3,rad=0.15"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c,
                              lw=0.9, alpha=1.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.17)
    for out in outputs:
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", outputs[0].rsplit("/", 1)[-1])

render({"1"}, ("papers/figures/ac2_case_study_node1.pdf",
               "papers/figures/ac2_case_study_node1.png"))
render({"1", "2"}, ("papers/figures/ac2_case_study_node2.pdf",
                    "papers/figures/ac2_case_study_node2.png"))
render({"1", "2", "3"}, ("papers/figures/ac2_case_study_node3.pdf",
                         "papers/figures/ac2_case_study_node3.png"))
render({"1", "2", "3", "4"}, ("papers/figures/ac2_case_study_node4.pdf",
                              "papers/figures/ac2_case_study_node4.png"))
render({"1", "2", "3", "4", "7"}, ("papers/figures/ac2_case_study_node7.pdf",
                                   "papers/figures/ac2_case_study_node7.png"))
render({"1", "2", "3", "4", "7", "8"}, ("papers/figures/ac2_case_study.pdf",
                                        "papers/figures/ac2_case_study.png"))

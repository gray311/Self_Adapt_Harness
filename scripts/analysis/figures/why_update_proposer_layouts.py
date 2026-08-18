#!/usr/bin/env python3
"""Render discussion drafts for the update-executor/update-proposer figure.

These are intentionally layout studies, not a final camera-ready figure.  Each
draft uses the same evidence boundaries:

* TTT-style adaptation routes reward into executor weights.
* SAH/NexAU routes reward into a proposer that emits an explicit harness while
  the executor remains frozen.
* The four proposed advantages are visualized as mechanisms: within-rollout
  exploration control, artifact-level attribution, fail-closed integrity
  gates, and cross-model artifact transfer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


NAVY = "#172033"
INK = "#2C3443"
MUTED = "#687386"
LIGHT_LINE = "#CBD5E1"
TOP_FILL = "#E8EEF2"
TOP_EDGE = "#8194A3"
OURS_FILL = "#FFF5EE"
OURS_EDGE = "#EB6834"
BLUE = "#2A78D6"
BLUE_FILL = "#EEF6FF"
PURPLE = "#7E57C2"
PURPLE_FILL = "#F4F0FF"
GREEN = "#16803A"
GREEN_FILL = "#EDF9F0"
RED = "#B43A36"
RED_FILL = "#FFF1F0"
GOLD = "#AA721A"
GOLD_FILL = "#FFF8E8"
WHITE = "#FFFFFF"
BG = "#FCFDFE"


def canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=False)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fc: str = WHITE,
    ec: str = LIGHT_LINE,
    lw: float = 1.2,
    radius: float = 0.12,
    fontsize: float = 9.5,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    ls: str = "solid",
    zorder: int = 2,
    pad: float = 0.12,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.18
        ax.text(
            tx,
            y + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            weight=weight,
            linespacing=1.2,
            zorder=zorder + 1,
        )
    return patch


def pill(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    text: str,
    *,
    fc: str,
    ec: str,
    color: str,
    fontsize: float = 9,
) -> None:
    box(
        ax,
        x,
        y,
        w,
        0.34,
        text,
        fc=fc,
        ec=ec,
        lw=1.0,
        radius=0.17,
        fontsize=fontsize,
        color=color,
        weight="bold",
        pad=0.04,
    )


def arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = MUTED,
    lw: float = 1.5,
    style: str = "-|>",
    mutation: float = 13,
    connection: str = "arc3,rad=0",
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def badge(ax: plt.Axes, x: float, y: float, number: int, color: str) -> None:
    ax.add_patch(Circle((x, y), 0.16, facecolor=color, edgecolor=WHITE, linewidth=1.0, zorder=8))
    ax.text(x, y - 0.005, str(number), ha="center", va="center", color=WHITE, weight="bold", fontsize=8.5, zorder=9)


def dashed_group(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, *, ec: str = MUTED) -> None:
    box(ax, x, y, w, h, fc="none", ec=ec, lw=1.1, radius=0.13, ls=(0, (4, 3)), pad=0.03)
    ax.text(x + 0.12, y + h + 0.08, label, ha="left", va="bottom", fontsize=8.3, color=MUTED, weight="bold")


def title(ax: plt.Axes, heading: str, subtitle: str) -> None:
    ax.text(0.55, 8.55, heading, ha="left", va="center", fontsize=20, color=NAVY, weight="bold")
    ax.text(0.57, 8.22, subtitle, ha="left", va="center", fontsize=10.2, color=MUTED)


def harness_stack(ax: plt.Axes, x: float, y: float, w: float = 1.7, scale: float = 1.0) -> None:
    rows = [
        ("PROMPT", BLUE_FILL, BLUE),
        ("TOOL", PURPLE_FILL, PURPLE),
        ("SKILL", GREEN_FILL, GREEN),
        ("MIDDLEWARE", GOLD_FILL, GOLD),
    ]
    rh = 0.28 * scale
    gap = 0.06 * scale
    for index, (text, fc, ec) in enumerate(rows):
        yy = y + (len(rows) - 1 - index) * (rh + gap)
        box(ax, x, yy, w, rh, text, fc=fc, ec=ec, lw=0.9, radius=0.06, fontsize=7.4 * scale, color=ec, weight="bold", pad=0.02)


def strategy_strip(ax: plt.Axes, x: float, y: float, w: float, *, compact: bool = False) -> None:
    labels = ["A", "B", "C"]
    cell_w = (w - 0.34) / 3
    for index, label_text in enumerate(labels):
        xx = x + index * (cell_w + 0.17)
        box(
            ax,
            xx,
            y,
            cell_w,
            0.42 if compact else 0.5,
            f"strategy {label_text}" if not compact else label_text,
            fc=BLUE_FILL,
            ec=BLUE,
            lw=0.9,
            radius=0.06,
            fontsize=7 if compact else 7.5,
            color=BLUE,
            weight="bold",
            pad=0.02,
        )
        if index < 2:
            arrow(ax, xx + cell_w + 0.02, y + (0.21 if compact else 0.25), xx + cell_w + 0.14, y + (0.21 if compact else 0.25), color=BLUE, lw=1.0, mutation=9)


def model_fanout(ax: plt.Axes, x: float, y: float, *, width: float = 1.35) -> None:
    models = ["GLM", "Qwen", "GPT", "Claude"]
    for index, model in enumerate(models):
        yy = y + (3 - index) * 0.42
        box(ax, x, yy, width, 0.31, model, fc=BLUE_FILL, ec=BLUE, lw=0.8, radius=0.06, fontsize=7.3, color=BLUE, weight="bold", pad=0.02)


def layout_a(out: Path) -> None:
    """Reference-like mirrored loops with four callouts embedded in ours."""
    fig, ax = canvas()
    title(
        ax,
        "Draft A · Mirrored discovery loops",
        "Closest to the reference: wide executor-TTT rollouts above, learned harness control below.",
    )

    # Top: update executor.
    ax.add_patch(Rectangle((0.35, 4.72), 15.3, 3.22, facecolor="#F7F9FA", edgecolor="none", zorder=0))
    pill(ax, 0.58, 7.48, 3.15, "TTT-DISCOVER · UPDATE EXECUTOR θ", fc=TOP_FILL, ec=TOP_EDGE, color=INK)
    box(ax, 0.62, 5.78, 1.18, 0.72, "task +\nseed", fc=WHITE, ec=TOP_EDGE, weight="bold")
    dashed_group(ax, 2.08, 5.18, 3.42, 1.85, "wide rollout batch · diversity by sampling")
    for idx in range(5):
        yy = 5.36 + idx * 0.31
        box(ax, 2.35, yy, 2.85, 0.23, f"executor trajectory {idx + 1 if idx < 4 else '…N'}", fc=TOP_FILL, ec=TOP_EDGE, lw=0.7, radius=0.04, fontsize=7.1, pad=0.01)
        arrow(ax, 1.8, 6.14, 2.29, yy + 0.11, color=TOP_EDGE, lw=0.9, mutation=8)
    box(ax, 5.87, 5.72, 1.32, 0.84, "evaluate\nsolutions", fc=TOP_FILL, ec=TOP_EDGE, weight="bold")
    arrow(ax, 5.5, 6.14, 5.82, 6.14, color=TOP_EDGE)
    box(ax, 7.62, 5.86, 1.08, 0.56, "reward", fc=WHITE, ec=TOP_EDGE, weight="bold")
    arrow(ax, 7.19, 6.14, 7.57, 6.14, color=TOP_EDGE)
    box(ax, 9.08, 5.72, 1.55, 0.84, "∇θ\nupdate executor", fc="#DDE6EC", ec=TOP_EDGE, weight="bold")
    arrow(ax, 8.70, 6.14, 9.03, 6.14, color=TOP_EDGE)
    box(ax, 11.08, 5.72, 1.55, 0.84, "executor θ′\n(new weights)", fc="#CDD9E1", ec=TOP_EDGE, weight="bold")
    arrow(ax, 10.63, 6.14, 11.03, 6.14, color=TOP_EDGE)
    box(ax, 13.18, 5.72, 1.72, 0.84, "next discovery\nround", fc=WHITE, ec=TOP_EDGE, weight="bold")
    arrow(ax, 12.63, 6.14, 13.13, 6.14, color=TOP_EDGE)
    arrow(ax, 14.03, 6.64, 2.85, 7.13, color=TOP_EDGE, lw=1.0, mutation=10, connection="arc3,rad=0.15")
    ax.text(10.85, 7.02, "behavior is stored in a checkpoint", fontsize=8.3, color=MUTED, ha="center")

    # Divider and lower route.
    ax.plot([0.45, 15.55], [4.52, 4.52], color=LIGHT_LINE, lw=1.2)
    pill(ax, 0.58, 4.00, 3.45, "OURS · UPDATE PROPOSER φ, FREEZE EXECUTOR", fc=OURS_FILL, ec=OURS_EDGE, color="#A33F18")
    box(ax, 0.62, 2.35, 1.05, 0.66, "reward", fc=WHITE, ec=OURS_EDGE, weight="bold")
    box(ax, 2.03, 2.25, 1.35, 0.86, "∇φ\nupdate proposer", fc=OURS_FILL, ec=OURS_EDGE, weight="bold")
    arrow(ax, 1.67, 2.68, 1.98, 2.68, color=OURS_EDGE)
    box(ax, 3.78, 2.25, 1.2, 0.86, "proposer φ′", fc=OURS_FILL, ec=OURS_EDGE, weight="bold")
    arrow(ax, 3.38, 2.68, 3.73, 2.68, color=OURS_EDGE)
    harness_stack(ax, 5.36, 2.06, w=1.55, scale=0.95)
    arrow(ax, 4.98, 2.68, 5.30, 2.68, color=OURS_EDGE)
    box(ax, 7.30, 2.25, 1.32, 0.86, "frozen\nexecutor M₀", fc=BLUE_FILL, ec=BLUE, weight="bold")
    arrow(ax, 6.94, 2.68, 7.25, 2.68, color=BLUE)
    dashed_group(ax, 9.02, 1.90, 2.46, 1.55, "one controlled trajectory")
    strategy_strip(ax, 9.23, 2.38, 2.02, compact=True)
    box(ax, 9.24, 1.98, 2.01, 0.26, "diversity controller forces a switch", fc=PURPLE_FILL, ec=PURPLE, lw=0.7, radius=0.04, fontsize=6.5, color=PURPLE, weight="bold", pad=0.01)
    arrow(ax, 8.62, 2.68, 8.97, 2.68, color=BLUE)
    box(ax, 11.86, 2.25, 1.23, 0.86, "evaluate +\nvalidity gates", fc=GREEN_FILL, ec=GREEN, weight="bold")
    arrow(ax, 11.48, 2.68, 11.81, 2.68, color=GREEN)
    model_fanout(ax, 14.18, 1.95, width=1.10)
    arrow(ax, 13.09, 2.68, 14.10, 2.68, color=BLUE)

    # Four mechanism callouts.
    callouts = [
        (1, 9.10, 3.62, 2.45, "EXPLORATION EFFICIENCY", "force A→B→C inside\none trajectory", BLUE),
        (2, 5.05, 0.55, 2.45, "TRACEABLE HARNESS", "prompt · tool · skill\n· middleware", PURPLE),
        (3, 10.75, 0.55, 2.60, "INTEGRITY", "schema · sandbox · scope\n· fail closed", GREEN),
        (4, 13.62, 3.62, 1.85, "TRANSFER", "same artifact →\nnew frozen models", OURS_EDGE),
    ]
    for num, x, y, w, head, body, color in callouts:
        box(ax, x, y, w, 0.72, "", fc=WHITE, ec=color, lw=1.0, radius=0.09, pad=0.04)
        badge(ax, x + 0.18, y + 0.54, num, color)
        ax.text(x + 0.40, y + 0.55, head, ha="left", va="center", fontsize=7.4, color=color, weight="bold")
        ax.text(x + 0.18, y + 0.24, body, ha="left", va="center", fontsize=6.9, color=INK, linespacing=1.15)

    ax.text(15.38, 1.72, "portable,\nmodel-conditional", fontsize=6.7, color=MUTED, ha="center", va="top")
    save(fig, out)


def mini_parallel_rollouts(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    for idx in range(5):
        yy = y + 0.14 + idx * (h - 0.28) / 5
        box(ax, x + 0.20, yy, w - 0.40, 0.18, f"trajectory {idx + 1 if idx < 4 else '…N'}", fc=TOP_FILL, ec=TOP_EDGE, lw=0.6, radius=0.03, fontsize=6.2, pad=0.005)


def layout_b(out: Path) -> None:
    """Four-column contrast matrix: each reason gets its own mini-diagram."""
    fig, ax = canvas()
    title(
        ax,
        "Draft B · Four reasons, four mechanism contrasts",
        "Most explicit reading order: each column contrasts update-executor (top) with update-proposer (bottom).",
    )

    headers = [
        ("1", "EXPLORATION\nEFFICIENCY", BLUE),
        ("2", "TRACEABLE\nHARNESS", PURPLE),
        ("3", "REWARD-HACKING\nCONTROL", GREEN),
        ("4", "TRANSFER", OURS_EDGE),
    ]
    col_x = [2.35, 5.67, 8.99, 12.31]
    col_w = 3.02
    for (number, label_text, color), x in zip(headers, col_x):
        box(ax, x, 7.53, col_w, 0.58, "", fc=WHITE, ec=color, lw=1.2, radius=0.10, pad=0.03)
        badge(ax, x + 0.25, 7.82, int(number), color)
        ax.text(x + 0.52, 7.82, label_text, ha="left", va="center", fontsize=8.1, color=color, weight="bold", linespacing=0.95)

    # Row labels.
    box(ax, 0.42, 5.15, 1.55, 1.85, "TTT-DISCOVER\n\nREWARD\n↓\nEXECUTOR θ", fc=TOP_FILL, ec=TOP_EDGE, lw=1.2, radius=0.12, fontsize=8.4, weight="bold")
    box(ax, 0.42, 1.08, 1.55, 3.45, "OURS\n\nREWARD\n↓\nPROPOSER φ\n↓\nEXPLICIT HARNESS\n+\nFROZEN EXECUTOR", fc=OURS_FILL, ec=OURS_EDGE, lw=1.3, radius=0.12, fontsize=8.0, color="#8F3518", weight="bold")
    ax.text(1.20, 7.30, "WHERE REWARD GOES", ha="center", va="bottom", fontsize=7.5, color=MUTED, weight="bold")

    # Column backgrounds and row divider.
    for x in col_x:
        box(ax, x, 1.08, col_w, 5.92, "", fc="#FEFEFF", ec=LIGHT_LINE, lw=0.9, radius=0.10, pad=0.02, zorder=0)
        ax.plot([x + 0.08, x + col_w - 0.08], [4.78, 4.78], color=LIGHT_LINE, lw=0.9)

    # 1 Exploration.
    x = col_x[0]
    mini_parallel_rollouts(ax, x + 0.20, 5.25, col_w - 0.40, 1.30)
    ax.text(x + col_w / 2, 5.05, "wide rollout batch supplies diversity", ha="center", va="top", fontsize=7.2, color=MUTED)
    box(ax, x + 0.30, 3.60, col_w - 0.60, 0.42, "diversity_controller.py", fc=PURPLE_FILL, ec=PURPLE, lw=0.9, radius=0.06, fontsize=7.0, color=PURPLE, weight="bold", pad=0.02)
    strategy_strip(ax, x + 0.37, 2.73, col_w - 0.74, compact=True)
    arrow(ax, x + col_w / 2, 3.58, x + col_w / 2, 3.20, color=PURPLE, lw=1.1, mutation=10)
    ax.text(x + col_w / 2, 2.36, "one trajectory must switch\nconstruction family after a stall", ha="center", va="top", fontsize=7.2, color=INK)
    pill(ax, x + 0.56, 1.52, 1.90, "STRUCTURAL DIVERSITY", fc=BLUE_FILL, ec=BLUE, color=BLUE, fontsize=7.2)

    # 2 Traceability.
    x = col_x[1]
    box(ax, x + 0.62, 5.52, 1.78, 0.78, "executor θ′", fc="#D4E0E7", ec=TOP_EDGE, weight="bold")
    for idx in range(5):
        ax.plot([x + 0.72 + idx * 0.32, x + 0.90 + idx * 0.32], [5.72 + (idx % 2) * 0.18, 6.09 - (idx % 2) * 0.18], color=TOP_EDGE, lw=2.1, alpha=0.7)
    ax.text(x + col_w / 2, 5.12, "behavior is distributed\ninside updated weights", ha="center", va="top", fontsize=7.2, color=MUTED)
    harness_stack(ax, x + 0.53, 2.65, w=1.95, scale=1.05)
    ax.text(x + col_w / 2, 2.24, "every change is a named artifact", ha="center", va="top", fontsize=7.2, color=INK)
    pill(ax, x + 0.53, 1.52, 1.97, "VISIBLE ATTRIBUTION", fc=PURPLE_FILL, ec=PURPLE, color=PURPLE, fontsize=7.2)

    # 3 Integrity.
    x = col_x[2]
    box(ax, x + 0.45, 5.60, 0.85, 0.58, "model", fc=TOP_FILL, ec=TOP_EDGE, weight="bold", fontsize=8)
    box(ax, x + 1.72, 5.60, 0.85, 0.58, "score", fc=TOP_FILL, ec=TOP_EDGE, weight="bold", fontsize=8)
    arrow(ax, x + 1.30, 5.89, x + 1.67, 5.89, color=TOP_EDGE)
    ax.text(x + col_w / 2, 5.20, "a bad exploit can be hard\nto localize in θ′", ha="center", va="top", fontsize=7.2, color=MUTED)
    gates = [
        ("schema", 3.72),
        ("scope", 3.24),
        ("sandbox", 2.76),
        ("fail closed", 2.28),
    ]
    for label_text, yy in gates:
        box(ax, x + 0.52, yy, 1.98, 0.32, label_text, fc=GREEN_FILL, ec=GREEN, lw=0.8, radius=0.05, fontsize=7, color=GREEN, weight="bold", pad=0.01)
    ax.text(x + col_w / 2, 1.98, "no evaluator / answer-file access\nEVOLVE-BLOCK-only edits", ha="center", va="top", fontsize=7.0, color=INK)
    pill(ax, x + 0.65, 1.28, 1.70, "FAIL-CLOSED GATES", fc=GREEN_FILL, ec=GREEN, color=GREEN, fontsize=7.2)

    # 4 Transfer.
    x = col_x[3]
    box(ax, x + 0.62, 5.62, 1.78, 0.72, "executor checkpoint θ′", fc="#D4E0E7", ec=TOP_EDGE, weight="bold", fontsize=8)
    ax.text(x + col_w / 2, 5.15, "behavior remains tied\nto that checkpoint", ha="center", va="top", fontsize=7.2, color=MUTED)
    harness_stack(ax, x + 0.30, 2.57, w=1.20, scale=0.78)
    model_fanout(ax, x + 1.78, 2.42, width=0.90)
    for yy in (2.57, 2.99, 3.41, 3.83):
        arrow(ax, x + 1.52, 3.12, x + 1.72, yy + 0.15, color=BLUE, lw=0.8, mutation=7)
    ax.text(x + col_w / 2, 2.03, "same serialized harness\nnew frozen executors", ha="center", va="top", fontsize=7.2, color=INK)
    pill(ax, x + 0.45, 1.28, 2.15, "PORTABLE, CONDITIONAL", fc=OURS_FILL, ec=OURS_EDGE, color="#A33F18", fontsize=7.1)

    ax.text(15.42, 0.58, "Draft note: transferability ≠ universal positive transfer.", ha="right", va="center", fontsize=7.2, color=MUTED)
    save(fig, out)


def layout_c(out: Path) -> None:
    """A clean control-plane story with four benefits radiating from H."""
    fig, ax = canvas()
    title(
        ax,
        "Draft C · Learning location → reusable control plane",
        "Cleaner architecture view: two reward routes above, four harness capabilities below.",
    )

    # Two routes.
    box(ax, 0.62, 6.34, 14.76, 1.30, "", fc="#F7F9FA", ec=LIGHT_LINE, lw=1.0, radius=0.15, pad=0.03)
    pill(ax, 0.86, 7.12, 2.42, "TTT-DISCOVER", fc=TOP_FILL, ec=TOP_EDGE, color=INK)
    box(ax, 3.60, 6.63, 1.10, 0.58, "reward", fc=WHITE, ec=TOP_EDGE, weight="bold")
    arrow(ax, 4.70, 6.92, 5.20, 6.92, color=TOP_EDGE)
    box(ax, 5.25, 6.53, 1.55, 0.78, "∇θ\nupdate executor", fc=TOP_FILL, ec=TOP_EDGE, weight="bold")
    arrow(ax, 6.80, 6.92, 7.30, 6.92, color=TOP_EDGE)
    box(ax, 7.35, 6.53, 1.75, 0.78, "executor θ′", fc="#D4E0E7", ec=TOP_EDGE, weight="bold")
    arrow(ax, 9.10, 6.92, 9.60, 6.92, color=TOP_EDGE)
    box(ax, 9.65, 6.53, 2.10, 0.78, "improved solution", fc=WHITE, ec=TOP_EDGE, weight="bold")
    ax.text(13.34, 6.92, "learning and execution\nshare one weight space", ha="center", va="center", fontsize=8.2, color=MUTED)

    box(ax, 0.62, 4.70, 14.76, 1.30, "", fc="#FFFCFA", ec="#F0C4AA", lw=1.1, radius=0.15, pad=0.03)
    pill(ax, 0.86, 5.48, 2.42, "OURS · NexAU / SAH", fc=OURS_FILL, ec=OURS_EDGE, color="#A33F18")
    box(ax, 3.30, 4.99, 1.10, 0.58, "reward", fc=WHITE, ec=OURS_EDGE, weight="bold")
    arrow(ax, 4.40, 5.28, 4.90, 5.28, color=OURS_EDGE)
    box(ax, 4.95, 4.89, 1.55, 0.78, "∇φ\nupdate proposer", fc=OURS_FILL, ec=OURS_EDGE, weight="bold")
    arrow(ax, 6.50, 5.28, 7.00, 5.28, color=OURS_EDGE)
    box(ax, 7.05, 4.89, 1.75, 0.78, "proposer φ′", fc=OURS_FILL, ec=OURS_EDGE, weight="bold")
    arrow(ax, 8.80, 5.28, 9.30, 5.28, color=OURS_EDGE)
    box(ax, 9.35, 4.82, 2.00, 0.92, "EVOLVED HARNESS H", fc=PURPLE_FILL, ec=PURPLE, color=PURPLE, weight="bold")
    arrow(ax, 11.35, 5.28, 11.85, 5.28, color=BLUE)
    box(ax, 11.90, 4.89, 1.55, 0.78, "frozen\nexecutor M₀", fc=BLUE_FILL, ec=BLUE, weight="bold")
    arrow(ax, 13.45, 5.28, 13.95, 5.28, color=BLUE)
    box(ax, 14.00, 4.99, 1.05, 0.58, "solution", fc=WHITE, ec=BLUE, weight="bold", fontsize=8.5)

    # Central control-plane hub.
    ax.add_patch(Circle((8.0, 2.45), 0.82, facecolor=PURPLE_FILL, edgecolor=PURPLE, linewidth=1.6, zorder=4))
    ax.text(8.0, 2.58, "HARNESS", ha="center", va="center", fontsize=10.5, color=PURPLE, weight="bold", zorder=5)
    ax.text(8.0, 2.28, "control plane", ha="center", va="center", fontsize=7.5, color=INK, zorder=5)
    arrow(ax, 10.35, 4.78, 8.42, 3.20, color=PURPLE, lw=1.2, mutation=11, connection="arc3,rad=0.08")

    cards = [
        (0.70, 2.64, 2.80, 1.38, 1, "EXPLORATION EFFICIENCY", "diversity_controller.py\nforces A→B→C in one rollout", BLUE),
        (4.05, 0.50, 2.85, 1.40, 2, "TRACEABLE ARTIFACTS", "prompt.md · geometry_check.py\nskill · middleware · parameters", PURPLE),
        (9.10, 0.50, 2.85, 1.40, 3, "REWARD-HACKING CONTROL", "schema + sandbox + scopes\nno evaluator / answer-file access", GREEN),
        (12.48, 2.64, 2.82, 1.38, 4, "TRANSFER", "serialize H once; execute with\nheterogeneous frozen models", OURS_EDGE),
    ]
    for x, y, w, h, num, head, body, color in cards:
        box(ax, x, y, w, h, "", fc=WHITE, ec=color, lw=1.25, radius=0.13, pad=0.04)
        badge(ax, x + 0.25, y + h - 0.26, num, color)
        ax.text(x + 0.50, y + h - 0.27, head, ha="left", va="center", fontsize=7.7, color=color, weight="bold")
        ax.text(x + 0.22, y + 0.46, body, ha="left", va="center", fontsize=7.4, color=INK, linespacing=1.25)

    # Spokes from harness hub.
    arrow(ax, 7.34, 2.65, 3.58, 3.22, color=BLUE, lw=1.0, mutation=9, connection="arc3,rad=0.05")
    arrow(ax, 7.52, 1.79, 6.88, 1.42, color=PURPLE, lw=1.0, mutation=9)
    arrow(ax, 8.48, 1.79, 9.10, 1.42, color=GREEN, lw=1.0, mutation=9)
    arrow(ax, 8.66, 2.65, 12.40, 3.22, color=OURS_EDGE, lw=1.0, mutation=9, connection="arc3,rad=-0.05")
    model_fanout(ax, 13.45, 1.00, width=1.10)
    ax.text(15.26, 0.72, "portable; effect remains\nmodel-conditional", ha="right", va="top", fontsize=6.8, color=MUTED)

    save(fig, out)


def layout_d(out: Path) -> None:
    """Polished, strictly symmetric paired comparison."""
    fig, ax = canvas()

    ax.text(8.0, 8.66, "Why update the proposer?", ha="center", va="center", fontsize=20, color=NAVY, weight="bold")
    ax.text(
        8.0,
        8.38,
        "The same task reward improves a different object—and exposes a different set of capabilities.",
        ha="center",
        va="center",
        fontsize=9.7,
        color=MUTED,
    )

    columns = [
        ("1", "EXPLORATION EFFICIENCY", BLUE),
        ("2", "TRACEABLE ARTIFACTS", PURPLE),
        ("3", "REWARD-HACKING CONTROL", GREEN),
        ("4", "TRANSFER", OURS_EDGE),
    ]
    xs = [0.70, 4.45, 8.20, 11.95]
    cw = 3.52
    top_y, bottom_y, card_h = 4.72, 1.62, 2.14

    # Identical four-column grid.
    for x, (number, label_text, color) in zip(xs, columns):
        box(ax, x, 7.72, cw, 0.47, "", fc=WHITE, ec=color, lw=1.2, radius=0.10, pad=0.025)
        badge(ax, x + 0.27, 7.955, int(number), color)
        ax.text(x + 0.52, 7.955, label_text, ha="left", va="center", fontsize=7.9, color=color, weight="bold")
        box(ax, x, top_y, cw, card_h, "", fc="#FBFCFD", ec=LIGHT_LINE, lw=0.95, radius=0.09, pad=0.02, zorder=0)
        box(ax, x, bottom_y, cw, card_h, "", fc="#FFFEFD", ec=LIGHT_LINE, lw=0.95, radius=0.09, pad=0.02, zorder=0)
        ax.add_patch(Rectangle((x, top_y + card_h - 0.055), cw, 0.055, facecolor=TOP_EDGE, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, bottom_y + card_h - 0.055), cw, 0.055, facecolor=color, edgecolor="none", zorder=2))

    # Full-width, aligned route labels.
    box(ax, 0.70, 7.08, 14.77, 0.39, "", fc=TOP_FILL, ec=TOP_EDGE, lw=0.9, radius=0.08, pad=0.02)
    ax.text(0.94, 7.275, "TTT-DISCOVER", ha="left", va="center", fontsize=8.2, color=INK, weight="bold")
    ax.text(8.08, 7.275, "REWARD  →  UPDATE EXECUTOR WEIGHTS  θ", ha="center", va="center", fontsize=9.1, color=INK, weight="bold")
    ax.text(15.22, 7.275, "executor adapts", ha="right", va="center", fontsize=7.5, color=MUTED)

    box(ax, 0.70, 4.03, 14.77, 0.39, "", fc=OURS_FILL, ec=OURS_EDGE, lw=1.0, radius=0.08, pad=0.02)
    ax.text(0.94, 4.225, "NexAU / SAH", ha="left", va="center", fontsize=8.2, color="#9A3D1D", weight="bold")
    ax.text(8.08, 4.225, "REWARD  →  UPDATE PROPOSER  φ    •    FREEZE EXECUTOR  M₀", ha="center", va="center", fontsize=9.1, color="#9A3D1D", weight="bold")
    ax.text(15.22, 4.225, "harness adapts", ha="right", va="center", fontsize=7.5, color="#A65A3B")

    # 1 · Exploration efficiency.
    x = xs[0]
    ax.text(x + cw / 2, 6.57, "independent rollout batch", ha="center", va="center", fontsize=7.2, color=MUTED, weight="bold")
    for idx in range(5):
        yy = 6.29 - idx * 0.22
        box(
            ax,
            x + 0.43,
            yy,
            cw - 0.86,
            0.14,
            f"trajectory {idx + 1 if idx < 4 else '… N'}",
            fc=TOP_FILL,
            ec=TOP_EDGE,
            lw=0.55,
            radius=0.025,
            fontsize=5.9,
            pad=0.004,
        )
    ax.text(x + cw / 2, 4.98, "diversity comes from more trajectories", ha="center", va="center", fontsize=7.2, color=INK)

    box(ax, x + 0.51, 3.22, cw - 1.02, 0.29, "diversity_controller.py", fc=PURPLE_FILL, ec=PURPLE, lw=0.8, radius=0.05, fontsize=6.9, color=PURPLE, weight="bold", pad=0.01)
    arrow(ax, x + cw / 2, 3.19, x + cw / 2, 2.91, color=PURPLE, lw=1.0, mutation=9)
    strategy_strip(ax, x + 0.54, 2.44, cw - 1.08, compact=True)
    ax.text(x + cw / 2, 1.90, "one trajectory · forced strategy switch", ha="center", va="center", fontsize=7.2, color=INK)

    # 2 · Traceability.
    x = xs[1]
    box(ax, x + 0.66, 5.66, 2.20, 0.70, "executor checkpoint  θ′", fc="#D9E3E9", ec=TOP_EDGE, lw=1.0, radius=0.10, fontsize=8.2, color=INK, weight="bold")
    for idx in range(6):
        ax.plot(
            [x + 0.83 + idx * 0.30, x + 1.00 + idx * 0.30],
            [5.84 + (idx % 2) * 0.16, 6.17 - (idx % 2) * 0.16],
            color=TOP_EDGE,
            lw=1.7,
            alpha=0.55,
            zorder=4,
        )
    ax.text(x + cw / 2, 5.15, "behavior is implicit in updated weights", ha="center", va="center", fontsize=7.2, color=INK)

    artifact_specs = [
        ("PROMPT", BLUE_FILL, BLUE),
        ("TOOL", PURPLE_FILL, PURPLE),
        ("SKILL", GREEN_FILL, GREEN),
        ("MIDDLEWARE", GOLD_FILL, GOLD),
    ]
    for idx, (label_text, fc, ec) in enumerate(artifact_specs):
        xx = x + 0.44 + (idx % 2) * 1.35
        yy = 2.83 - (idx // 2) * 0.50
        box(ax, xx, yy, 1.18, 0.34, label_text, fc=fc, ec=ec, lw=0.8, radius=0.05, fontsize=6.8, color=ec, weight="bold", pad=0.01)
    ax.text(x + cw / 2, 1.90, "every improvement is a named, inspectable diff", ha="center", va="center", fontsize=7.2, color=INK)

    # 3 · Reward-hacking control.
    x = xs[2]
    box(ax, x + 0.42, 5.73, 1.05, 0.54, "program", fc=TOP_FILL, ec=TOP_EDGE, lw=0.9, radius=0.08, fontsize=7.5, weight="bold")
    box(ax, x + 2.05, 5.73, 1.05, 0.54, "score", fc=TOP_FILL, ec=TOP_EDGE, lw=0.9, radius=0.08, fontsize=7.5, weight="bold")
    arrow(ax, x + 1.47, 6.00, x + 2.01, 6.00, color=TOP_EDGE, lw=1.2, mutation=10)
    ax.add_patch(Circle((x + 1.76, 6.40), 0.13, facecolor=RED_FILL, edgecolor=RED, linewidth=1.0, zorder=6))
    ax.text(x + 1.76, 6.39, "!", ha="center", va="center", fontsize=8.0, color=RED, weight="bold", zorder=7)
    ax.text(x + cw / 2, 5.16, "a scoring exploit can be hard to localize", ha="center", va="center", fontsize=7.2, color=INK)

    gate_specs = [("schema", 0, 0), ("scope", 1, 0), ("sandbox", 0, 1), ("fail closed", 1, 1)]
    for label_text, col, row in gate_specs:
        xx = x + 0.44 + col * 1.35
        yy = 2.83 - row * 0.50
        box(ax, xx, yy, 1.18, 0.34, label_text, fc=GREEN_FILL, ec=GREEN, lw=0.8, radius=0.05, fontsize=6.8, color=GREEN, weight="bold", pad=0.01)
    ax.text(x + cw / 2, 1.90, "forbidden edits and evaluator access fail closed", ha="center", va="center", fontsize=7.2, color=INK)

    # 4 · Transfer.
    x = xs[3]
    box(ax, x + 0.60, 5.72, 2.32, 0.60, "executor checkpoint  θ′", fc="#D9E3E9", ec=TOP_EDGE, lw=1.0, radius=0.09, fontsize=8.0, weight="bold")
    arrow(ax, x + 1.76, 5.68, x + 1.76, 5.39, color=TOP_EDGE, lw=1.0, mutation=9)
    pill(ax, x + 1.05, 4.98, 1.42, "checkpoint-bound", fc=TOP_FILL, ec=TOP_EDGE, color=INK, fontsize=6.7)

    harness_stack(ax, x + 0.34, 2.33, w=1.15, scale=0.72)
    compact_models = [("GLM", 3.17), ("Qwen", 2.77), ("GPT", 2.37), ("Claude", 1.97)]
    for model_name, yy in compact_models:
        box(ax, x + 2.25, yy, 0.92, 0.27, model_name, fc=BLUE_FILL, ec=BLUE, lw=0.75, radius=0.05, fontsize=6.2, color=BLUE, weight="bold", pad=0.01)
        arrow(ax, x + 1.53, 2.69, x + 2.19, yy + 0.13, color=BLUE, lw=0.75, mutation=7)
    ax.text(x + cw / 2, 1.81, "same harness · new frozen models", ha="center", va="center", fontsize=7.2, color=INK)

    ax.text(
        8.0,
        0.87,
        "Explicit harness control makes exploration, attribution, integrity, and reuse inspectable; transfer remains model-conditional.",
        ha="center",
        va="center",
        fontsize=8.1,
        color=MUTED,
    )
    save(fig, out)


def layout_e(out: Path) -> None:
    """Reference-style, symmetric evolution loops with explicit feedback."""
    fig, ax = canvas()

    ax.text(8.0, 8.66, "Two evolution loops, two update targets", ha="center", va="center", fontsize=20, color=NAVY, weight="bold")
    ax.text(
        8.0,
        8.36,
        "TTT-Discover evolves executor weights; NexAU / SAH evolves an explicit harness around a frozen executor.",
        ha="center",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )

    # Large paired process regions.
    box(ax, 0.46, 5.06, 15.08, 2.92, "", fc="#F4F7F9", ec="#C8D4DC", lw=1.0, radius=0.12, pad=0.025, zorder=0)
    box(ax, 0.46, 1.22, 15.08, 3.56, "", fc="#FFFCFA", ec="#F0C5B0", lw=1.0, radius=0.12, pad=0.025, zorder=0)

    # Route headers.
    pill(ax, 0.70, 7.48, 3.18, "TTT-DISCOVER  ·  EVOLVE EXECUTOR", fc=TOP_FILL, ec=TOP_EDGE, color=INK, fontsize=8.2)
    ax.text(8.00, 7.65, "θₜ  →  θₜ₊₁", ha="center", va="center", fontsize=10.2, color=TOP_EDGE, weight="bold")
    ax.text(15.20, 7.65, "executor adapts", ha="right", va="center", fontsize=7.5, color=MUTED)

    pill(ax, 0.70, 4.29, 3.18, "NexAU / SAH  ·  EVOLVE PROPOSER", fc=OURS_FILL, ec=OURS_EDGE, color="#9A3D1D", fontsize=8.2)
    ax.text(8.00, 4.46, "φₜ  →  φₜ₊₁    ·    executor M₀ stays frozen", ha="center", va="center", fontsize=10.0, color="#A04421", weight="bold")
    ax.text(15.20, 4.46, "harness adapts", ha="right", va="center", fontsize=7.5, color="#A65A3B")

    stage_x = [0.64 + i * 1.88 for i in range(8)]
    stage_w = 1.47
    top_y, bottom_y, stage_h = 5.53, 2.66, 0.94

    def stage(
        x: float,
        y: float,
        label_text: str,
        *,
        fc: str,
        ec: str,
        color: str = INK,
        fontsize: float = 7.7,
    ) -> None:
        box(ax, x, y, stage_w, stage_h, label_text, fc=fc, ec=ec, lw=0.95, radius=0.10, fontsize=fontsize, color=color, weight="bold", pad=0.025)

    # TTT-Discover evolution path.
    top_specs = [
        ("TASK\n+ SEED", WHITE, TOP_EDGE),
        ("EXECUTOR\nθₜ", TOP_FILL, TOP_EDGE),
        ("", TOP_FILL, TOP_EDGE),
        ("CANDIDATE\nPROGRAMS", WHITE, TOP_EDGE),
        ("EVALUATOR", TOP_FILL, TOP_EDGE),
        ("REWARD\nr", WHITE, TOP_EDGE),
        ("∇θ\nUPDATE", TOP_FILL, TOP_EDGE),
        ("EXECUTOR\nθₜ₊₁", "#D7E2E8", TOP_EDGE),
    ]
    for x, (label_text, fc, ec) in zip(stage_x, top_specs):
        stage(x, top_y, label_text, fc=fc, ec=ec)
    for left, right in zip(stage_x[:-1], stage_x[1:]):
        arrow(ax, left + stage_w + 0.03, top_y + stage_h / 2, right - 0.07, top_y + stage_h / 2, color=TOP_EDGE, lw=1.15, mutation=10)

    # Mini parallel-rollout stack in stage 3.
    rollout_x = stage_x[2]
    ax.text(rollout_x + stage_w / 2, top_y + 0.77, "ROLLOUTS", ha="center", va="center", fontsize=6.6, color=INK, weight="bold")
    for idx in range(4):
        yy = top_y + 0.55 - idx * 0.14
        box(ax, rollout_x + 0.19, yy, stage_w - 0.38, 0.095, f"trajectory {idx + 1 if idx < 3 else '… N'}", fc=WHITE, ec=TOP_EDGE, lw=0.45, radius=0.018, fontsize=4.9, pad=0.002)

    # Executor feedback loop.
    arrow(
        ax,
        stage_x[7] + stage_w / 2,
        top_y + stage_h + 0.03,
        stage_x[1] + stage_w / 2,
        top_y + stage_h + 0.03,
        color=TOP_EDGE,
        lw=1.05,
        mutation=10,
        connection="arc3,rad=0.18",
        zorder=3,
    )
    ax.text(8.65, 7.02, "next TTT round", ha="center", va="center", fontsize=7.0, color=MUTED, weight="bold")

    # NexAU / SAH evolution path.
    bottom_specs = [
        ("TASK\n+ Hₜ", WHITE, OURS_EDGE),
        ("PROPOSER\nφₜ", OURS_FILL, OURS_EDGE),
        ("", PURPLE_FILL, PURPLE),
        ("FROZEN\nEXECUTOR M₀", BLUE_FILL, BLUE),
        ("", BLUE_FILL, BLUE),
        ("EVALUATOR\n+ GATES", GREEN_FILL, GREEN),
        ("REWARD\nr", WHITE, OURS_EDGE),
        ("∇φ\nUPDATE PROPOSER", OURS_FILL, OURS_EDGE),
    ]
    for x, (label_text, fc, ec) in zip(stage_x, bottom_specs):
        stage(x, bottom_y, label_text, fc=fc, ec=ec, color=ec if label_text else INK, fontsize=7.4)
    for left, right in zip(stage_x[:-1], stage_x[1:]):
        arrow(ax, left + stage_w + 0.03, bottom_y + stage_h / 2, right - 0.07, bottom_y + stage_h / 2, color=OURS_EDGE, lw=1.15, mutation=10)

    # Explicit candidate harness in stage 3.
    harness_x = stage_x[2]
    ax.text(harness_x + stage_w / 2, bottom_y + 0.79, "CANDIDATE HARNESS", ha="center", va="center", fontsize=5.9, color=PURPLE, weight="bold")
    mini_artifacts = [("PROMPT", BLUE), ("TOOL", PURPLE), ("SKILL", GREEN), ("MIDDLEWARE", GOLD)]
    for idx, (label_text, color) in enumerate(mini_artifacts):
        xx = harness_x + 0.15 + (idx % 2) * 0.60
        yy = bottom_y + 0.45 - (idx // 2) * 0.25
        box(ax, xx, yy, 0.54, 0.17, label_text, fc=WHITE, ec=color, lw=0.55, radius=0.025, fontsize=4.5, color=color, weight="bold", pad=0.003)

    # One controlled trajectory in stage 5.
    controlled_x = stage_x[4]
    ax.text(controlled_x + stage_w / 2, bottom_y + 0.77, "ONE TRAJECTORY", ha="center", va="center", fontsize=5.9, color=BLUE, weight="bold")
    node_w = 0.30
    for idx, label_text in enumerate(("A", "B", "C")):
        xx = controlled_x + 0.17 + idx * 0.43
        box(ax, xx, bottom_y + 0.32, node_w, 0.27, label_text, fc=WHITE, ec=BLUE, lw=0.65, radius=0.04, fontsize=5.8, color=BLUE, weight="bold", pad=0.003)
        if idx < 2:
            arrow(ax, xx + node_w + 0.02, bottom_y + 0.455, xx + 0.40, bottom_y + 0.455, color=BLUE, lw=0.7, mutation=6)
    ax.text(controlled_x + stage_w / 2, bottom_y + 0.13, "forced switch", ha="center", va="center", fontsize=5.2, color=PURPLE, weight="bold")

    # Fail-closed gates visible inside evaluator stage.
    gate_x = stage_x[5]
    for idx, gate_name in enumerate(("schema", "sandbox")):
        box(ax, gate_x + 0.15 + idx * 0.61, bottom_y + 0.15, 0.55, 0.17, gate_name, fc=WHITE, ec=GREEN, lw=0.55, radius=0.025, fontsize=4.6, color=GREEN, weight="bold", pad=0.003)

    # Proposer feedback loop.
    arrow(
        ax,
        stage_x[7] + stage_w / 2,
        bottom_y - 0.03,
        stage_x[1] + stage_w / 2,
        bottom_y - 0.03,
        color=OURS_EDGE,
        lw=1.10,
        mutation=10,
        connection="arc3,rad=-0.20",
        zorder=3,
    )
    ax.text(8.65, 1.74, "next harness-evolution round", ha="center", va="center", fontsize=7.0, color="#A65A3B", weight="bold")

    # Numbered mechanism notes; these annotate the loop rather than replace it.
    notes = [
        (0.58, 0.35, 3.55, 1, "EXPLORATION EFFICIENCY", "structural diversity inside one trajectory", BLUE),
        (4.35, 0.35, 3.55, 2, "TRACEABLE HARNESS", "named prompt · tool · skill · middleware", PURPLE),
        (8.12, 0.35, 3.55, 3, "REWARD-HACKING CONTROL", "schema · scope · sandbox · fail closed", GREEN),
        (11.89, 0.35, 3.55, 4, "TRANSFER", "same harness → heterogeneous frozen models", OURS_EDGE),
    ]
    for x, y, w, number, heading, detail, color in notes:
        box(ax, x, y, w, 0.61, "", fc=WHITE, ec=color, lw=0.95, radius=0.09, pad=0.02)
        badge(ax, x + 0.24, y + 0.31, number, color)
        ax.text(x + 0.49, y + 0.40, heading, ha="left", va="center", fontsize=6.7, color=color, weight="bold")
        ax.text(x + 0.49, y + 0.18, detail, ha="left", va="center", fontsize=5.8, color=INK)

    # Small numbered anchors on the relevant SAH stages.
    anchor_y = bottom_y + stage_h + 0.32
    badge(ax, stage_x[4] + 0.18, anchor_y, 1, BLUE)
    badge(ax, stage_x[2] + 0.18, anchor_y, 2, PURPLE)
    badge(ax, stage_x[5] + 0.18, anchor_y, 3, GREEN)
    badge(ax, stage_x[2] + stage_w - 0.18, anchor_y, 4, OURS_EDGE)

    save(fig, out)


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def contact_sheet(paths: list[Path], out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 25.5))
    fig.patch.set_facecolor(WHITE)
    for ax, path, label_text in zip(axes, paths, ("A · Mirrored loops", "B · Four contrasts", "C · Control plane")):
        ax.imshow(plt.imread(path.with_suffix(".png")))
        ax.axis("off")
        ax.set_title(label_text, loc="left", fontsize=15, color=NAVY, weight="bold", pad=10)
    fig.subplots_adjust(hspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figure_drafts/why_update_proposer"),
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    bases = [
        out_dir / "layout_a_mirrored_loops",
        out_dir / "layout_b_four_contrasts",
        out_dir / "layout_c_control_plane",
    ]
    layout_a(bases[0])
    layout_b(bases[1])
    layout_c(bases[2])
    contact_sheet(bases, out_dir / "layout_contact_sheet.png")
    layout_d(out_dir / "layout_d_symmetric_comparison")
    layout_e(out_dir / "layout_e_symmetric_evolution_loops")


if __name__ == "__main__":
    main()

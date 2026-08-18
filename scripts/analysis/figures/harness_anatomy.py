#!/usr/bin/env python3
"""Harness anatomy: fragment-level prior work vs our complete generated H2.

Companion panel to figures/overview_1.pdf.  Left: what patch/skill-level
self-evolving methods emit (e.g., Harness-R1's add_code_hook JSON patches)
into a fixed scaffold.  Right: the complete executable harness our proposer
generates -- agent.yaml as the mount authority over prompt, tools (incl.
generated tool code), skills, middlewares, and the LLM config.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, DBLUE, GREY, LGREY, ORANGE = "#2B5F97", "#084594", "#8a8a8a", "#b9b9b9", "#e07b28"

fig, ax = plt.subplots(figsize=(11.6, 5.0))
ax.set_xlim(0, 11.6); ax.set_ylim(0, 5.0); ax.axis("off")

def card(x, y, w, h, title, lines, ec, fc="white", title_c=None, lw=1.2,
         mono_lines=True, ls="-", z=3, title_fs=9.0, line_fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    ax.text(x + 0.12, y + h - 0.13, title, fontsize=title_fs, weight="bold",
            color=title_c or ec, va="top", zorder=z + 1)
    body = "\n".join(lines)
    ax.text(x + 0.12, y + h - 0.40, body, fontsize=line_fs, color="#333333",
            va="top", family="monospace" if mono_lines else None,
            linespacing=1.45, zorder=z + 1)

# ============ LEFT: fragment-level prior work ============
ax.text(1.62, 4.72, "Prior self-evolving methods:\nfragments into a fixed scaffold",
        fontsize=10, weight="bold", color="#444444", ha="center", va="top")

card(0.25, 2.55, 2.75, 1.35, "a code patch",
     ['{"add_code_hook": {', '   "on_init": ...,', '   "on_before_action": ...}}',
      "# hooks over a FIXED agent"],
     GREY, ls=(0, (4, 2.4)))
card(0.55, 0.75, 2.75, 1.15, "a skill / prompt snippet",
     ["Try step functions first;", "restart when stalled.",
      "# advice, not machinery"],
     GREY, ls=(0, (4, 2.4)), mono_lines=False)

ax.text(1.62, 0.34, "prompt, tools, config stay fixed",
        fontsize=8.2, color="#777777", ha="center", style="italic")

# ============ RIGHT: our complete generated harness ============
ax.text(7.6, 4.72, "Ours: the proposer generates the complete harness $H_k$",
        fontsize=10.5, weight="bold", color=DBLUE, ha="center", va="top")

# package container
ax.add_patch(FancyBboxPatch((3.75, 0.28), 7.6, 4.02,
    boxstyle="round,pad=0.02,rounding_size=0.09",
    fc="#f4f8fc", ec=DBLUE, lw=1.6, zorder=1))
ax.text(11.2, 0.42, "compiled into a runnable agent",
        fontsize=8.2, color=DBLUE, ha="right", style="italic", zorder=2)

# mount authority
card(6.55, 3.30, 2.15, 0.72, "agent.yaml", ["mount authority:", "binds every component"],
     DBLUE, fc="#dce9f6", lw=1.5, title_fs=9.5)

# component cards (2 rows x 3)
comp = [
    (3.95, 1.85, "prompt.md",
     ["executor system prompt:", "strategy + workflow"]),
    (6.55, 1.85, "tools/",
     ["descriptions +", "custom_tools/*.py", "(generated tool code)"]),
    (9.15, 1.85, "skills/*/SKILL.md",
     ["task playbooks,", "inherited + new"]),
    (3.95, 0.55, "middlewares/*.py",
     ["runtime hooks: reminders,", "probe-budget gates,", "forced diversity"]),
    (6.55, 0.55, "llm config",
     ["temperature, top_p, top_k,", "max_tokens, max_iterations"]),
    (9.15, 0.55, "sampling & control",
     ["evaluator budget use,", "stall / restart policy"]),
]
W, H = 2.30, 1.08
for x, y, title, lines in comp:
    card(x, y, W, H, title, lines, BLUE, lw=1.2, title_fs=8.8,
         mono_lines=False)

# mount edges from agent.yaml
for x, y, *_ in comp:
    tx, ty = x + W / 2, y + H
    sx = 6.55 + 2.15 / 2 + (0.55 if tx > 8.6 else (-0.55 if tx < 5.6 else 0.0))
    ax.add_patch(FancyArrowPatch((sx, 3.30), (tx, ty + 0.03),
        arrowstyle="-", color=DBLUE, lw=0.9, alpha=0.55,
        connectionstyle="arc3,rad=0.0", zorder=2))

# correspondence: their whole output ~= one of our component types
ax.add_patch(FancyArrowPatch((2.35, 2.55), (3.95, 1.15),
    arrowstyle="-|>", mutation_scale=10, color=ORANGE, lw=1.4,
    ls=(0, (4, 2.2)), connectionstyle="arc3,rad=-0.30", zorder=4))
ax.text(2.05, 2.22, "their entire output\n$\\approx$ one component type",
        fontsize=7.8, color=ORANGE, ha="center", va="center", zorder=4)

fig.tight_layout()
for out in ("papers/figures/harness_anatomy.pdf", "papers/figures/harness_anatomy.png"):
    fig.savefig(out, dpi=200)
print("wrote harness_anatomy.{pdf,png}")

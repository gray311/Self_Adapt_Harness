#!/usr/bin/env python3
"""Method-comparison radar over the 11 tasks.

Values are frozen campaign-result reporting values. Each axis linearly
rescales the plotted systems into the band
[R0, 1] (direction-corrected so outward is always better), so no system
collapses to the center and shapes are comparable within an axis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["Erdős", "AC1", "AC2", "CP", "Hadamard",
         "ahc039", "ahc058", "EPLB", "PRISM", "LLM-SQL", "Txn"]
LOWER_BETTER = {"Erdős", "AC1"}
R0 = 0.35

SERIES = {
 "Initial Program":
   ([0.495056, 1.5186, 0.8558, 0.959764, 0.143275, 534850, 0,
     0.1265, 21.89, 0.6856, 2824.86],
    dict(color="#b0b0b0", ls=":",  lw=1.6, zorder=3)),
 "Best Human":
   ([0.380927, 1.5097, 0.9015, 2.634000, 0.935673, 566997, 847674723,
     0.1265, 21.89, 0.6920, 2724.80],
    dict(color="#333333", ls="-.", lw=1.6, zorder=4)),
 "Qwen3.5-9B + OpenEvolve":
   ([0.385512, 1.5186, 0.8801, 1.172702, 0.397184, 553582, 134486700,
     0.1269, 22.36, 0.6858, 3584.23],
    dict(color="#8a8a8a", ls="-",  lw=1.8, zorder=5)),
 "Finch-9B + OpenEvolve":
   ([0.381100, 1.5141, 0.9122, 1.936000, 0.480585, 553759, 525286896,
     0.1265, 23.93, 0.7024, 3636.36],
    dict(color="#e07b28", ls="-",  lw=2.0, zorder=6)),
 "Previous SOTA ($\\leq$10B)":
   ([0.380932, 1.5031, 0.9472, 2.635983, 0.576400, 557168, 525286896,
     0.1270, 24.70, 0.7341, 4761.90],
    dict(color="#c0392b", ls="--", lw=2.0, zorder=7)),
 "HarnessRL (ours)":
   ([0.380919, 1.5098, 0.9339, 2.541000, 0.573283, 559534, 713552303,
     0.1272, 26.26, 0.7415, 4255.32],
    dict(color="#084594", ls="-",  lw=2.6, zorder=9, fill=True)),
}

vals = np.array([v for v, _ in SERIES.values()], dtype=float)
norm = np.zeros_like(vals)
for j, task in enumerate(TASKS):
    col = vals[:, j].copy()
    if task in LOWER_BETTER:
        col = -col
    lo, hi = col.min(), col.max()
    norm[:, j] = R0 + (1 - R0) * (col - lo) / (hi - lo)

N = len(TASKS)
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])

fig, ax = plt.subplots(figsize=(7.4, 7.4), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_ylim(0, 1.06)
ax.set_yticks([0.35, 0.5675, 0.785, 1.0]); ax.set_yticklabels([])
ax.grid(color="#e3e3e3", lw=0.8)
ax.spines["polar"].set_color("#c8c8c8")
ax.set_xticks(ang); ax.set_xticklabels(TASKS, fontsize=11.5)
ax.tick_params(pad=12)

for (name, (v, st)), row in zip(SERIES.items(), norm):
    r = np.concatenate([row, row[:1]])
    ax.plot(ang_c, r, color=st["color"], ls=st["ls"], lw=st["lw"],
            zorder=st["zorder"], label=name,
            marker="o" if st.get("fill") else None, ms=4)
    if st.get("fill"):
        ax.fill(ang_c, r, color=st["color"], alpha=0.12, zorder=st["zorder"]-1)

fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
           bbox_to_anchor=(0.5, 0.005), fontsize=9.8, frameon=False,
           handlelength=2.2, ncols=3, columnspacing=1.4)
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.14)
for out in ("papers/figures/radar_method_ablation.pdf",
            "papers/figures/radar_method_ablation.png"):
    fig.savefig(out, dpi=200)
print("wrote radar_method_ablation.{pdf,png}")

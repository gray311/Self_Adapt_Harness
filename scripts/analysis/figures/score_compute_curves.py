#!/usr/bin/env python3
r"""Score-vs-compute curves, one panel per task.

Tests the hypothesis that updating the PROPOSER internalizes reward faster per
executor rollout than (a) leaving the proposer fixed and evolving only external
context, and (b) adapting the executor's own weights (test-time RL).

Everything plotted from our side is measured, not schematic:
  x = cumulative EXECUTOR ROLLOUTS actually spent on the task (one rollout = one
      candidate harness rolled out once by the frozen executor)
  y = best VALID score so far, direction-corrected and normalized so that 0 is
      the task's seed program and 1.0 is the published <=10B best

Series:
  "Update proposer (ours)"  rounds driven by a trained phi (mphi_*)
  "Fixed proposer"          rounds driven by the untrained base phi -- the
                            OpenEvolve-like condition: the harness/context still
                            evolves, the policy proposing it does not
  "TTT-Discover"            published endpoint only. Their repo ships final
                            scores, not per-compute traces, and reproducing the
                            trace needs their training API, so we plot the single
                            (budget, score) point they report and say so.
"""
import json, glob, os, re, sys, collections

RUN = os.environ.get("RUN_ROOT") or "/lustre/fsw/portfolios/av/users/yingzim/runs"
R = f"{RUN}/self_adapt_harness"
BASE_PHI = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

TASKS = [
    ("eft__math__erdos_min_overlap",    "Erdős min-overlap",       False),
    ("eft__math__circle_packing",       "Circle packing (n=26)",   False),
    ("eft__math__hadamard_maximal_det", "Hadamard max-det",        False),
    ("eft__math__first_autocorr_ineq",  "Autocorrelation I",       False),
    ("eft__math__second_autocorr_ineq", "Autocorrelation II",      False),
    ("eft__ahc_simpletes__ahc039",      "AHC039",                  False),
]

# seed (0.0 on the y axis) and published <=10B best (1.0), in COMBINED units
ANCHOR = {   # (initial program, published <=10B best) in combined units
    "eft__math__erdos_min_overlap":    (0.769452, 0.999974),
    "eft__math__circle_packing":       (0.364237, 1.000373),
    "eft__math__hadamard_maximal_det": (0.143275, 0.576400),
    "eft__math__first_autocorr_ineq":  (0.991237, 1.001437),
    "eft__math__second_autocorr_ineq": (0.954836, 1.056813),
    "eft__ahc_simpletes__ahc039":      (2.377111, 2.476302),
}

# Local TTT-Discover-style arm: update the EXECUTOR on its own high-scoring
# rollouts with a fixed harness and no proposer.  This is a budget-scaled
# reference ablation, not an official reproduction. Each task's point sits at
# the number of executor rollouts whose programs went into its training set,
# plus the K rollouts spent evaluating it.
TTT_DIR = f"{R}/ttt_arm"
# TTT-Discover's own Qwen3-8B run (arXiv:2601.16175, Table 2) -- the like-for-like
# ~10B comparison -- at their reported budget of 512 rollouts/step x 50 steps.
TTT_BUDGET = 25600
TTT_PUBLISHED = {
    "eft__math__erdos_min_overlap":    0.380922 / 0.380932,   # 0.380932 raw
    "eft__math__first_autocorr_ineq":  1.505293 / 1.50525,    # 1.50525 raw
    "eft__math__second_autocorr_ineq": 0.9472 / 0.896280,     # 0.9472 raw
}
TTT_FINAL = {
    "eft__math__erdos_min_overlap":    0.999974,   # holds the <=10B best
    "eft__math__second_autocorr_ineq": 1.056813,   # holds the <=10B best
    "eft__math__first_autocorr_ineq":  None,
    "eft__math__circle_packing":       None,
    "eft__math__hadamard_maximal_det": None,
    "eft__ahc_simpletes__ahc039":      None,
}


def round_phi_map():
    m = {}
    for lg in glob.glob(f"{R}/*/*/driver*.log") + glob.glob(f"{R}/*/driver*.log"):
        try: txt = open(lg, errors="ignore").read()
        except OSError: continue
        for mm in re.finditer(r"round(\d+) propose \(phi=([^)]+)\)", txt):
            m[int(mm.group(1))] = mm.group(2)
    return m


def series(task, lower, phi_map):
    """Best-so-far over the FULL shared campaign timeline for this task.

    The campaign's rounds share one program ratchet: a round driven by the base
    proposer can inherit an incumbent that trained-proposer rounds built. Giving
    each proposer kind its own cumulative-rollout axis therefore credits the base
    rounds with compute they never spent -- which is what made the context-only
    curve look faster than it is. So we plot a single campaign curve on the real
    shared budget, and take the controlled arms (context_v2, TTT) from runs that
    each start from the fixed initial harness with their own ratchet.
    """
    items = []
    for f in glob.glob(f"{R}/outer/round*/round_summary.json"):
        rd = int(re.search(r"round(\d+)", f).group(1))
        if rd not in phi_map:
            continue
        try: g = json.load(open(f))["groups"].get(task)
        except Exception: continue
        if not g: continue
        rows = g.get("rows") or []
        sc = [r["score"] for r in rows if r.get("valid") and r.get("score") is not None]
        items.append((rd, len(rows), sc))
    items.sort()
    out, cum, best = [], 0, None
    for rd, n, sc in items:
        cum += n
        for s in sc:
            if s <= 0: continue
            best = s if best is None else max(best, s)
        if best is not None:
            out.append((cum, best))
    return {"campaign": out, "fixed": []}


def arm_curve(task, ws, need_analyst):
    """Controlled arm from context_ablation.sh: same start, own ratchet, own budget."""
    log = f"{R}/{ws}/driver.log"
    if not os.path.exists(log):
        return []
    txt = open(log, errors="ignore").read()
    rounds = [int(x) for x in re.findall(r"round(\d+) over", txt)]
    jobs = re.findall(r"job (\d+)", txt)
    L = "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm"
    pts, cum, best = [], 0, None
    for idx, rd in enumerate(rounds):
        f = f"{R}/outer/round{rd}/round_summary.json"
        if not os.path.exists(f):
            continue
        try: g = json.load(open(f))["groups"].get(task)
        except Exception: continue
        if not g: continue
        rows = g.get("rows") or []
        cum += len(rows)
        if need_analyst:
            job = jobs[idx] if idx < len(jobs) else None
            lf = f"{L}/sah-outer-{job}.out" if job else None
            nb = sum(1 for l in open(lf, errors="ignore") if "analysis brief attached" in l) \
                 if lf and os.path.exists(lf) else 0
            if nb == 0:
                continue
        for r in rows:
            s = r.get("score")
            if r.get("valid") and s and s > 0:
                best = s if best is None else max(best, s)
        if best is not None:
            pts.append((cum, best))
    return pts


def context_curve(task):
    """Controlled context-only arm: fixed proposer weights, analyst on, campaign-local
    ratchet, started from the fixed initial harness (context_v2)."""
    ws = f"{R}/context_v2"
    log = f"{ws}/driver.log"
    if not os.path.exists(log):
        return []
    txt = open(log, errors="ignore").read()
    rounds = [int(x) for x in re.findall(r"round(\d+) over", txt)]
    jobs = re.findall(r"job (\d+)", txt)
    L = "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm"
    pts, cum, best = [], 0, None
    for idx, rd in enumerate(rounds):
        job = jobs[idx] if idx < len(jobs) else None
        lf = f"{L}/sah-outer-{job}.out" if job else None
        briefs = 0
        if lf and os.path.exists(lf):
            briefs = sum(1 for l in open(lf, errors="ignore") if "analysis brief attached" in l)
        f = f"{R}/outer/round{rd}/round_summary.json"
        if not os.path.exists(f):
            continue
        try: g = json.load(open(f))["groups"].get(task)
        except Exception: continue
        if not g: continue
        rows = g.get("rows") or []
        cum += len(rows)
        if briefs == 0:      # round 1 has no analyst -> not the context condition
            continue
        for r in rows:
            s = r.get("score")
            if r.get("valid") and s and s > 0:
                best = s if best is None else max(best, s)
        if best is not None:
            pts.append((cum, best))
    return pts


TAGS = {"eft__math__erdos_min_overlap": "erdos_min", "eft__math__circle_packing": "circle_pa",
        "eft__math__hadamard_maximal_det": "hadamard_", "eft__math__first_autocorr_ineq": "first_aut",
        "eft__math__second_autocorr_ineq": "second_au", "eft__ahc_simpletes__ahc039": "ahc039"}


def ttt_curve(task):
    """[(cumulative rollouts, best-so-far)] from the iterative TTT run."""
    tag = TAGS.get(task, "")
    pts = []
    for f in (f"{TTT_DIR}/iter_{tag}/curve.jsonl", f"{TTT_DIR}/iter2_{tag}/curve.jsonl",
              f"{TTT_DIR}/iter3_{tag}/curve.jsonl"):
        if not os.path.exists(f):
            continue
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("best") is not None:
                pts.append((int(d["cum_rollouts"]), float(d["best"])))
    # best-so-far envelope over whatever budget we actually spent
    pts.sort()
    out, run = [], None
    for x, y in pts:
        run = y if run is None else max(run, y)
        out.append((x, run))
    return out


def normalize(task, v, lower):
    seed, ref = ANCHOR[task]
    if lower:                      # smaller is better -> flip
        return (seed - v) / (seed - ref) if seed != ref else 0.0
    return (v - seed) / (ref - seed) if ref != seed else 0.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi_map = round_phi_map()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), constrained_layout=True)
    fig.suptitle("Where should the reward go?  Updating the proposer vs. the executor vs. context alone",
                 fontsize=15)
    fig.text(0.5, -0.015,
             "x = executor rollouts spent by that arm (log).  Matched arms (green/grey) share start, budget and "
             "ratchet and differ only in what is updated;\nthe blue curve is the full observed campaign and is not "
             "budget-matched; the star is TTT-Discover's published Qwen3-8B result at its 25,600-rollout budget.",
             ha="center", fontsize=8.5, style="italic")

    for ax, (task, title, lower) in zip(axes.ravel(), TASKS):
        s = series(task, lower, phi_map)
        for kind, color, style, marker, label in [
            ("campaign", "#1f4e79", "-", "o", "Update proposer (ours, full campaign)"),
        ]:
            pts = s[kind]
            if not pts: continue
            xy = [(p[0], normalize(task, p[1], lower)) for p in pts]
            xy = [(x, y) for x, y in xy if y > -0.05]     # drop degenerate rollouts
            if not xy: continue
            xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
            ax.plot(xs, ys, style, color=color, marker=marker, ms=4.5, lw=2.2,
                    mfc="white" if kind == "fixed" else color, label=label)

        pa = arm_curve(task, "arm_proposer", False)
        if pa:
            ax.plot([p[0] for p in pa], [normalize(task, p[1], lower) for p in pa],
                    "-", color="#2e9e5b", marker="D", ms=5.5, lw=2.4,
                    label="Update proposer (matched arm)")
        cc = arm_curve(task, "arm_context_long", True) or context_curve(task)
        if cc:
            ax.plot([p[0] for p in cc], [normalize(task, p[1], lower) for p in cc],
                    "-.", color="#7f7f7f", marker="^", ms=5.5, lw=2.2, mfc="white",
                    label="Context only (matched arm)")

        tc = ttt_curve(task)
        if tc:
            ax.plot([p[0] for p in tc], [normalize(task, p[1], lower) for p in tc],
                    "--", color="#d67c1c", marker="s", ms=5.5, lw=2.2, mfc="white",
                    label="Update executor (local TTT-Discover-style)")

        pub = TTT_PUBLISHED.get(task)
        if pub is not None:
            ax.plot([TTT_BUDGET], [normalize(task, pub, lower)], marker="*", ms=16,
                    color="#b03a2e", ls="none",
                    label="TTT-Discover Qwen3-8B (published, 25.6k rollouts)")

        ax.axhline(1.0, color="black", ls=":", lw=1.2, alpha=.7)
        ax.text(0.015, 1.005, "published ≤10B best", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color="black", va="bottom")
        ax.set_title(title, fontsize=12)
        ax.set_xscale("log")
        ax.grid(alpha=.25, ls="--")
        ax.set_xlabel("cumulative executor rollouts (log)", fontsize=9.5)
        ax.set_ylim(-0.05, 1.25)
        if ax in (axes[0][0], axes[1][0]):
            ax.set_ylabel("normalized best validated score", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=.9)

    out = "/lustre/fsw/portfolios/av/users/yingzim/code/self_adapt_harness/papers/figures/score_compute_curves.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)

    for task, title, lower in TASKS:
        s = series(task, lower, phi_map)["campaign"]
        c = context_curve(task)
        tt = ttt_curve(task)
        def fmt(p):
            return f"{p[0]:5d} rollouts -> {normalize(task, p[1], lower):.3f}" if p else "n/a"
        print(f"  {title:22s} campaign: {fmt(s[-1] if s else None):28s}"
              f" context: {fmt(c[-1] if c else None):28s} TTT: {fmt(tt[-1] if tt else None)}")


if __name__ == "__main__":
    main()

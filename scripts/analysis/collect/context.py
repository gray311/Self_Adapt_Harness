#!/usr/bin/env python3
r"""Extract the \method (context) row from a context-ablation run.

Only counts rounds where the analyst actually fired: the analyst needs
prior-round feedback, so round 1 never has it, and a round without a brief is
not the (context) condition at all -- it is just a cold base proposer.  Each
round is verified against its Slurm log for "analysis brief attached".
"""
import json, os, re, subprocess, sys, glob

RUN = os.environ.get("RUN_ROOT") or "/lustre/fsw/portfolios/av/users/yingzim/runs"
R   = f"{RUN}/self_adapt_harness"
LOG = "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm"
WS  = sys.argv[1] if len(sys.argv) > 1 else f"{R}/context_v2"
LOWER = {"eft__math__erdos_min_overlap", "eft__math__first_autocorr_ineq"}

# display-scale conversions (verified against rollouts reporting both scales)
def to_display(t, c):
    if t == "eft__math__erdos_min_overlap":      return 0.380922 / c
    if t == "eft__math__first_autocorr_ineq":    return 1.505293 / c
    if t == "eft__math__second_autocorr_ineq":   return c * 0.896280
    if t == "eft__math__circle_packing":         return c * 2.635
    if t == "eft__ahc_simpletes__ahc039":        return c * 225_000
    if t == "eft__ahc_simpletes__ahc058":        return c * 4.5e8
    return c                                     # hadamard, eplb, prism, sql, txn

rounds = []
for line in open(os.path.join(WS, "driver.log")):
    m = re.search(r"round(\d+) over", line)
    if m: rounds.append(int(m.group(1)))
jobs = re.findall(r"job (\d+)", open(os.path.join(WS, "driver.log")).read())

best = {}
used = []
for idx, rd in enumerate(rounds):
    job = jobs[idx] if idx < len(jobs) else None
    f = f"{LOG}/sah-outer-{job}.out" if job else None
    briefs = 0
    if f and os.path.exists(f):
        briefs = sum(1 for l in open(f, errors="ignore") if "analysis brief attached" in l)
    if briefs == 0:
        print(f"  round{rd}: 0 analyst briefs -> SKIPPED (not the context condition)")
        continue
    s = f"{R}/outer/round{rd}/round_summary.json"
    if not os.path.exists(s):
        print(f"  round{rd}: {briefs} briefs but not collected yet"); continue
    used.append(rd)
    for t, v in json.load(open(s))["groups"].items():
        sc = v.get("best_score")
        if sc is None or sc == 0: continue
        # combined scores are higher-is-better on EVERY task (the display-scale
        # conversion below re-applies each metric's direction), so always take max
        best[t] = sc if t not in best else max(best[t], sc)
    print(f"  round{rd}: {briefs} analyst briefs -> USED")

print(f"\nrounds used: {used}   tasks covered: {len(best)}/11")
ORDER = ["eft__math__erdos_min_overlap","eft__math__first_autocorr_ineq",
 "eft__math__second_autocorr_ineq","eft__math__circle_packing",
 "eft__math__hadamard_maximal_det","eft__ahc_simpletes__ahc039",
 "eft__ahc_simpletes__ahc058","adrs__eplb","adrs__prism","adrs__llm_sql",
 "adrs__txn_scheduling"]
cells = []
for t in ORDER:
    cells.append(f"{to_display(t, best[t]):.6g}" if t in best else "--")
print("\nLaTeX row:")
print("& " + " & ".join(cells[:5]) + "\n& " + " & ".join(cells[5:]) + r" \\")

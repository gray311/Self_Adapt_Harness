#!/usr/bin/env python3
"""Build the cross-task transfer matrix from the rows submitted by
cross_task_transfer.sh.

Cell (i,j) = direction-normalized improvement of source adapter phi_i on target
task tau_j over the BASE proposer phi_0 on the same task, so a positive number
means "this adapter proposes better harnesses for that task than the untrained
proposer does".  Minimized tasks are sign-flipped so >0 is always better.
"""
import json, os, sys, glob

OUT = os.path.expandvars("$RUN_ROOT/self_adapt_harness/outer")
WS  = os.path.expandvars("$RUN_ROOT/self_adapt_harness/cross_task")
LOWER = {"eft__math__erdos_min_overlap", "eft__math__first_autocorr_ineq"}
SHORT = {"eft__math__erdos_min_overlap":"Erd","eft__math__first_autocorr_ineq":"AC1",
 "eft__math__second_autocorr_ineq":"AC2","eft__math__circle_packing":"CP",
 "eft__math__hadamard_maximal_det":"Had","eft__ahc_simpletes__ahc039":"a039",
 "eft__ahc_simpletes__ahc058":"a058","adrs__eplb":"EPLB","adrs__prism":"PRI",
 "adrs__llm_sql":"SQL","adrs__txn_scheduling":"Txn"}

# a source row may be split across two half-jobs (different round ids); merge them
merged = {}
order = []
for fn in ("rows.txt", "rows2.txt"):
    p = os.path.join(WS, fn)
    if not os.path.exists(p):
        continue
    for line in open(p):
        parts = line.split()
        if len(parts) < 3:
            continue
        src, rnd = parts[0], parts[1]
        f = os.path.join(OUT, f"round{int(rnd):03d}", "round_summary.json")
        if not os.path.exists(f):
            print(f"  (round{rnd} for {src} not collected yet)", file=sys.stderr); continue
        g = json.load(open(f))["groups"]
        if src not in merged:
            merged[src] = {}; order.append(src)
        for t, v in g.items():
            s = v.get("best_score")
            if s is None: continue
            merged[src][t] = s if t not in merged[src] else max(merged[src][t], s)
rows = [(s, merged[s]) for s in order if merged.get(s)]

if not rows:
    sys.exit("no collected rows yet")
base = next((dict(sc) for src, sc in rows if src == "BASE"), {})
if not base:
    print("  (BASE row not collected yet -- cells cannot be normalized)", file=sys.stderr)
tasks = [t for t in SHORT if any(t in r[1] for r in rows)]

print("source".ljust(22) + "".join(SHORT[t].rjust(8) for t in tasks) + "    mean")
for src, sc in rows:
    cells, out = [], []
    for t in tasks:
        v, b = sc.get(t), base.get(t)
        # a zero score means the rollout could not run at all (e.g. the AHC rows
        # submitted before the native-tester env was set); that is an
        # infrastructure failure, not a transfer result, so leave the cell blank
        if v is None or b is None or b == 0 or v == 0:
            out.append("   --"); continue
        d = (v - b) / abs(b)   # combined scores: higher is better on every task
        cells.append(d); out.append(f"{100*d:+7.1f}")
    m = f"{100*sum(cells)/len(cells):+7.1f}" if cells else "     --"
    print(src.replace("mphi_f_", "").ljust(22) + "".join(out) + m)
print("\ncells are % improvement over the BASE proposer on the same task; >0 = better.")

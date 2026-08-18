#!/usr/bin/env python3
"""One-glance status of the three arms: budget spent, best-so-far, and whether
the arm has stopped improving (plateau) -- so we can tell when each has been
pushed to its own ceiling rather than just cut short."""
import json, os, re, glob
RUN = os.environ.get("RUN_ROOT") or "/lustre/fsw/portfolios/av/users/yingzim/runs"
R = f"{RUN}/self_adapt_harness"
L = "/lustre/fsw/portfolios/av/users/yingzim/logs/slurm"
SIX = [("eft__math__erdos_min_overlap","Erd"),("eft__math__first_autocorr_ineq","AC1"),
       ("eft__math__second_autocorr_ineq","AC2"),("eft__math__circle_packing","CP"),
       ("eft__math__hadamard_maximal_det","Had"),("eft__ahc_simpletes__ahc039","a039")]
TAGS = {"eft__math__erdos_min_overlap":"erdos_min","eft__math__circle_packing":"circle_pa",
        "eft__math__hadamard_maximal_det":"hadamard_","eft__math__first_autocorr_ineq":"first_aut",
        "eft__math__second_autocorr_ineq":"second_au","eft__ahc_simpletes__ahc039":"ahc039"}

def arm(ws, task, need_analyst):
    log = f"{R}/{ws}/driver.log"
    if not os.path.exists(log): return []
    txt = open(log, errors="ignore").read()
    rounds = [int(x) for x in re.findall(r"round(\d+) over", txt)]
    jobs = re.findall(r"job (\d+)", txt)
    pts, cum, best = [], 0, None
    for i, rd in enumerate(rounds):
        f = f"{R}/outer/round{rd}/round_summary.json"
        if not os.path.exists(f): continue
        try: g = json.load(open(f))["groups"].get(task)
        except Exception: continue
        if not g: continue
        rows = g.get("rows") or []; cum += len(rows)
        if need_analyst:
            lf = f"{L}/sah-outer-{jobs[i]}.out" if i < len(jobs) else None
            if not (lf and os.path.exists(lf) and any("analysis brief attached" in l
                    for l in open(lf, errors="ignore"))): continue
        for r in rows:
            s = r.get("score")
            if r.get("valid") and s and s > 0: best = s if best is None else max(best, s)
        if best is not None: pts.append((cum, best))
    return pts

def ttt(task):
    pts = []
    for pat in ("iter_", "iter2_", "iter3_", "iter4_"):
        f = f"{R}/ttt_arm/{pat}{TAGS[task]}/curve.jsonl"
        if not os.path.exists(f): continue
        for l in open(f):
            try: d = json.loads(l)
            except Exception: continue
            if d.get("best") is not None: pts.append((int(d["cum_rollouts"]), float(d["best"])))
    pts.sort(); out, run = [], None
    for x, y in pts:
        run = y if run is None else max(run, y); out.append((x, run))
    return out

def plateau(pts, k=2):
    """no improvement over the last k recorded points"""
    if len(pts) <= k: return ""
    return "  PLATEAU" if pts[-1][1] <= pts[-1-k][1] + 1e-12 else ""

print(f"{'task':6s} {'arm':22s} {'rollouts':>9s} {'best':>12s}")
for task, short in SIX:
    for name, pts in (("proposer (matched)", arm("arm_proposer", task, False)),
                      ("context  (matched)", arm("arm_context_long", task, True)),
                      ("executor (TTT)",     ttt(task))):
        if pts: print(f"{short:6s} {name:22s} {pts[-1][0]:9d} {pts[-1][1]:12.6f}{plateau(pts)}")
        else:   print(f"{short:6s} {name:22s} {'--':>9s} {'--':>12s}")

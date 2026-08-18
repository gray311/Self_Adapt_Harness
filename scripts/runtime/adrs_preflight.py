#!/usr/bin/env python3
"""Preflight the ADRS System-Performance seeds in the container."""
import importlib.util, os, sys, traceback

A = "/lustre/fsw/portfolios/av/users/yingzim/code/evolution-fine-tuning/skydiscover/benchmarks/ADRS"
SHIM = "/lustre/fsw/portfolios/av/users/yingzim/datasets/self_adapt_harness/runtime/skydiscover_min"

for t in ["prism", "txn_scheduling", "llm_sql", "eplb"]:
    tdir = os.path.join(A, t)
    ev = os.path.join(tdir, "evaluator", "evaluator.py")
    ip = os.path.join(tdir, "initial_program.py")
    if not os.path.exists(ip):
        ip = os.path.join(tdir, "initial_program_naive.py")
    print(f"===== {t} =====  eval={os.path.exists(ev)} seed={os.path.basename(ip)}")
    saved = list(sys.path)
    for p in (SHIM, os.path.join(tdir, "evaluator"), tdir):
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location(f"ev_{t}", ev)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        r = m.evaluate(ip)
        r = getattr(r, "__dict__", r) if not isinstance(r, dict) else r
        if isinstance(r, dict):
            cs = r.get("combined_score", (r.get("metrics") or {}).get("combined_score"))
            print(f"  -> combined_score={cs}  error={r.get('error')}  keys={list(r.keys())[:8]}")
        else:
            print(f"  -> returned {type(r)}: {r}")
    except Exception:
        print("  -> EXC:")
        traceback.print_exc()
    finally:
        sys.path[:] = saved
        for mod in list(sys.modules):
            if mod.startswith("program") or mod == f"ev_{t}":
                sys.modules.pop(mod, None)

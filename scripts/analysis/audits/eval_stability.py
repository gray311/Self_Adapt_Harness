#!/usr/bin/env python3
"""Re-evaluation stability check for the campaign's reported best programs.

For every task in best_programs.json, re-run the official evaluator N times on
the banked best program and report mean/std/min/max — validating that reported
numbers are not single-eval noise (plan.md §12.2 evidence).

Usage: eval_stability.py <best_programs.json> <out.json> [n_runs]
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from inner.tasks.eft_task import get_task            # noqa: E402
from inner.evaluation.eval_runner import evaluate_program  # noqa: E402


def main() -> None:
    bp_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    bp = json.loads(bp_path.read_text())
    report = {}
    for tid, ent in sorted(bp.items()):
        prog = ent.get("program")
        if not prog:
            continue
        try:
            task = get_task(tid)
        except Exception as e:
            report[tid] = {"error": f"task load: {e}"}
            continue
        scores, walls, errors = [], [], 0
        for i in range(n_runs):
            t0 = time.time()
            out = evaluate_program(task, prog, timeout_s=420)
            walls.append(time.time() - t0)
            if out.error:
                errors += 1
            else:
                scores.append(out.combined_score)
            print(f"[{tid}] run {i+1}/{n_runs}: "
                  f"{out.combined_score:.6g} ({walls[-1]:.0f}s)"
                  + (f" ERR={out.error[:60]}" if out.error else ""), flush=True)
        rep = {"reported": ent.get("score"), "n": len(scores), "errors": errors}
        if scores:
            rep.update({
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores), "max": max(scores),
            })
        report[tid] = rep
        out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

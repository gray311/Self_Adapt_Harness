#!/usr/bin/env python3
"""Successive-halving promotion: rank each task's candidates by their best
screening score and print the top-N as "task_id:k" lines (consumed by the
worker's cascade pass 2).

Usage: cascade_promote.py <round_dir> <top_n>
"""
import glob
import json
import sys


def main() -> None:
    round_dir, top = sys.argv[1], int(sys.argv[2])
    best = {}
    for f in glob.glob(f"{round_dir}/rollouts/*/cand*/*/results/*.json") + \
             glob.glob(f"{round_dir}/rollouts/*/cand*/*/checkpoints/*.json"):
        parts = f.split("/rollouts/")[1].split("/")
        tid, k = parts[0], int(parts[1].replace("cand", ""))
        try:
            s = json.load(open(f)).get("best_score")
        except Exception:
            continue
        if s is not None:
            key = (tid, k)
            best[key] = max(best.get(key, float("-inf")), float(s))

    by_task = {}
    for (tid, k), s in best.items():
        by_task.setdefault(tid, []).append((s, k))
    for tid, lst in by_task.items():
        for s, k in sorted(lst, reverse=True)[:top]:
            print(f"{tid}:{k}")


if __name__ == "__main__":
    main()

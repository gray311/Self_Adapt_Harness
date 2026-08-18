#!/usr/bin/env python3
"""Drop infrastructure-poisoned rows from a round's grpo_batch.jsonl.

A row is poisoned when its rollout never reached the LLM (llm_calls == 0 with
stop_reason harness_error — e.g. the replica it was routed to was dead). Such
rewards are pure infrastructure noise, unrelated to candidate quality, so the
row is removed and GRPO advantages are recomputed over the surviving group.

Usage: sanitize_grpo_batch.py <round_dir>
Idempotent; rewrites grpo_batch.jsonl in place (backup: grpo_batch.jsonl.orig).
"""
import json
import math
import shutil
import sys
from pathlib import Path

EPS = 1e-9


def poisoned_ks(round_dir: Path, task_id: str) -> set:
    bad = set()
    for res in (round_dir / "rollouts" / task_id).glob("cand*/*/results/*.json"):
        try:
            d = json.loads(res.read_text())
        except Exception:
            continue
        led = d.get("ledger") or {}
        if (led.get("llm_calls") or 0) == 0 and d.get("stop_reason") == "harness_error":
            bad.add(int(res.parents[2].name.replace("cand", "")))
    return bad


def main() -> None:
    round_dir = Path(sys.argv[1])
    batch = round_dir / "grpo_batch.jsonl"
    rows = [json.loads(l) for l in batch.read_text().splitlines() if l.strip()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    kept, dropped = [], []
    for tid, group in by_task.items():
        bad = poisoned_ks(round_dir, tid)
        clean = [r for r in group if r["k"] not in bad]
        dropped += [r for r in group if r["k"] in bad]
        if len(clean) < 2:  # can't form a group; keep as-is rather than starve
            kept += group
            continue
        rewards = [r["reward"] for r in clean]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
        for r in clean:
            r["advantage"] = (r["reward"] - mean) / (std + EPS)
            r["group_size"] = len(clean)
        kept += clean

    if not dropped:
        print("[sanitize] no poisoned rows; grpo_batch untouched")
        return
    if not batch.with_suffix(".jsonl.orig").exists():
        shutil.copy2(batch, batch.with_suffix(".jsonl.orig"))
    batch.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"[sanitize] dropped {len(dropped)} poisoned row(s) "
          f"(k={[r['k'] for r in dropped]}), kept {len(kept)}; "
          "advantages recomputed")


if __name__ == "__main__":
    main()

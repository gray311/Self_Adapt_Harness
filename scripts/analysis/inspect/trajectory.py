#!/usr/bin/env python3
"""Pretty-print the full M_phi+H1 -> H2 -> inner chain for a round's candidates.

Usage:
  python3 scripts/analysis/inspect/trajectory.py <round_dir> [task_id] [k]

With no k: lists every candidate's one-line summary.
With a k: dumps that candidate's full chain —
  1. the H1 user message (task context + inheritance + feedback + analyst notes)
  2. the H1 Cordis trajectory (M_phi designing the package, tool call by tool call)
  3. the submitted spec (raw YAML)
  4. generated plugins + deterministic validation audit
  5. the materialized cordis.yml plugin set (proof of structural divergence)
  6. the inner rollout outcome (score, evals, whether custom tools were called)
"""
import json
import sys
from pathlib import Path


def _text(content):
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def main() -> None:
    rd = Path(sys.argv[1])
    meta = json.loads((rd / "round.json").read_text())
    trajs = {(t["task_id"], t["k"]): t
             for t in json.loads((rd / "trajectories.json").read_text())}
    prompts = json.loads((rd / "prompts.json").read_text())

    tasks = meta["tasks_order"]
    if len(sys.argv) >= 3:
        tasks = [sys.argv[2]]

    if len(sys.argv) < 4:  # summary mode
        for tid in tasks:
            print(f"\n=== {tid} ===")
            for c in meta["per_task"][tid]["candidates"]:
                rl = c.get("review_log") or []
                tools = ",".join(
                    f"{x.get('kind', 'component')}:{x.get('name', '?')}/"
                    f"{x.get('status', 'unknown')}" for x in rl
                )
                changed = [f.split(".")[-1] for f in c.get("changed_fields", [])]
                print(f"  cand{c['k']:02d} valid={int(c['valid'])} "
                      f"changed={changed} tools=[{tools}] "
                      f"stop={c.get('stop_reason')}")
        print("\nrun with a k to see one candidate's full chain.")
        return

    tid, k = sys.argv[2], int(sys.argv[3])
    cdir = rd / "tasks" / tid / f"cand{k:02d}"
    cand = next(c for c in meta["per_task"][tid]["candidates"] if c["k"] == k)

    print("#" * 78)
    print(f"# {tid}  cand{k:02d}   round {meta['round']}   valid={cand['valid']}")
    print("#" * 78)

    print("\n===== 1. H1 USER MESSAGE (what M_phi was conditioned on) =====")
    print(prompts.get(tid, "")[:6000])

    print("\n===== 2. H1 TRAJECTORY (M_phi designing the harness) =====")
    for i, m in enumerate(trajs.get((tid, k), {}).get("trajectory", [])):
        role = m.get("role", "?")
        tc = [x["function"]["name"] for x in (m.get("tool_calls") or [])]
        if isinstance(m.get("content"), list):
            tc.extend(
                str(x.get("name")) for x in m["content"]
                if isinstance(x, dict) and x.get("type") == "tool_use"
            )
        body = _text(m.get("content"))
        head = f"[{i}] {role}"
        if tc:
            print(f"{head}  ->tools {tc}")
        if body.strip():
            print(f"{head}: {body.strip()[:1400]}")

    print("\n===== 3. SUBMITTED SPEC (raw) =====")
    print((trajs.get((tid, k), {}).get("raw_submission") or "")[:4000])

    print("\n===== 4. COMPONENT VALIDATION LOG (no post-submit repair) =====")
    for r in (cand.get("review_log") or []):
        print(f"  {r.get('kind', 'component')}:{r.get('name')}: "
              f"status={r.get('status')} runtime={r.get('runtime')} "
              f"error={r.get('error')}")
        for h in r.get("history", []):
            print(f"      {h}")

    if (cdir / "plugins").exists():
        print("\n===== 4b. CORDIS PLUGIN CODE =====")
        for f in sorted((cdir / "plugins").glob("*.mjs")):
            print(f"--- {f.name} ---")
            print(f.read_text())

    if (cdir / "cordis.yml").exists():
        import yaml
        cfg = yaml.safe_load((cdir / "cordis.yml").read_text()) or []
        print("\n===== 5. MATERIALIZED cordis.yml PLUGIN SET =====")
        for operation in cfg:
            if not isinstance(operation, dict):
                continue
            for row in operation.get("insert") or []:
                sah = ((row.get("config") or {}).get("sah") or {})
                print(f"  {str(sah.get('kind', 'plugin')):12s} "
                      f"{str(sah.get('name', row.get('id', '?'))):24s} "
                      f"-> {row.get('name')}")

    print("\n===== 6. INNER ROLLOUT OUTCOME =====")
    hits = list((rd / "rollouts" / tid / f"cand{k:02d}").glob("*/results/*.json"))
    if not hits:
        hits = list((rd / "rollouts" / tid / f"cand{k:02d}").glob("*/checkpoints/*.json"))
    if hits:
        d = json.loads(sorted(hits)[-1].read_text())
        led = d.get("ledger") or {}
        print(f"  best_score={d.get('best_score')} evals={led.get('evaluator_calls')} "
              f"probes={led.get('probe_calls')} stop={d.get('stop_reason')}")
        notes = [s for s in (d.get("steps") or [])
                 if isinstance(s, dict) and s.get("kind") == "note"]
        if notes:
            print(f"  custom-tool notes: {len(notes)}")
            for n in notes[:5]:
                print(f"    - {n.get('edit_note','')[:120]}")
    else:
        print("  (no rollout output yet)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unified evolve runner / live campaign viewer.

One entrypoint, three subcommands, consistent timestamped logging:

  watch   — follow a RUNNING campaign in real time (safe, read-only):
              python3 src/runner.py watch  <namespace> [--interval 15]
  status  — one-shot snapshot of the same view:
              python3 src/runner.py status <namespace>
  round   — execute ONE evolve round's four stages in sequence with unified
            logging (debug use; expects the round job environment: snapshot
            sourced, vLLM pools up, CURVE_* env set — the normal path is
            still drive_cp_curve.sh -> round.sbatch):
              python3 src/runner.py round --round-dir <dir> --k 8 ...

The watcher reads only round artifacts (current_phase, gate audits, rollout
results, curve.jsonl, driver.log) — it never touches campaign state.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("sah")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# shared campaign inspection
# --------------------------------------------------------------------------- #

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _task_dir(ns: Path) -> Path:
    """The campaign's task directory, discovered rather than assumed.

    The tag is set by the campaign (SAH_TASK_TAG / the route layout), so
    hardcoding one task's name blinds the viewer on every other task.
    """
    route = ns / "update_harness"
    if not route.is_dir():
        return ns
    candidates = [d for d in sorted(route.iterdir())
                  if d.is_dir() and (d / "curve.jsonl").is_file()]
    if not candidates:
        candidates = [d for d in sorted(route.iterdir())
                      if d.is_dir() and (d / "rounds").is_dir()]
    return candidates[0] if candidates else route


def campaign_state(ns: Path) -> dict:
    """Everything the live view needs, from artifacts only."""
    task_dir = _task_dir(ns)
    state: dict = {"ns": str(ns), "rounds": [], "driver": None}
    curve = []
    cpath = task_dir / "curve.jsonl"
    if cpath.is_file():
        curve = [json.loads(l) for l in cpath.read_text().splitlines()
                 if l.strip()]
    state["curve"] = [
        {"x": r["x"], "score": round(r["score"], 4),
         "delta": round(r.get("best_causal_delta") or 0, 4),
         "repaired": len((r.get("proposal_slots") or {}).get("repaired") or [])}
        for r in curve
    ]
    for rd in sorted((task_dir / "rounds").glob("round0*")):
        row: dict = {"name": rd.name}
        phase = rd / "runtime" / "current_phase"
        if phase.is_file():
            kv = dict(l.split("=", 1) for l in phase.read_text().splitlines()
                      if "=" in l)
            row["phase"] = kv.get("phase")
            row["job"] = kv.get("job_id")
        row["complete"] = (rd / "ROUND_COMPLETE").is_file()
        gate = _read_json(rd / "proposal_gate_audit.json")
        if gate:
            t = list(gate["tasks"].values())[0]
            row["gate"] = f"{t['valid_slots']}/8 valid, {t['repaired_slots']} repaired"
        for side, dname in (("cand", "rollouts"), ("ctrl", "paired_controls")):
            root = rd / dname
            if root.is_dir():
                row[side] = sum(1 for _ in root.glob("*/cand*/*/results/*.json"))
        tr = rd / "training"
        if (tr / "TRAIN_COMPLETE").is_file():
            txt = (tr / "TRAIN_COMPLETE").read_text()
            row["train"] = ("skipped-zerograd" if "zero_gradient" in txt
                            else "complete")
        elif tr.is_dir():
            row["train"] = "running"
        state["rounds"].append(row)
    dlog = ns / "driver.log"
    if dlog.is_file():
        tail = dlog.read_text().splitlines()[-3:]
        state["driver"] = [l.split("[driver] ")[-1] for l in tail]
    return state


def render(state: dict, last: dict | None) -> dict:
    """Log only what changed since the previous poll; return fingerprints."""
    fp: dict = {}
    for pt in state["curve"]:
        key = f"curve:{pt['x']}"
        fp[key] = json.dumps(pt)
        if not last or last.get(key) != fp[key]:
            log.info("[curve] x%-2d score=%.4f  delta=%+.4f  repaired=%d",
                     pt["x"], pt["score"], pt["delta"], pt["repaired"])
    for row in state["rounds"]:
        if row.get("complete"):
            key = f"{row['name']}:done"
            fp[key] = "1"
            if not last or key not in last:
                log.info("[round] %s COMPLETE  (%s)", row["name"],
                         row.get("gate", "gate n/a"))
            tkey = f"{row['name']}:train"
            fp[tkey] = row.get("train") or ""
            if row.get("train") and (not last or last.get(tkey) != fp[tkey]):
                log.info("[train] %s: %s", row["name"], row["train"])
            continue
        key = f"{row['name']}:live"
        parts = [f"phase={row.get('phase')}"]
        if row.get("gate"):
            parts.append(f"gate={row['gate']}")
        if row.get("cand") is not None:
            parts.append(f"rollouts cand={row.get('cand', 0)} "
                         f"ctrl={row.get('ctrl', 0)}")
        fp[key] = " ".join(parts)
        if not last or last.get(key) != fp[key]:
            log.info("[round] %s  %s", row["name"], fp[key])
    if state.get("driver"):
        key = "driver"
        fp[key] = state["driver"][-1]
        if not last or last.get(key) != fp[key]:
            log.info("[driver] %s", state["driver"][-1])
    return fp


def cmd_watch(args) -> None:
    ns = Path(args.namespace).resolve()
    log.info("[watch] following %s (interval %ss, Ctrl-C to stop)",
             ns.name, args.interval)
    last: dict | None = None
    while True:
        try:
            last = render(campaign_state(ns), last)
        except Exception as e:
            log.warning("[watch] poll failed: %s", e)
        if args.once:
            return
        time.sleep(args.interval)


# --------------------------------------------------------------------------- #
# one evolve round as a single logged sequence (debug entrypoint)
# --------------------------------------------------------------------------- #

STAGES = (
    ("propose", ["-m", "outer.rounds.outer_round", "propose"]),
    ("slot-plan", ["-m", "outer.reward.trajectory_budget"]),
    ("collect", ["-m", "outer.rounds.outer_round", "collect"]),
)


def cmd_round(args) -> None:
    """Run propose -> (external rollouts) -> collect with tagged streaming.

    Rollout launching stays with the worker (it owns seeds/ports/pairing);
    this sequencer is for driving/debugging the python stages by hand.
    """
    base = [sys.executable]
    for name, argv in STAGES:
        if name == "slot-plan" and args.skip_plan:
            continue
        cmd = base + argv + args.stage_args
        log.info("[%s] exec: %s", name, " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.info("[%s] %s", name, line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            log.error("[%s] FAILED rc=%s", name, proc.returncode)
            raise SystemExit(proc.returncode)
        log.info("[%s] done", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="follow a running campaign live")
    w.add_argument("namespace")
    w.add_argument("--interval", type=int, default=15)
    w.add_argument("--once", action="store_true")
    w.set_defaults(fn=cmd_watch, once=False)
    s = sub.add_parser("status", help="one-shot campaign snapshot")
    s.add_argument("namespace")
    s.set_defaults(fn=cmd_watch, interval=0, once=True)
    r = sub.add_parser("round", help="run one round's python stages, logged")
    r.add_argument("--skip-plan", action="store_true")
    r.add_argument("stage_args", nargs=argparse.REMAINDER,
                   help="arguments passed through to every stage")
    r.set_defaults(fn=cmd_round)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

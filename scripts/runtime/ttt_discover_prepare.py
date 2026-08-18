#!/usr/bin/env python3
"""Prepare one budget-scaled TTT-Discover executor update.

The input is one directory containing K fixed-harness executor rollouts
(``k*/<timestamp>/summary.json``).  This command:

* records the batch in a persistent per-task state/archive;
* computes TTT-Discover's adaptive-beta LOO entropic advantages;
* emits executor replay rows for the host-submitted LoRA trainer;
* appends an auditable score/rollout point to ``curve.jsonl``; and
* selects the next program parent with the official PUCT score form.

It deliberately does not submit Slurm jobs.  Training and merge must be
submitted from the host, never from inside a Pyxis container.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from cordis_runtime.config import system_persona

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "src" / "inner" / "harness" / "tools"

# The PUCT root is ``task.initial_program``.  Its value must therefore be the
# evaluator score of that exact program, not the best score found by a separate
# H2 trajectory.  Confusing those two quantities can make PUCT repeatedly pick
# an unevaluated/weak root merely because it was assigned H2's stronger score.
ROOT_PROGRAM_SCORES = {
    "eft__math__erdos_min_overlap": 0.7694488256878874,
    "eft__math__circle_packing": 0.36423689449571406,
    "eft__math__hadamard_maximal_det": 0.14327485380116958,
    "eft__math__first_autocorr_ineq": 0.991407005765425,
    "eft__math__second_autocorr_ineq": 0.9550495052953716,
    "eft__ahc_simpletes__ahc039": 2.474696296296296,
    "eft__ahc_simpletes__ahc058": 0.0,
    "adrs__eplb": 0.1263734727283441,
    "adrs__prism": 21.891622105209393,
    "adrs__llm_sql": 0.0,
    "adrs__txn_scheduling": 2638.5224274406332,
}

# One historical base-model/fixed-H2 trajectory (20 evaluator calls) is shared
# by all three reward-route arms as the pre-adaptation warm-start measurement.
# It is legitimate in the best-so-far curve, but its discovered program is NOT
# inherited by any arm and its score must not be attached to the PUCT root.
H2_WARM_START_SCORES = {
    "eft__math__erdos_min_overlap": 0.8342771200086017,
    "eft__math__circle_packing": 0.5608225468207615,
    "eft__math__hadamard_maximal_det": 0.14327485380116958,
    "eft__math__first_autocorr_ineq": 0.9914694359272692,
    "eft__math__second_autocorr_ineq": 0.9997888936813033,
    "eft__ahc_simpletes__ahc039": 2.476554,
    "eft__ahc_simpletes__ahc058": 0.298859,
    "adrs__eplb": 0.1265392786992853,
    "adrs__prism": 24.021666874072007,
    "adrs__llm_sql": 0.09343955531989306,
    "adrs__txn_scheduling": 3610.1083032490974,
}
TOPK_CHILDREN = 2


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def tool_schemas() -> list[dict[str, Any]]:
    out = []
    for path in sorted(TOOLS_DIR.glob("*.tool.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        out.append({"type": "function", "function": {
            "name": doc["name"],
            "description": doc.get("description", ""),
            "parameters": doc.get("input_schema", {}),
        }})
    return out


def task_info(task_id: str):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from inner.tasks.eft_task import get_task
    return get_task(task_id)


def load_batch(round_dir: Path, task_id: str) -> list[dict[str, Any]]:
    """Load at most one result per launched k directory."""
    rows = []
    kdirs = [p for p in round_dir.glob("k*")
             if p.is_dir() and p.name[1:].isdigit()]
    for kd in sorted(kdirs, key=lambda p: int(p.name[1:])):
        candidates = sorted(kd.glob("*/summary.json"))
        if not candidates:
            continue
        try:
            payload = json.loads(candidates[-1].read_text())
        except Exception:
            continue
        payload = payload if isinstance(payload, list) else [payload]
        row = next((x for x in payload if x.get("task_id") == task_id), None)
        if not row:
            continue
        score, program = row.get("best_score"), row.get("best_program")
        if score is None or not program:
            continue
        rows.append({
            "k": int(kd.name[1:]),
            "score": float(score),
            "program": str(program),
            "evaluations": int(row.get("evaluations") or 0),
            "summary": str(candidates[-1]),
        })
    return rows


def kl_to_uniform(rewards: list[float], beta: float) -> float:
    k = len(rewards)
    mx = max(rewards)
    logits = [beta * (r - mx) for r in rewards]
    z = sum(math.exp(x) for x in logits)
    logz = math.log(z)
    return sum(math.exp(x - logz) * (x - logz + math.log(k)) for x in logits)


def adaptive_entropic_advantages(rewards: list[float]) -> tuple[float, list[float]]:
    """Match ttt_discover.rl.train.compute_advantages exactly in scalar form."""
    if len(rewards) < 2 or max(rewards) == min(rewards):
        return 0.0, [0.0 for _ in rewards]
    target, lo, hi = math.log(2.0), 0.0, 1.0
    while hi < 1e6 and kl_to_uniform(rewards, hi) < target:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_to_uniform(rewards, mid) < target:
            lo = mid
        else:
            hi = mid
    beta = hi
    mx = max(rewards)
    weights = [math.exp(beta * (r - mx)) for r in rewards]
    total, k = sum(weights), len(weights)
    adv = [w / max((total - w) / (k - 1), 1e-12) - 1.0 for w in weights]
    return beta, adv


def program_hash(program: str) -> str:
    return hashlib.sha256(program.encode()).hexdigest()


def root_state(task, root_score: float | None = None) -> dict[str, Any]:
    program = task.initial_program
    return {
        "id": "root", "program": program, "program_hash": program_hash(program),
        "score": (
            float(root_score)
            if root_score is not None else ROOT_PROGRAM_SCORES[task.task_id]
        ),
        "score_semantics": "score of task.initial_program",
        "visits": 0, "max_child": None,
        "created_step": -1,
    }


def migrate_archive(state: dict[str, Any], task) -> None:
    """Bring pre-release state in line with official top-k PUCT buffering."""
    archive = state["archive"]
    root = archive["root"]
    if root.get("score") is None:
        root["score"] = ROOT_PROGRAM_SCORES[task.task_id]

    # Official PUCTSampler._filter_topk_per_parent defaults to two children.
    # Preserve only descendants of retained parents so there are no orphans.
    children: dict[str, list[dict[str, Any]]] = {}
    for node in archive.values():
        parent_id = node.get("parent_id")
        if parent_id is not None:
            children.setdefault(parent_id, []).append(node)
    keep = {"root"}
    frontier = ["root"]
    while frontier:
        parent_id = frontier.pop()
        ranked = sorted(children.get(parent_id, []),
                        key=lambda n: (float(n.get("score") or float("-inf")), n["id"]),
                        reverse=True)[:TOPK_CHILDREN]
        for node in ranked:
            if node["id"] not in keep:
                keep.add(node["id"])
                frontier.append(node["id"])
    state["archive"] = {sid: node for sid, node in archive.items() if sid in keep}


def update_archive(
    state: dict[str, Any], rows: list[dict[str, Any]], parent_id: str,
    step: int, checkpoint: str,
) -> None:
    archive = state["archive"]
    if parent_id not in archive:
        raise SystemExit(f"parent {parent_id!r} is absent from archive")
    parent = archive[parent_id]
    # TTT-Discover increments the selected state and all of its ancestors.
    ancestor = parent
    seen_ancestors: set[str] = set()
    while ancestor and ancestor["id"] not in seen_ancestors:
        seen_ancestors.add(ancestor["id"])
        ancestor["visits"] = int(ancestor.get("visits", 0)) + 1
        ancestor = archive.get(ancestor.get("parent_id"))
    batch_best = max(r["score"] for r in rows)
    prev = parent.get("max_child")
    parent["max_child"] = batch_best if prev is None else max(float(prev), batch_best)
    seen = {v["program_hash"] for v in archive.values()}
    retained = []
    for row in sorted(rows, key=lambda x: x["score"], reverse=True):
        ph = program_hash(row["program"])
        if ph in seen:
            continue
        retained.append(row)
        if len(retained) >= TOPK_CHILDREN:
            break
    for row in retained:
        ph = program_hash(row["program"])
        sid = f"s{step:02d}-{row['k']:02d}-{ph[:10]}"
        archive[sid] = {
            "id": sid, "program": row["program"], "program_hash": ph,
            "score": row["score"], "parent_id": parent_id, "visits": 0,
            "max_child": None, "created_step": step, "k": row["k"],
            "source_summary": row["summary"],
            "executor_checkpoint": checkpoint,
        }
        seen.add(ph)
    state["total_expansions"] = int(state.get("total_expansions", 0)) + 1


def select_puct(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    nodes = list(state["archive"].values())
    scored = [n for n in nodes if n.get("score") is not None]
    candidates = scored or nodes
    vals = [float(n.get("score") or 0.0) for n in candidates]
    # Official sampler excludes initial states when computing reward scale, but
    # includes them in the rank prior and candidate list.
    non_initial = [float(n["score"]) for n in candidates
                   if int(n.get("created_step", -1)) >= 0]
    scale_values = non_initial or vals
    scale = max(max(scale_values) - min(scale_values), 1e-6)
    order = sorted(range(len(candidates)), key=lambda i: vals[i], reverse=True)
    rank_weight = {idx: len(candidates) - rank for rank, idx in enumerate(order)}
    denom = sum(rank_weight.values())
    total = int(state.get("total_expansions", 0))
    best_node, best_stats = None, None
    for i, node in enumerate(candidates):
        visits = int(node.get("visits", 0))
        reward = float(node.get("score") or 0.0)
        q = float(node.get("max_child")) if visits and node.get("max_child") is not None else reward
        prior = rank_weight[i] / denom
        bonus = scale * prior * math.sqrt(1.0 + total) / (1.0 + visits)
        stats = {"q": q, "prior": prior, "bonus": bonus, "puct": q + bonus}
        if best_stats is None or (stats["puct"], reward) > (best_stats["puct"], float(best_node.get("score") or 0.0)):
            best_node, best_stats = node, stats
    assert best_node is not None and best_stats is not None
    return best_node, best_stats


def make_replay(task, rows: list[dict[str, Any]], advantages: list[float], beta: float,
                parent: dict[str, Any], step: int, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user = (f"# Task\n{task.spec}\n\n# Current program\n```{task.language}\n"
            f"{parent['program']}\n```\n\nImprove the EVOLVE-BLOCK. Call edit_solution "
            "with the new program, then evaluate_solution.")
    out = []
    for row, advantage in zip(rows, advantages):
        call = ("<tool_call>\n<function=edit_solution>\n<parameter=code>\n"
                f"{row['program']}\n</parameter>\n</function>\n</tool_call>")
        out.append({
            "messages": [
                {"role": "system", "content": system_persona(
                    REPO / "src/inner/harness/cordis.yml"
                )},
                {"role": "user", "content": user},
                {"role": "assistant", "content": call},
                {"role": "tool", "content":
                    f"Edit applied. evaluate_solution -> combined_score {row['score']:.12g}."},
            ],
            "tools": tools,
            "metadata": {
                "advantage": advantage, "reward": row["score"],
                "task_id": task.task_id, "valid": True,
                "arm": "ttt_discover_style_executor", "step": step,
                "k": row["k"], "parent_id": parent["id"],
                "adaptive_beta": beta, "tools": tools,
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--round-dir", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int,
                    help="zero-based generated batch index")
    ap.add_argument("--launched", type=int, default=16,
                    help="actual executor trajectories launched, including failed ones")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parent-id", default="root")
    ap.add_argument("--max-train-rows", type=int, default=16)
    ap.add_argument(
        "--min-train-rows", type=int, default=4,
        help="emit replay only when this many four-way-safe usable rows remain",
    )
    ap.add_argument(
        "--root-score", type=float, default=None,
        help=("score of the exact task.initial_program used as the PUCT root; "
              "defaults to the legacy frozen table"),
    )
    ap.add_argument(
        "--warm-start-score", type=float, default=None,
        help=("best-so-far score before this executor batch; defaults to the "
              "legacy fixed-H2 warm-start table"),
    )
    ap.add_argument(
        "--shared-anchor-trajectories", type=int, default=1,
        help="number of charged model trajectories before step 0",
    )
    args = ap.parse_args()
    if args.shared_anchor_trajectories < 0:
        raise SystemExit("--shared-anchor-trajectories must be >= 0")

    task = task_info(args.task)
    rows = load_batch(args.round_dir, args.task)
    # Keep a complete same-parent group whose size is safe for four-way FSDP.
    cap = min(len(rows), args.max_train_rows)
    cap -= cap % 4
    train_rows = rows[:cap] if cap >= args.min_train_rows else []

    args.state_dir.mkdir(parents=True, exist_ok=True)
    # Serialize the state read/update/curve append transaction.  This is also a
    # last line of defense if a host monitor is resumed twice.
    lock_fh = (args.state_dir / ".prepare.lock").open("a")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    state_file = args.state_dir / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        root = root_state(task, args.root_score)
        warm_start = (
            float(args.warm_start_score)
            if args.warm_start_score is not None
            else float(H2_WARM_START_SCORES[args.task])
        )
        no_shared_anchor = args.shared_anchor_trajectories == 0
        state = {"schema": 2, "task_id": args.task, "archive": {"root": root},
                 "total_expansions": 0, "cum_rollouts": 0,
                 "best": warm_start,
                 "common_warm_start": {
                     "score": warm_start,
                     "executor_trajectories": args.shared_anchor_trajectories,
                     "max_evaluator_calls": None if no_shared_anchor else 20,
                     "program_inherited": False,
                     "semantics": (
                         "external evaluator preflight score of the exact "
                         "task.initial_program at x=0"
                         if no_shared_anchor else
                         "shared fixed-H2 pre-adaptation measurement; PUCT root "
                         "remains task.initial_program"
                     ),
                 },
                 "batches": []}
    migrate_archive(state, task)
    existing_batch = next(
        (b for b in state["batches"] if int(b["step"]) == args.step), None
    )
    if existing_batch is not None:
        # Crash-safe completion of the small multi-file publish transaction.
        # state.json is the authoritative commit; regenerate any later marker
        # that was not written before preemption, without charging the batch or
        # mutating the archive twice.
        replay_path = args.state_dir / f"replay_step{args.step + 1:02d}.jsonl"
        selected_meta = state.get("selected_parent") or {}
        selected_id = selected_meta.get("id")
        selected = state["archive"].get(selected_id)
        if selected is None:
            raise SystemExit(
                f"committed step {args.step} lacks its selected parent"
            )
        pstats = {
            key: float(selected_meta[key])
            for key in ("q", "prior", "bonus", "puct")
            if selected_meta.get(key) is not None
        }
        curve = args.state_dir / "curve.jsonl"
        old_curve = read_jsonl(curve) if curve.exists() else []
        if not any(int(row.get("step", -1)) == args.step for row in old_curve):
            charged = sum(
                int(batch.get("launched", 0))
                for batch in state["batches"]
                if int(batch["step"]) <= args.step
            )
            anchor = int(
                (state.get("common_warm_start") or {}).get(
                    "executor_trajectories", args.shared_anchor_trajectories
                )
            )
            with curve.open("a") as fh:
                fh.write(json.dumps({
                    "step": args.step, "cum_rollouts": charged,
                    "cum_inference_trajectories": anchor + charged,
                    "trajectory_axis_unit": "generated_agent_trajectory",
                    "shared_anchor_x": anchor,
                    "best": existing_batch.get("best"),
                    "batch_best": existing_batch.get("batch_best"),
                    "usable": existing_batch.get("usable"),
                    "launched": existing_batch.get("launched"),
                    "checkpoint": existing_batch.get("checkpoint"),
                    "parent_id": existing_batch.get("parent_id"),
                    "recovered_from_committed_state": True,
                }) + "\n")
        parent_path = args.state_dir / f"parent_step{args.step + 1:02d}.json"
        if not parent_path.exists():
            atomic_json(parent_path, {args.task: {
                "program": selected["program"], "score": selected.get("score"),
                "state_id": selected["id"], "puct": pstats,
            }})
        manifest_path = args.state_dir / f"prepare_step{args.step:02d}.json"
        if not manifest_path.exists():
            atomic_json(manifest_path, {
                "task": args.task, "step": args.step,
                "source": existing_batch.get("round_dir"),
                "checkpoint": existing_batch.get("checkpoint"),
                "parent_id": existing_batch.get("parent_id"),
                "launched": existing_batch.get("launched"),
                "usable": existing_batch.get("usable"),
                "train_rows": existing_batch.get("train_rows"),
                "adaptive_beta": existing_batch.get("adaptive_beta"),
                "advantage_min": None, "advantage_max": None,
                "replay": str(replay_path) if replay_path.exists() else None,
                "update_eligible": replay_path.exists(),
                "update_skip_reason": existing_batch.get("update_skip_reason"),
                "next_parent": str(parent_path),
                "next_parent_id": selected["id"],
                "nonce": str(uuid.uuid4()),
                "recovered_from_committed_state": True,
            })
        print(manifest_path.read_text())
        lock_fh.close()
        return
    if args.parent_id not in state["archive"]:
        raise SystemExit(f"unknown parent {args.parent_id}")
    parent = state["archive"][args.parent_id]

    beta, advantages = adaptive_entropic_advantages([r["score"] for r in train_rows])
    replay = make_replay(task, train_rows, advantages, beta, parent, args.step, tool_schemas()) \
        if train_rows else []
    replay_path = args.state_dir / f"replay_step{args.step + 1:02d}.jsonl"
    if replay:
        replay_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in replay))
    else:
        replay_path.unlink(missing_ok=True)

    if rows:
        update_archive(state, rows, args.parent_id, args.step, args.checkpoint)
    state["cum_rollouts"] = int(state.get("cum_rollouts", 0)) + args.launched
    batch_best = max((r["score"] for r in rows), default=None)
    if batch_best is not None:
        fallback_best = (
            float(args.warm_start_score)
            if args.warm_start_score is not None
            else float(H2_WARM_START_SCORES[args.task])
        )
        state["best"] = max(float(state.get("best") if state.get("best") is not None
                                  else fallback_best), batch_best)
    state["batches"].append({
        "step": args.step, "round_dir": str(args.round_dir),
        "checkpoint": args.checkpoint, "parent_id": args.parent_id,
        "launched": args.launched, "usable": len(rows), "train_rows": len(train_rows),
        "evaluator_calls": sum(r["evaluations"] for r in rows),
        "batch_best": batch_best, "best": state["best"], "adaptive_beta": beta,
        "update_eligible": len(train_rows) >= args.min_train_rows,
        "update_skip_reason": (
            None if len(train_rows) >= args.min_train_rows
            else f"usable_rows_{len(rows)}_below_min_train_rows_{args.min_train_rows}"
        ),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    selected, pstats = select_puct(state)
    state["selected_parent"] = {"id": selected["id"], **pstats}
    atomic_json(state_file, state)

    curve = args.state_dir / "curve.jsonl"
    with curve.open("a") as fh:
        fh.write(json.dumps({
            "step": args.step, "cum_rollouts": state["cum_rollouts"],
            "cum_inference_trajectories": (
                args.shared_anchor_trajectories + state["cum_rollouts"]
            ),
            "trajectory_axis_unit": "generated_agent_trajectory",
            "shared_anchor_x": args.shared_anchor_trajectories,
            "best": state["best"], "batch_best": batch_best,
            "usable": len(rows), "launched": args.launched,
            "checkpoint": args.checkpoint, "parent_id": args.parent_id,
        }) + "\n")
    parent_path = args.state_dir / f"parent_step{args.step + 1:02d}.json"
    atomic_json(parent_path, {args.task: {
        "program": selected["program"], "score": selected.get("score"),
        "state_id": selected["id"], "puct": pstats,
    }})
    manifest = {
        "task": args.task, "step": args.step, "source": str(args.round_dir),
        "checkpoint": args.checkpoint, "parent_id": args.parent_id,
        "launched": args.launched, "usable": len(rows), "train_rows": len(train_rows),
        "adaptive_beta": beta,
        "advantage_min": min(advantages) if advantages else None,
        "advantage_max": max(advantages) if advantages else None,
        "replay": str(replay_path) if replay else None,
        "update_eligible": bool(replay),
        "update_skip_reason": (
            None if replay
            else f"usable_rows_{len(rows)}_below_min_train_rows_{args.min_train_rows}"
        ),
        "next_parent": str(parent_path), "next_parent_id": selected["id"],
        "nonce": str(uuid.uuid4()),
    }
    atomic_json(args.state_dir / f"prepare_step{args.step:02d}.json", manifest)
    print(json.dumps(manifest, indent=2))
    lock_fh.close()


if __name__ == "__main__":
    main()

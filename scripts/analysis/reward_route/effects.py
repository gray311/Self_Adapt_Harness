#!/usr/bin/env python3
"""Build a per-batch effect/provenance ledger for inference16-v1.

The score transition after an update is observational: the next batch also
contains fresh sampling and a possibly changed program parent.  The ledger
therefore never calls that delta a causal weight effect.  It does bind every
transition to the exact saved trajectory/replay hashes, checkpoints, context
snapshots, and fixed 16-trajectory budget needed for a later replay audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "reward-route-inference16-v1"
ROUNDS = 19
PER_BATCH = 16
FINAL_X = 305
TASKS = (
    ("erdos", "eft__math__erdos_min_overlap", "Erdős min-overlap", 2000, 3000),
    ("ac2", "eft__math__second_autocorr_ineq", "Autocorrelation II", 2100, 3100),
    ("hadamard", "eft__math__hadamard_maximal_det", "Hadamard max-det", 2200, 3200),
    ("eplb", "adrs__eplb", "EPLB", 2300, 3300),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": sha256(path),
    }


def bundle_hash(paths: Iterable[Path]) -> dict[str, Any]:
    files = sorted({path for path in paths if path.is_file()}, key=lambda p: str(p))
    digest = hashlib.sha256()
    members = []
    for path in files:
        member_sha = sha256(path)
        assert member_sha is not None
        digest.update(str(path).encode() + b"\0" + member_sha.encode() + b"\n")
        members.append({"path": str(path), "sha256": member_sha, "bytes": path.stat().st_size})
    return {
        "sha256": digest.hexdigest() if members else None,
        "file_count": len(members),
        "members": members,
    }


def task_summary(path: Path, task: str) -> dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception:
        return None
    rows = payload if isinstance(payload, list) else [payload]
    return next((row for row in rows if row.get("task_id") == task), None)


def rollout_evidence(round_dir: Path, task: str) -> dict[str, Any]:
    summaries = sorted((round_dir / "rollouts" / task).glob("cand*/*/summary.json"))
    rows = []
    for path in summaries:
        row = task_summary(path, task)
        if row is None:
            continue
        rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "best_score": row.get("best_score"),
            "evaluations": int(row.get("evaluations") or 0),
            "llm_calls": int((row.get("ledger") or {}).get("llm_calls") or 0),
            "decode_seed": row.get("decode_seed"),
            "seed_program_provenance": row.get("seed_program_provenance"),
            "h2_package_provenance": row.get("h2_package_provenance"),
            "score_eligible": row.get("score_eligible"),
            "has_program": bool(row.get("best_program")),
            "error": row.get("error"),
        })
    return {
        "bundle": bundle_hash(summaries),
        "terminal_summaries": len(rows),
        "usable": sum(row["best_score"] is not None and row["has_program"] for row in rows),
        "evaluator_calls": sum(row["evaluations"] for row in rows),
        "executor_model_calls": sum(row["llm_calls"] for row in rows),
        "batch_best": max(
            (float(row["best_score"]) for row in rows if row["best_score"] is not None),
            default=None,
        ),
        "rows": rows,
    }


def parse_job_ids(path: Path) -> dict[str, int | None]:
    text = path.read_text() if path.is_file() else ""
    out: dict[str, int | None] = {"train_job": None, "merge_job": None, "eval_job": None}
    patterns = {
        "train_job": (r"train job:\s*(\d+)", r"^TRAIN_JOB=(\d+)$"),
        "merge_job": (r"merge job:\s*(\d+)", r"^MERGE_JOB=(\d+)$"),
        "eval_job": (r"^EVAL_JOB=(\d+)$",),
    }
    for key, candidates in patterns.items():
        for pattern in candidates:
            found = re.search(pattern, text, re.MULTILINE)
            if found:
                out[key] = int(found.group(1))
                break
    return out


def expected_x(index: int) -> int:
    return 1 + PER_BATCH * (index + 1)


def h1_batch(
    *, run_root: Path, model_root: Path, method: str, tag: str, task: str,
    round_base: int, index: int, score_before: float,
) -> dict[str, Any] | None:
    outer = run_root / "self_adapt_harness" / f"outer-{PROTOCOL}-{method}-{tag}"
    round_dir = outer / f"round{round_base + index:03d}"
    required = (
        round_dir / "round.json",
        round_dir / "round_summary.json",
        round_dir / "next_bases.json",
        round_dir / "h2_slot_plan.json",
        round_dir / "trajectories.json",
        round_dir / "prompts.json",
        round_dir / "seed_programs_in.json",
        round_dir / "program_ratchet_audit.json",
    )
    if not all(path.is_file() for path in required):
        return None
    meta = load_json(round_dir / "round.json")
    summary = load_json(round_dir / "round_summary.json")
    next_bases = load_json(round_dir / "next_bases.json")
    slots = load_json(round_dir / "h2_slot_plan.json")
    budget = meta.get("inference_trajectory_budget") or {}
    if not (
        budget.get("axis_unit") == "generated_agent_trajectory"
        and budget.get("fixed_h1_plus_h2_slots") is True
        and int(budget.get("logical_round_index")) == index
        and int(budget.get("h1_slots_per_task")) == 8
        and int(budget.get("h2_slots_per_task")) == 8
        and int(budget.get("axis_x_after_round")) == expected_x(index)
        and len(slots.get("slots") or []) == 8
    ):
        raise AssertionError(f"invalid inference budget in {round_dir}")
    group = summary["groups"][task]
    task_meta = meta["per_task"][task]
    score_after = float(next_bases[task]["score"])
    valid_h1 = sum(bool(row.get("valid")) for row in task_meta["candidates"])
    rollout = rollout_evidence(round_dir, task)
    if rollout["terminal_summaries"] != 8 or any(
        not row.get("seed_program_provenance")
        or not row.get("h2_package_provenance")
        for row in rollout["rows"]
    ):
        raise AssertionError(
            f"incomplete seed/H2 rollout provenance in {round_dir}/{task}"
        )
    trajectory_bundle = bundle_hash([
        round_dir / "trajectories.json",
        round_dir / "h2_slot_plan.json",
        round_dir / "seed_programs_in.json",
        round_dir / "program_ratchet_audit.json",
        *[Path(row["path"]) for row in rollout["rows"]],
    ])
    prompt_payload = load_json(round_dir / "prompts.json")
    prompt_text = str(prompt_payload[task])
    analysis = task_meta.get("analysis") or {}
    ratchet = load_json(round_dir / "program_ratchet_audit.json")
    if meta.get("program_ratchet_mode") != "strict_single" \
            or ratchet.get("mode") != "strict_single":
        raise AssertionError(f"noncanonical program ratchet in {round_dir}")
    update: dict[str, Any]
    if method == "proposer":
        replay = round_dir / "replay.jsonl"
        replay_manifest_path = round_dir / "replay_manifest.json"
        replay_manifest = (
            load_json(replay_manifest_path)
            if replay_manifest_path.is_file() else {}
        )
        receipt = round_dir / "train_submit.log"
        expected_merged = (
            model_root / "exports" / "self_adapt_harness"
            / f"mphi_rri16_{tag}_proposer_{index:02d}"
        )
        expected_lora = (
            model_root / "checkpoints" / "self_adapt_harness"
            / f"mphi_rri16_{tag}_proposer_{index:02d}"
        )
        optimizer_rows = sum(
            1 for line in replay.read_text().splitlines() if line.strip()
        ) if replay.is_file() else 0
        trainable_h1 = int(
            replay_manifest.get("generated_trainable_rows", optimizer_rows)
        )
        eligible = trainable_h1 >= 4 and index < ROUNDS - 1
        materialized = (
            (expected_merged / "config.json").is_file()
            and any(expected_merged.glob("*.safetensors"))
        )
        update = {
            "target": "proposer_weights",
            "opportunity": index < ROUNDS - 1,
            "eligible": eligible,
            "materialized": bool(eligible and materialized),
            "applied": False if not eligible else None,
            "evaluated_in_next_batch": False,
            "skip_reason": (
                "final_measurement_batch" if index == ROUNDS - 1
                else (f"trainable_h1_{trainable_h1}_below_4"
                      if trainable_h1 < 4 else None)
            ),
            "trainable_h1_rows": trainable_h1,
            "optimizer_rows": int(
                replay_manifest.get("optimizer_rows", optimizer_rows)
            ),
            "archive_mixed_rows": int(
                replay_manifest.get("archive_mixed_rows", 0)
            ),
            "zero_advantage_padding_rows": int(
                replay_manifest.get("zero_advantage_padding_rows", 0)
            ),
            "pre_checkpoint": (meta.get("proposer") or {}).get("checkpoint"),
            "post_checkpoint": str(expected_merged) if materialized else None,
            "lora_checkpoint": str(expected_lora) if materialized else None,
            "training_input": artifact(replay),
            "training_manifest": artifact(replay_manifest_path),
            "submission_receipt": artifact(receipt),
            "jobs": parse_job_ids(receipt),
        }
    else:
        feedback_after = round_dir / "task_feedback_after.json"
        programs_after = round_dir / "best_programs_after.json"
        update = {
            "target": "analyzer_context",
            "opportunity": index < ROUNDS - 1,
            "eligible": index < ROUNDS - 1,
            "materialized": feedback_after.is_file(),
            "applied": False if index == ROUNDS - 1 else None,
            "evaluated_in_next_batch": False,
            "weights_changed": False,
            "input_feedback": analysis.get("feedback_input"),
            "output_feedback": artifact(feedback_after),
            "output_program_context": artifact(programs_after),
        }
    return {
        "batch_index": index,
        "round": round_base + index,
        "x_before": 1 if index == 0 else expected_x(index - 1),
        "x_after": expected_x(index),
        "h1_trajectories": 8,
        "h2_trajectories": 8,
        "h1_valid": valid_h1,
        "h1_invalid": 8 - valid_h1,
        "incumbent_fallbacks": sum(
            row.get("h2_slot_mode") == "incumbent_fallback"
            for row in slots["slots"]
        ),
        "score_before": score_before,
        "score_after": score_after,
        "search_gain": score_after - score_before,
        "batch_candidate_best": group.get("best_score"),
        "batch_improved": bool(group.get("improved")),
        "accepted_improvement": bool(group.get("accepted_improvement")),
        "program_ratchet": ratchet["tasks"][task],
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "proposer_checkpoint": (meta.get("proposer") or {}).get("checkpoint"),
        "round_job": int(meta["slurm_job_id"]) if meta.get("slurm_job_id") else None,
        "executor_checkpoint": (meta.get("proposer") or {}).get("executor_checkpoint"),
        "analysis": {
            "enabled": bool(analysis.get("enabled")),
            "feedback_available": bool(analysis.get("feedback_available")),
            "brief_attached": bool(analysis.get("brief_attached")),
            "model_calls": int(analysis.get("model_calls") or 0),
            "specialists": list(analysis.get("specialists") or []),
            "feedback_input": analysis.get("feedback_input"),
        },
        "cross_round_inputs": meta.get("cross_round_inputs") or {},
        "h1_model_calls": sum(int(row.get("llm_calls") or 0) for row in task_meta["candidates"]),
        "post_submit_reviewer_model_calls": 0,
        "component_validation_records": sum(
            len(row.get("review_log") or []) for row in task_meta["candidates"]
        ),
        "rollout": rollout,
        "trajectory_bundle": trajectory_bundle,
        "artifacts": {
            name: artifact(round_dir / name)
            for name in (
                "round.json", "round_summary.json", "next_bases.json",
                "h2_slot_plan.json", "trajectories.json", "prompts.json",
                "seed_programs_in.json", "program_ratchet_audit.json",
                "grpo_batch.jsonl", "replay_manifest.json",
            )
        },
        "outgoing_update": update,
    }


def executor_batch(
    *, run_root: Path, tag: str, task: str, index: int, row: dict[str, Any],
    next_row: dict[str, Any] | None,
) -> dict[str, Any]:
    state_dir = run_root / "self_adapt_harness" / PROTOCOL / "executor" / tag
    eval_dir = state_dir / f"eval_rri16e_u{index}"
    prepare_path = state_dir / f"prepare_step{index:02d}.json"
    prepare = load_json(prepare_path) if prepare_path.is_file() else {}
    eval_manifest_path = eval_dir / "eval_manifest.json"
    eval_manifest = load_json(eval_manifest_path) if eval_manifest_path.is_file() else {}
    summaries = sorted(eval_dir.glob("k*/*/summary.json"))
    summary_rows = [
        task_summary(path, task) for path in summaries
    ]
    summary_rows = [row for row in summary_rows if row is not None]
    receipt = state_dir / f"jobs_rri16e_u{index + 1}.env"
    next_replay = state_dir / f"replay_step{index + 1:02d}.jsonl"
    opportunity = index < ROUNDS - 1
    eligible = bool(prepare.get("update_eligible")) if opportunity else False
    applied = bool(
        opportunity
        and next_row is not None
        and next_row.get("checkpoint") != row.get("checkpoint")
    )
    return {
        "batch_index": index,
        "step": int(row["step"]),
        "x_before": 1 if index == 0 else expected_x(index - 1),
        "x_after": expected_x(index),
        "h1_trajectories": 0,
        "h2_trajectories": 16,
        "score_before": None,
        "score_after": float(row["best"]),
        "search_gain": None,
        "batch_candidate_best": row.get("batch_best"),
        "executor_checkpoint": row.get("checkpoint"),
        "eval_job": (
            int(eval_manifest["slurm_job_id"])
            if eval_manifest.get("slurm_job_id") else None
        ),
        "usable": int(row.get("usable") or 0),
        "rollout": {
            "bundle": bundle_hash(summaries),
            "terminal_summaries": len(summaries),
            "usable": int(eval_manifest.get("usable") or row.get("usable") or 0),
            "evaluator_calls": int(eval_manifest.get("evaluator_calls") or 0),
            "executor_model_calls": int(
                eval_manifest.get("executor_model_calls")
                or sum(
                    int((summary.get("ledger") or {}).get("llm_calls") or 0)
                    for summary in summary_rows
                )
            ),
            "batch_best": eval_manifest.get("batch_best"),
        },
        "trajectory_bundle": bundle_hash(summaries),
        "artifacts": {
            "eval_manifest": artifact(eval_manifest_path),
            "prepare": artifact(prepare_path),
        },
        "outgoing_update": {
            "target": "executor_weights",
            "opportunity": opportunity,
            "eligible": eligible,
            "materialized": applied,
            "applied": applied if next_row is not None else (False if not eligible else None),
            "evaluated_in_next_batch": applied,
            "skip_reason": (
                "final_measurement_batch" if not opportunity
                else prepare.get("update_skip_reason")
            ),
            "pre_checkpoint": row.get("checkpoint"),
            "post_checkpoint": next_row.get("checkpoint") if next_row else None,
            "training_rows": int(prepare.get("train_rows") or 0),
            "training_input": artifact(next_replay),
            "submission_receipt": artifact(receipt),
            "jobs": parse_job_ids(receipt),
        },
    }


def attach_next_observations(batches: list[dict[str, Any]], human: float) -> None:
    for index, batch in enumerate(batches):
        batch["score_after_normalized"] = batch["score_after"] / human
        update = batch["outgoing_update"]
        if index + 1 >= len(batches):
            batch["next_batch_observation"] = None
            continue
        nxt = batches[index + 1]
        applied = False
        if update["target"] == "proposer_weights":
            applied = bool(
                update.get("materialized")
                and update.get("post_checkpoint")
                and nxt.get("proposer_checkpoint") == update.get("post_checkpoint")
            )
        elif update["target"] == "analyzer_context":
            output_sha = (update.get("output_feedback") or {}).get("sha256")
            next_input = (nxt.get("analysis") or {}).get("feedback_input") or {}
            applied = bool(output_sha and next_input.get("sha256") == output_sha)
        elif update["target"] == "executor_weights":
            applied = bool(
                update.get("eligible")
                and update.get("post_checkpoint")
                and nxt.get("executor_checkpoint") == update.get("post_checkpoint")
            )
        update["applied"] = applied
        update["evaluated_in_next_batch"] = applied
        delta = float(nxt["score_after"]) - float(batch["score_after"])
        batch["next_batch_observation"] = {
            "x": int(nxt["x_after"]),
            "score": float(nxt["score_after"]),
            "delta": delta,
            "normalized_delta": delta / human,
            "interpretation": (
                "observational next-batch best-so-far change; fresh sampling and "
                "parent/context changes prevent a causal weight-only attribution"
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=os.environ.get("RUN_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/runs"),
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MODEL_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/model_weights"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "reward_route_inference16_effects.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    run_root, model_root = Path(args.run_root), Path(args.model_root)
    baseline_path = REPO / "results" / "baseline_h2_20ev.json"
    human_path = REPO / "results" / "human_best_references.json"
    baseline = load_json(baseline_path)["baseline"]
    human_refs = load_json(human_path)["tasks"]
    output: dict[str, Any] = {
        "schema": 1,
        "protocol": PROTOCOL,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "not_started",
        "effect_semantics": {
            "search_gain": "best-so-far change within the current fixed trajectory batch",
            "next_batch_delta": (
                "observational transition after the route update; not a causal estimate "
                "because the next batch also resamples and may change its parent/context"
            ),
            "same_log_binding": (
                "trajectory_bundle and training_input SHA-256 bind every update to the exact "
                "saved evidence; a causal replay score must be reported separately"
            ),
        },
        "trajectory_grid": [1] + [expected_x(i) for i in range(ROUNDS)],
        "common_final_x": FINAL_X,
        "sources": {
            "baseline": artifact(baseline_path),
            "human_best": artifact(human_path),
        },
        "tasks": {},
    }
    complete_routes = 0
    started_routes = 0
    for tag, task, title, prop_base, ctx_base in TASKS:
        anchor = float(baseline[task]["h2_best"])
        human = float(human_refs[task]["human_best_combined_score"])
        routes: dict[str, Any] = {}
        for method, base in (("proposer", prop_base), ("context", ctx_base)):
            batches = []
            score = anchor
            for index in range(ROUNDS):
                row = h1_batch(
                    run_root=run_root, model_root=model_root, method=method,
                    tag=tag, task=task, round_base=base, index=index,
                    score_before=score,
                )
                if row is None:
                    break
                batches.append(row)
                score = float(row["score_after"])
            attach_next_observations(batches, human)
            routes[method] = {
                "status": "complete" if len(batches) == ROUNDS else ("live" if batches else "not_started"),
                "anchor": {"x": 1, "score": anchor, "normalized": anchor / human},
                "completed_batches": len(batches),
                "batches": batches,
            }
        curve_path = run_root / "self_adapt_harness" / PROTOCOL / "executor" / tag / "curve.jsonl"
        curve_rows = []
        if curve_path.is_file():
            curve_rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
        batches = []
        for index, row in enumerate(curve_rows[:ROUNDS]):
            if int(row.get("step", -1)) != index or int(row.get("launched", -1)) != 16:
                raise AssertionError(f"noncanonical executor row {tag}/{index}: {row}")
            nxt = curve_rows[index + 1] if index + 1 < len(curve_rows) else None
            batches.append(executor_batch(
                run_root=run_root, tag=tag, task=task, index=index,
                row=row, next_row=nxt,
            ))
        previous = anchor
        for batch in batches:
            batch["score_before"] = previous
            batch["search_gain"] = float(batch["score_after"]) - previous
            previous = float(batch["score_after"])
        attach_next_observations(batches, human)
        routes["executor"] = {
            "status": "complete" if len(batches) == ROUNDS else ("live" if batches else "not_started"),
            "anchor": {"x": 1, "score": anchor, "normalized": anchor / human},
            "completed_batches": len(batches),
            "curve_source": artifact(curve_path),
            "batches": batches,
        }
        for route in routes.values():
            started_routes += route["status"] != "not_started"
            complete_routes += route["status"] == "complete"
        output["tasks"][task] = {
            "tag": tag,
            "title": title,
            "human_best_combined_score": human,
            "routes": routes,
        }
    total_routes = len(TASKS) * 3
    output["status"] = (
        "complete" if complete_routes == total_routes
        else ("live" if started_routes else "not_started")
    )
    output["route_progress"] = {
        "complete": complete_routes,
        "started": started_routes,
        "total": total_routes,
    }
    if args.require_complete and output["status"] != "complete":
        raise SystemExit(
            f"inference16 effect ledger is {output['status']}: "
            f"{complete_routes}/{total_routes} routes complete"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps({
        "status": output["status"],
        "route_progress": output["route_progress"],
        "out": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

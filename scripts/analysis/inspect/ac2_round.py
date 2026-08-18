#!/usr/bin/env python3
"""Build a lossless, human-navigable AC2 H1 -> H2 -> reward evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ID = "eft__math__second_autocorr_ineq"
ROUND_NAME = "round001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def message_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact messages presented before the first assistant turn."""
    prefix = []
    for message in messages:
        role = str(message.get("role", "")).lower().split(".")[-1]
        if role == "assistant":
            break
        prefix.append(message)
    return prefix


def model_conditioning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project message envelopes to fields that condition the model."""
    rows = []
    for message in messages:
        row = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("name") is not None:
            row["name"] = message["name"]
        rows.append(row)
    return rows


def text_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = None
            encoding = "binary-not-embedded"
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": encoding,
            "content": content,
        })
    return rows


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_index.json":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def result_path(round_dir: Path, k: int) -> Path | None:
    root = round_dir / "rollouts" / TASK_ID / f"cand{k:02d}"
    hits = sorted(root.glob(f"*/results/{TASK_ID}.json"))
    return hits[0] if len(hits) == 1 else None


def provenance_path(result: Path) -> Path:
    return result.parents[1] / "provenance.json"


def relative_or_absolute(path: Path, run: Path) -> str:
    try:
        return path.resolve().relative_to(run.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    run = args.run_dir.resolve()
    round_dir = run / "rounds" / ROUND_NAME
    if not (round_dir / "round.json").is_file():
        raise SystemExit(f"proposal artifacts are not ready: {round_dir}")

    out = run / "inspection" / "ac2_round001"
    out.mkdir(parents=True, exist_ok=True)
    atomic_text(out / "builder_snapshot.py", Path(__file__).read_text())
    meta = json.loads((round_dir / "round.json").read_text())
    if meta.get("tasks_order") != [TASK_ID]:
        raise SystemExit(f"unexpected task set: {meta.get('tasks_order')}")
    k_target = int(meta["k"])

    prompts = json.loads((round_dir / "prompts.json").read_text())
    trajectory_rows = json.loads((round_dir / "trajectories.json").read_text())
    trajectories = {
        int(row["k"]): row for row in trajectory_rows if row.get("task_id") == TASK_ID
    }
    if sorted(trajectories) != list(range(k_target)):
        raise SystemExit("H1 trajectory indices do not match round K")

    common_input_hashes = []
    candidates: list[dict[str, Any]] = []
    for k in range(k_target):
        cand = f"cand{k:02d}"
        cand_out = out / "candidates" / cand
        h1 = trajectories[k]
        exact_h1_input = message_prefix(h1["trajectory"])
        conditioned_h1_input = model_conditioning(exact_h1_input)
        input_bytes = json.dumps(
            conditioned_h1_input, sort_keys=True, ensure_ascii=False
        ).encode()
        input_hash = hashlib.sha256(input_bytes).hexdigest()
        common_input_hashes.append(input_hash)

        atomic_json(cand_out / "01_proposer_exact_input.json", {
            "task_id": TASK_ID,
            "k": k,
            "exact_message_envelopes_before_first_assistant": exact_h1_input,
            "model_conditioning_messages": conditioned_h1_input,
            "outer_user_prompt_from_prompts_json": prompts[TASK_ID],
            "source_outer_agent_yaml": str(
                (run / "harness_snapshot" / "outer_h1" / "agent.yaml").resolve()
            ),
            "source_outer_system_md": str(
                (run / "harness_snapshot" / "outer_h1" / "system.md").resolve()
            ),
            "model_conditioning_sha256": input_hash,
        })
        atomic_json(cand_out / "02_proposer_full_trajectory.json", h1["trajectory"])
        atomic_text(cand_out / "03_proposer_raw_submission.txt",
                    str(h1.get("raw_submission") or ""))

        harness_dir = round_dir / "tasks" / TASK_ID / cand
        if not harness_dir.is_dir():
            raise SystemExit(f"materialized harness is missing: {harness_dir}")
        harness_files = text_inventory(harness_dir)
        atomic_json(cand_out / "04_generated_harness.json", {
            "task_id": TASK_ID,
            "k": k,
            "source_directory": str(harness_dir.resolve()),
            "files": harness_files,
        })

        candidate_row: dict[str, Any] = {
            "k": k,
            "proposer_model_conditioning_sha256": input_hash,
            "proposer_trajectory_sha256": sha256(
                cand_out / "02_proposer_full_trajectory.json"
            ),
            "raw_submission_sha256": sha256(
                cand_out / "03_proposer_raw_submission.txt"
            ),
            "generated_harness_file_count": len(harness_files),
            "generated_harness_source": str(harness_dir.resolve()),
            "executor_ready": False,
        }

        result = result_path(round_dir, k)
        if result is not None:
            payload = json.loads(result.read_text())
            provenance_file = provenance_path(result)
            provenance = json.loads(provenance_file.read_text())
            seed = payload.get("seed_program_provenance") or {}
            seed_path = Path(str(seed.get("program_path", "")))
            seed_content = seed_path.read_text() if seed_path.is_file() else None
            h2_messages = payload.get("trajectory") or []
            atomic_json(cand_out / "05_executor_exact_input.json", {
                "task_id": TASK_ID,
                "k": k,
                "messages_before_first_assistant": message_prefix(h2_messages),
                "server_and_decode_provenance": provenance,
                "materialized_harness_provenance": (
                    payload.get("h2_package_provenance") or {}
                ),
                "initial_program_provenance": seed,
                "initial_program_content": seed_content,
            })
            atomic_json(cand_out / "06_executor_full_trajectory.json", h2_messages)
            atomic_json(cand_out / "07_executor_reward.json", {
                "task_id": TASK_ID,
                "k": k,
                "score_eligible": payload.get("score_eligible"),
                "seed_score": payload.get("seed_score"),
                "best_score": payload.get("best_score"),
                "best_metrics": payload.get("best_metrics"),
                "stop_reason": payload.get("stop_reason"),
                "error": payload.get("error"),
                "ledger": payload.get("ledger"),
                "steps": payload.get("steps"),
                "middleware_audit": payload.get("middleware_audit"),
                "tool_audit": payload.get("tool_audit"),
                "skill_audit": payload.get("skill_audit"),
            })
            atomic_text(cand_out / "08_executor_output_program.py",
                        str(payload.get("best_program") or ""))
            candidate_row.update({
                "executor_ready": True,
                "executor_result": relative_or_absolute(result, run),
                "executor_result_sha256": sha256(result),
                "score_eligible": payload.get("score_eligible"),
                "seed_score": payload.get("seed_score"),
                "best_score": payload.get("best_score"),
                "decode_seed": payload.get("decode_seed"),
            })
        candidates.append(candidate_row)

    grpo_path = round_dir / "grpo_batch.jsonl"
    replay_path = run / "replay" / "grpo_replay_keep_zero.jsonl"
    training_ready = grpo_path.is_file() and replay_path.is_file()
    training_summary: dict[str, Any] = {"ready": training_ready}
    score_table_md = "Reward/advantage artifacts are not ready yet."
    if training_ready:
        round_summary = json.loads((round_dir / "round_summary.json").read_text())
        reward_group = (round_summary.get("groups") or {}).get(TASK_ID) or {}
        grpo_rows = sorted(load_jsonl(grpo_path), key=lambda row: int(row["k"]))
        replay_rows = sorted(
            load_jsonl(replay_path),
            key=lambda row: int((row.get("metadata") or {})["k"]),
        )
        if len(grpo_rows) != k_target or len(replay_rows) != k_target:
            raise SystemExit("training rows do not match round K")
        table = []
        for grpo, replay in zip(grpo_rows, replay_rows):
            k = int(grpo["k"])
            if int(replay["metadata"]["k"]) != k:
                raise SystemExit("GRPO/replay candidate alignment failed")
            cand_out = out / "candidates" / f"cand{k:02d}"
            atomic_json(cand_out / "09_proposer_grpo_training_row.json", grpo)
            atomic_json(cand_out / "10_qwen_training_replay_row.json", replay)
            table.append({
                "k": k,
                "valid": grpo.get("valid"),
                "score": grpo.get("score"),
                "reward": grpo.get("reward"),
                "advantage": grpo.get("advantage"),
                "spec_hash": grpo.get("spec_hash"),
                "group_id": replay["metadata"].get("group_id"),
                "sample_id": replay["metadata"].get("sample_id"),
                "message_count": len(replay.get("messages") or []),
            })
        atomic_json(out / "training" / "reward_advantage_table.json", table)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        atomic_text(out / "training" / "reward_advantage_table.csv",
                    csv_buffer.getvalue())
        score_lines = [
            "| candidate | executor score | reward | advantage | proposer update |",
            "|---|---:|---:|---:|---|",
        ]
        for row in table:
            advantage = float(row["advantage"])
            direction = "increase H1 action likelihood" if advantage > 0 else (
                "decrease H1 action likelihood" if advantage < 0 else "no update"
            )
            score_lines.append(
                f"| cand{int(row['k']):02d} | {float(row['score']):.12f} | "
                f"{float(row['reward']):+.12f} | {advantage:+.12f} | {direction} |"
            )
        score_table_md = "\n".join(score_lines)
        atomic_json(out / "training" / "training_contract.json", {
            "task_id": TASK_ID,
            "round": 1,
            "group_size": k_target,
            "same_group_id": len({row["group_id"] for row in table}) == 1,
            "unique_sample_ids": len({row["sample_id"] for row in table}) == k_target,
            "advantage_sum": sum(float(row["advantage"]) for row in table),
            "positive_advantages": sum(float(row["advantage"]) > 0 for row in table),
            "negative_advantages": sum(float(row["advantage"]) < 0 for row in table),
            "zero_advantages": sum(float(row["advantage"]) == 0 for row in table),
            "reward_and_advantage": {
                "implementation": "SAH_ADV=v2",
                "base_score": reward_group.get("base_score"),
                "ceiling": reward_group.get("ceiling"),
                "reward_mean": reward_group.get("reward_mean"),
                "reward_std": reward_group.get("reward_std"),
                "advantage_mode": reward_group.get("adv_mode"),
                "raw_improved": reward_group.get("improved"),
                "accepted_improvement": reward_group.get("accepted_improvement"),
                "training_suppressed": reward_group.get("training_suppressed", False),
                "training_suppression_reason": reward_group.get(
                    "training_suppression_reason"
                ),
                "formula": (
                    "reward closes a fraction of the base-to-ceiling gap; advantage "
                    "is 0.7 * leave-one-out reward contrast + 0.3 * centered "
                    "softmax(max-oriented) weight, unless the group is no-signal or "
                    "strict program-ratchet attribution suppresses training"
                ),
            },
            "reward_source": str(
                (run / "source_snapshot" / "src" / "outer" / "rewards.py").resolve()
            ),
            "replay_converter": str(
                (run / "source_snapshot" / "src" / "training" /
                 "grpo_to_replay.py").resolve()
            ),
            "loss_target": (
                "Qwen3.5 assistant tokens in the full H1 trajectory; tool/user/"
                "system tokens are context-only, with each sample weighted by its "
                "saved scalar advantage"
            ),
        })
        training_summary = {
            "ready": True,
            "rows": k_target,
            "positive_advantages": sum(float(row["advantage"]) > 0 for row in table),
            "negative_advantages": sum(float(row["advantage"]) < 0 for row in table),
            "table": "training/reward_advantage_table.json",
            "contract": "training/training_contract.json",
        }

    # A later invocation of this same builder also captures the real optimizer
    # step.  Keeping this conditional lets the proposal/rollout bundle remain
    # useful while the GPU training job is still pending.
    train_phase = run / "phases" / "training_smoke"
    train_command_path = train_phase / "train_command.json"
    checkpoint_audit_path = run / "audits" / "training_checkpoint_audit.json"
    optimizer_summary: dict[str, Any] = {"ready": False}
    if train_command_path.is_file() and checkpoint_audit_path.is_file():
        train_command = json.loads(train_command_path.read_text())
        checkpoint_audit = json.loads(checkpoint_audit_path.read_text())
        exact_command_out = out / "training" / "11_exact_train_command.json"
        checkpoint_out = out / "training" / "12_optimizer_step_audit.json"
        atomic_json(exact_command_out, train_command)
        atomic_json(checkpoint_out, checkpoint_audit)
        train_log = train_phase / "train.log"
        train_log_out: Path | None = None
        if train_log.is_file():
            train_log_out = out / "training" / "13_full_train_log.txt"
            atomic_text(train_log_out, train_log.read_text(errors="replace"))
        close_loop_exports: dict[str, str] = {}
        close_loop_sources = [
            (
                run / "audits" / "merged_model_audit.json",
                "14_merged_model_audit.json",
                "merged_model_audit",
            ),
            (
                run / "phases" / "merged_serve_smoke" /
                "server_protocol" / "validation.json",
                "15_merged_serve_protocol.json",
                "merged_serve_protocol",
            ),
            (
                run / "audits" / "training_final_audit.json",
                "16_training_final_audit.json",
                "training_final_audit",
            ),
        ]
        for source, filename, key in close_loop_sources:
            if source.is_file():
                destination = out / "training" / filename
                atomic_json(destination, json.loads(source.read_text()))
                close_loop_exports[key] = destination.relative_to(out).as_posix()
        optimizer_summary = {
            "ready": checkpoint_audit.get("passed") is True,
            "global_step": (
                checkpoint_audit.get("checkpoint_metadata") or {}
            ).get("global_step"),
            "iteration": checkpoint_audit.get("iteration"),
            "grad_norms": checkpoint_audit.get("grad_norms"),
            "adapter_tensor_count": checkpoint_audit.get("adapter_tensor_count"),
            "lora_b_abs_sum": checkpoint_audit.get("lora_b_abs_sum"),
            "adapter_sha256": checkpoint_audit.get("adapter_sha256"),
            "exact_command": exact_command_out.relative_to(out).as_posix(),
            "checkpoint_audit": checkpoint_out.relative_to(out).as_posix(),
            "full_train_log": (
                train_log_out.relative_to(out).as_posix()
                if train_log_out is not None else None
            ),
            "close_loop": close_loop_exports,
        }
    training_summary["optimizer_step"] = optimizer_summary

    all_same_input = len(set(common_input_hashes)) == 1
    index = {
        "schema": "ac2-round-inspection/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "task_id": TASK_ID,
        "round": 1,
        "k": k_target,
        "all_candidates_share_model_conditioning": all_same_input,
        "proposer_model_conditioning_sha256": (
            common_input_hashes[0] if all_same_input else common_input_hashes
        ),
        "executor_results_ready": sum(row["executor_ready"] for row in candidates),
        "training": training_summary,
        "candidates": candidates,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    atomic_json(out / "manifest.json", index)

    readme = f"""# AC2 round-1 inspection bundle

This bundle is a lossless view over `{run}`. Original artifacts remain the
source of truth; every exported JSON keeps complete message content.

- Task: `{TASK_ID}`
- Harness candidates: `{k_target}`
- All candidates share identical proposer model-conditioning content:
  `{all_same_input}` (per-message envelope IDs are preserved separately)
- Executor results currently present: `{index['executor_results_ready']}/{k_target}`
- Training rows currently present: `{training_summary.get('rows', 0)}/{k_target}`

For each `candidates/candXX/` directory, read files in numeric order:

1. `01_proposer_exact_input.json`
2. `02_proposer_full_trajectory.json`
3. `03_proposer_raw_submission.txt`
4. `04_generated_harness.json`
5. `05_executor_exact_input.json`
6. `06_executor_full_trajectory.json`
7. `07_executor_reward.json`
8. `08_executor_output_program.py`
9. `09_proposer_grpo_training_row.json`
10. `10_qwen_training_replay_row.json`

The reward-to-training mapping is in `training/reward_advantage_table.json`
and `training/training_contract.json`. A positive advantage increases the
likelihood of the exact H1 assistant decisions that produced that harness; a
negative advantage decreases it. H2 messages are evidence used to compute the
outcome, not the proposer model's token-level training target.

## Candidate outcomes and proposer update direction

{score_table_md}

After the one-step LoRA job completes, the exact argv, checkpoint audit, full
trainer log, merged-weight audit, serve-protocol validation, and final audit
are copied losslessly to `training/11_*` through `training/16_*` by rerunning
this builder.
"""
    atomic_text(out / "README.md", readme)
    atomic_json(out / "artifact_index.json", {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": artifact_inventory(out),
    })
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

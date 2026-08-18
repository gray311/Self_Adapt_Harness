#!/usr/bin/env python3
"""Snapshot and fail-closed verify the source used by long-running campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable


SRC_SUFFIXES = {".py", ".yaml", ".yml"}
# Root-level prose is deliberately excluded.  It can explain a frozen run but
# cannot change its execution, and editing paper/protocol text while a campaign
# is live must not invalidate the byte-level runtime lineage.
ROOT_FILES: set[str] = set()

# Only code that can change search, evaluation, credit, or training belongs in
# the runtime bundle.  Plot/case-study/report scripts are deliberately absent:
# editing a figure while a two-week campaign is running must not invalidate an
# otherwise byte-identical scientific lineage.
RUNTIME_FILES = {
    "scripts/__init__.py",
    "scripts/_outer_round_worker.sh",
    "scripts/runtime/__init__.py",
    "scripts/runtime/audit_trajectories.py",
    "scripts/runtime/capture_shared_anchor.py",
    "scripts/runtime/cascade_promote.py",
    "scripts/runtime/collect_ttt_eval_manifest.py",
    "scripts/context_ablation.sh",
    "scripts/drive_reward_route_inference16_executor.sh",
    "scripts/drive_reward_route_inference16_h1.sh",
    "scripts/drive_ttt_executor_12h.sh",
    "scripts/fresh_campaign.sh",
    "scripts/runtime/hash_h2_package.py",
    "scripts/outer_round.sbatch",
    "scripts/reward_route_inference16_config.sh",
    "scripts/runtime/provenance.py",
    "scripts/runtime/sanitize_grpo_batch.py",
    "scripts/submit_ttt_executor_update.sh",
    "scripts/train_mphi_step.sh",
    "scripts/runtime/ttt_discover_prepare.py",
    "scripts/ttt_executor_eval.sbatch",
    "experiments/why_update_harness/config.sh",
    "experiments/why_update_harness/lib/vllm_pool.sh",
    "experiments/why_update_harness/analysis/causal_effects.py",
    "experiments/why_update_harness/analysis/fair_compute_audit.py",
    "experiments/why_update_harness/analysis/finalize.py",
    "experiments/why_update_harness/analysis/matched_controls.py",
    "experiments/why_update_harness/analysis/pilot_gate.py",
    "experiments/why_update_harness/analysis/trajectory_diagnostics.py",
    "experiments/why_update_harness/slurm/build_plot_env.sbatch",
    "experiments/why_update_harness/slurm/build_plot_env_worker.sh",
    "experiments/why_update_harness/slurm/finalize.sbatch",
    "experiments/why_update_harness/slurm/initialize.sbatch",
    "experiments/why_update_harness/slurm/initialize_worker.sh",
    "experiments/why_update_harness/slurm/pilot_gate.sbatch",
    "experiments/why_update_harness/slurm/prewarm_cache.sbatch",
    "experiments/why_update_harness/slurm/prewarm_cache_worker.sh",
    "experiments/why_update_harness/slurm/round.sbatch",
    "experiments/why_update_harness/slurm/round_worker.sh",
    "experiments/why_update_harness/slurm/task_gate.sbatch",
    "experiments/why_update_harness/slurm/train.sbatch",
    "experiments/why_update_harness/slurm/train_worker.sh",
    "experiments/why_update_harness/submit.sh",
    # HANDOFF §13.G: the fair16-v5 task-major scheduler ran outside the
    # snapshot manifest, so no single immutable identifier named both the
    # round workers and the controller.  Scheduler code is runtime code.
    "experiments/why_update_harness/analysis/round_report.py",
    "experiments/why_update_harness/drive_cp_curve.sh",
    "experiments/why_update_harness/analysis/task_gate.py",
    "experiments/why_update_harness/scheduler/recover_lane.sh",
    "experiments/why_update_harness/scheduler/submit_single_task.sh",
    "experiments/why_update_harness/scheduler/submit_task_major.sh",
}


def _selected(repo: Path) -> Iterable[Path]:
    src = repo / "src"
    for path in sorted(src.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        # Markdown is executable configuration only inside an H1/H2 harness
        # (system prompts and skills), not in src-level documentation.
        is_harness_markdown = path.suffix == ".md" and "harness" in path.parts
        if path.suffix in SRC_SUFFIXES or is_harness_markdown:
            yield path
    for relative in sorted(RUNTIME_FILES):
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime source is missing: {path}")
        yield path
    tests = repo / "tests"
    for path in sorted(tests.rglob("*.py")):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path
    for name in sorted(ROOT_FILES):
        path = repo / name
        if path.is_file():
            yield path
    for rel in (
        "results/baseline_h2_20ev.json",
        "results/baseline_h2_20ev_program_index.json",
        "results/finch_targets.json",
        "results/human_best_references.json",
    ):
        path = repo / rel
        if path.is_file():
            yield path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(repo: Path) -> list[dict]:
    rows = []
    for path in _selected(repo):
        rel = path.relative_to(repo).as_posix()
        rows.append({"path": rel, "sha256": _sha(path), "bytes": path.stat().st_size})
    return rows


def _bundle(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode() + b"\0" + row["sha256"].encode() + b"\n")
    return digest.hexdigest()


def snapshot(repo: Path, manifest: Path, snapshot_dir: Path) -> dict:
    repo = repo.resolve()
    rows = _inventory(repo)
    snapshot_dir = snapshot_dir.resolve()
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_dir.exists():
        # Crash recovery: the directory rename may have completed just before
        # the manifest write.  Reuse it only when every byte is identical.
        copied = _inventory(snapshot_dir)
        if copied != rows:
            raise FileExistsError(
                f"different immutable source snapshot already exists: {snapshot_dir}"
            )
    else:
        staging_parent = Path(tempfile.mkdtemp(
            prefix=f".{snapshot_dir.name}.staging-", dir=snapshot_dir.parent
        ))
        staging = staging_parent / "snapshot"
        try:
            for row in rows:
                source = repo / row["path"]
                target = staging / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copied = _inventory(staging)
            if copied != rows:
                raise RuntimeError("source snapshot copy failed content verification")
            os.replace(staging, snapshot_dir)
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)
    payload = {
        "schema": "runtime-source/1.0",
        "repo": str(repo),
        "bundle_sha256": _bundle(rows),
        "file_count": len(rows),
        "files": rows,
        "snapshot_dir": str(snapshot_dir.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, manifest)
    return payload


def verify(manifest: Path) -> dict:
    payload = json.loads(manifest.read_text())
    repo = Path(payload["repo"]).resolve()
    expected = payload["files"]
    actual = _inventory(repo)
    errors = []
    expected_paths = [row["path"] for row in expected]
    actual_paths = [row["path"] for row in actual]
    if actual_paths != expected_paths:
        errors.append("selected source-file set changed")
    expected_by_path = {row["path"]: row for row in expected}
    for row in actual:
        old = expected_by_path.get(row["path"])
        if old is not None and row["sha256"] != old["sha256"]:
            errors.append(f"changed: {row['path']}")
    if _bundle(actual) != payload.get("bundle_sha256"):
        errors.append("bundle digest changed")
    snapshot_dir = Path(payload["snapshot_dir"]).resolve()
    if not snapshot_dir.is_dir():
        errors.append("immutable snapshot directory is missing")
    else:
        snap = _inventory(snapshot_dir)
        if snap != expected or _bundle(snap) != payload.get("bundle_sha256"):
            errors.append("immutable snapshot bytes changed")
    if errors:
        raise SystemExit(
            "runtime source verification failed for " + str(manifest) + ": "
            + "; ".join(errors[:20])
        )
    return {
        "status": "verified",
        "manifest": str(manifest.resolve()),
        "bundle_sha256": payload["bundle_sha256"],
        "file_count": len(actual),
    }


def verify_snapshot(manifest: Path) -> dict:
    """Fail-closed check of the IMMUTABLE SNAPSHOT only.

    Per-job verification (HANDOFF §13.G): workers must refuse to run from a
    hot-patched snapshot, while the live repository is free to evolve during
    a campaign — so unlike ``verify`` this never inventories the repo.
    """

    payload = json.loads(manifest.read_text())
    expected = payload["files"]
    snapshot_dir = Path(payload["snapshot_dir"]).resolve()
    errors: list[str] = []
    if not snapshot_dir.is_dir():
        errors.append("immutable snapshot directory is missing")
    else:
        snap = _inventory(snapshot_dir)
        expected_by_path = {row["path"]: row for row in expected}
        snap_paths = {row["path"] for row in snap}
        for row in snap:
            old = expected_by_path.get(row["path"])
            if old is None:
                errors.append(f"unexpected: {row['path']}")
            elif row["sha256"] != old["sha256"]:
                errors.append(f"hot-patched: {row['path']}")
        for row in expected:
            if row["path"] not in snap_paths:
                errors.append(f"missing: {row['path']}")
        if not errors and _bundle(snap) != payload.get("bundle_sha256"):
            errors.append("bundle digest changed")
    if errors:
        raise SystemExit(
            "immutable snapshot verification failed for "
            + str(manifest) + ": " + "; ".join(errors[:20])
        )
    return {
        "status": "verified",
        "manifest": str(manifest.resolve()),
        "snapshot_dir": str(snapshot_dir),
        "bundle_sha256": payload["bundle_sha256"],
        "file_count": len(expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("snapshot")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--snapshot-dir", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    check_snapshot = sub.add_parser("verify-snapshot")
    check_snapshot.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.repo, args.manifest, args.snapshot_dir)
    elif args.command == "verify-snapshot":
        result = verify_snapshot(args.manifest)
    else:
        result = verify(args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

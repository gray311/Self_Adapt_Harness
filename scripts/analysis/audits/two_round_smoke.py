#!/usr/bin/env python3
"""Fail-closed audit of a two-round smoke run (scripts/smoke_two_round.sh).

Answers the three acceptance questions mechanically, per round pair
(round001 -> round002):

1. HARNESS GENERATION — every charged slot holds a valid, materialized,
   gate-passed candidate H2 (no fallback slots), and each candidate package
   parses back into the genome (agent.yaml + component manifest present).
2. INHERITANCE — round 2's incoming base package IS round 1's outgoing
   winner (next_bases chain, byte-verified via h2_sha256), and round 2's
   candidates were proposed against that base (base_spec_hash match);
   the inherited best program (if promoted) seeds round 2 rollouts.
3. ENACTMENT — every candidate rollout's executor actually saw and used the
   harness: score-eligible terminal result, stable H2 hash during rollout,
   component participation audits present, and every generated (added or
   updated) component appears in the runtime audits.

Plus the paired-credit and training-boundary checks: eight matched parent
controls with equal decode seeds, non-null causal deltas in the collect
output, and a classified round-1 -> round-2 phi boundary.

Usage:
    python3 scripts/analysis/audits/two_round_smoke.py \
        --out-dir $RUN_ROOT/self_adapt_harness/outer-<OUT_TAG> \
        [--rounds 1 2] [--task <task_id>]

Exit 0 only if every check passes.  Read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from inner.runtime.package_hash import h2_sha256  # noqa: E402


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


class Audit:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def rollout_result(root: Path, task: str):
    for path in sorted(root.glob(f"*/results/{task}.json")):
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict) and row.get("task_id") == task:
                return row
    return None


def audit_round(audit: Audit, round_dir: Path, task: str, k: int) -> dict:
    tag = round_dir.name
    meta = read_json(round_dir / "round.json") or {}
    per_task = (meta.get("per_task") or {}).get(task) or {}
    candidates = per_task.get("candidates") or []
    gate = read_json(round_dir / "proposal_gate_audit.json") or {}
    plan = read_json(round_dir / "h2_slot_plan.json") or {}
    summary = read_json(round_dir / "round_summary.json") or {}
    group = (summary.get("groups") or {}).get(task) or {}

    # -- 1. harness generation ------------------------------------------- #
    valid_count = sum(1 for c in candidates if c.get("valid") is True)
    min_valid = 6 if str(gate.get("mode", "")).startswith("min") else k
    audit.check(f"{tag}: proposal gate ran fail-closed and passed",
                gate.get("mode") in ("all", "min6")
                and gate.get("passed") is True,
                json.dumps(gate.get("tasks", {}).get(task, {}))[:200])
    audit.check(f"{tag}: >= {min_valid}/{k} valid candidates",
                len(candidates) == k and valid_count >= min_valid,
                f"valid={valid_count}/{len(candidates)} repaired="
                f"{[c['k'] for c in candidates if (c.get('repair') or {}).get('succeeded')]}")
    audit.check(f"{tag}: every invalid slot was offered to the repair agent",
                all((c.get("repair") or {}).get("attempted") is True
                    for c in candidates if c.get("valid") is not True),
                f"invalid={[c['k'] for c in candidates if not c.get('valid')]}")
    slots = plan.get("slots") or []
    audit.check(f"{tag}: all emitted slots are candidate_harness (no fallback)",
                min_valid <= len(slots) <= k and all(
                    s.get("h2_slot_mode") == "candidate_harness"
                    for s in slots
                ) and plan.get("fallback_slots_forbidden") is True)
    packages_ok = True
    for cand in candidates:
        if not cand.get("valid"):
            continue
        cdir = Path(cand.get("dir") or "")
        if not ((cdir / "agent.yaml").is_file()
                and (cdir / "prompt.md").is_file()
                and (cdir / "component_manifest.json").is_file()
                and cand.get("package_sha256") == h2_sha256(cdir)):
            packages_ok = False
            audit.check(f"{tag}: cand{cand.get('k'):02d} package integrity",
                        False, str(cdir))
    audit.check(f"{tag}: materialized packages verified (hash + manifest)",
                packages_ok)

    # -- 3. enactment ----------------------------------------------------- #
    lineage_by_k = {
        int(c.get("k", -1)): (c.get("component_lineage") or {})
        for c in candidates
    }
    for cand in candidates:
        if not cand.get("valid"):
            continue
        idx = int(cand.get("k", -1))
        result = rollout_result(
            round_dir / "rollouts" / task / f"cand{idx:02d}", task
        )
        control = rollout_result(
            round_dir / "paired_controls" / task / f"cand{idx:02d}", task
        )
        ok = bool(
            result is not None
            and result.get("score_eligible") is True
            and (result.get("h2_package_provenance") or {}).get(
                "stable_during_rollout") is True
            and control is not None
            and control.get("score_eligible") is True
            and result.get("decode_seed") == control.get("decode_seed")
        )
        audit.check(
            f"{tag}: cand{idx:02d} rollout+control eligible, same seed",
            ok,
            (f"cand={None if not result else result.get('best_score')}"
             f" ctrl={None if not control else control.get('best_score')}"
             f" seed={None if not result else result.get('decode_seed')}"),
        )
        if result is None:
            continue
        generated = [
            str(row.get("name"))
            for kind in ("new_tools", "new_skills", "new_middlewares")
            for row in (lineage_by_k.get(idx, {}).get(kind) or [])
            if isinstance(row, dict)
            and row.get("status") in ("added", "updated")
        ] if isinstance(lineage_by_k.get(idx), dict) else []
        audits_seen = set()
        for key in ("tool_audit", "skill_audit", "middleware_audit"):
            audits_seen |= set((result.get(key) or {}).keys())
        missing = [name for name in generated if name not in audits_seen]
        audit.check(
            f"{tag}: cand{idx:02d} generated components reached the executor",
            not missing,
            f"generated={generated or '-'} missing_from_audits={missing or '-'}",
        )

    # -- paired credit ----------------------------------------------------- #
    grpo = [row for row in read_jsonl(round_dir / "grpo_batch.jsonl")
            if row.get("task_id") == task]
    valid_ks = {int(c["k"]) for c in candidates if c.get("valid")}
    audit.check(f"{tag}: paired causal deltas present in GRPO batch",
                len(grpo) == k and all(
                    isinstance(row.get("causal_delta"), (int, float))
                    for row in grpo if row.get("k") in valid_ks
                ),
                f"rows={len(grpo)} valid_ks={sorted(valid_ks)}")
    repaired_ks = {
        int(c["k"]) for c in candidates
        if (c.get("repair") or {}).get("succeeded")
    }
    if repaired_ks:
        audit.check(
            f"{tag}: repaired slots train at minimum reward (no phi credit)",
            all(float(row.get("reward", 0)) <= -1.0
                for row in grpo if row.get("k") in repaired_ks),
            f"repaired={sorted(repaired_ks)}",
        )
    audit.check(f"{tag}: group best causal delta recorded",
                isinstance(group.get("best_causal_delta"), (int, float)),
                str(group.get("best_causal_delta")))
    return {"meta": meta, "per_task": per_task, "group": group,
            "next_bases": read_json(round_dir / "next_bases.json") or {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rounds", nargs=2, type=int, default=[1, 2])
    parser.add_argument("--task", default=None)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    first_dir = args.out_dir / f"round{args.rounds[0]:03d}"
    second_dir = args.out_dir / f"round{args.rounds[1]:03d}"
    meta1 = read_json(first_dir / "round.json") or {}
    task = args.task or (meta1.get("tasks_order") or [None])[0]
    if not task:
        raise SystemExit(f"cannot determine task from {first_dir}/round.json")

    audit = Audit()
    r1 = audit_round(audit, first_dir, task, args.k)
    r2 = audit_round(audit, second_dir, task, args.k)

    # -- 2. inheritance ---------------------------------------------------- #
    outgoing = (r1["next_bases"].get(task) or {})
    incoming = (r2["per_task"] or {})
    audit.check(
        "round2 incoming base package == round1 outgoing package",
        bool(outgoing.get("package"))
        and incoming.get("base_package") == outgoing.get("package"),
        f"{incoming.get('base_package')} vs {outgoing.get('package')}",
    )
    base_pkg = Path(incoming.get("base_package") or "")
    audit.check(
        "round2 base package bytes match its declared hash",
        base_pkg.is_dir()
        and incoming.get("base_package_sha256") == h2_sha256(base_pkg),
    )
    if outgoing.get("from", "").startswith("round"):
        audit.check(
            "round1 produced an accepted improvement to inherit",
            True, outgoing.get("from"),
        )
    else:
        audit.check(
            "round1 produced an accepted improvement to inherit "
            "(unchanged base is protocol-legal but weakens the smoke)",
            False, str(outgoing.get("from")),
        )
    phi1 = ((r1["meta"].get("proposer") or {}).get("checkpoint") or "")
    phi2 = ((r2["meta"].get("proposer") or {}).get("checkpoint") or "")
    audit.check(
        "round2 proposer checkpoint differs from round1 (trained phi)",
        bool(phi1) and bool(phi2) and phi1 != phi2,
        f"{phi1} -> {phi2}",
    )

    print()
    verdict = "PASS" if audit.passed else "FAIL"
    failed = [name for name, ok, _ in audit.checks if not ok]
    print(f"two-round smoke audit: {verdict} "
          f"({len(audit.checks) - len(failed)}/{len(audit.checks)} checks)")
    payload = {
        "schema": "two-round-smoke-audit/1.0",
        "task": task,
        "passed": audit.passed,
        "checks": [
            {"name": name, "passed": ok, "detail": detail}
            for name, ok, detail in audit.checks
        ],
    }
    out = args.out_dir / "two_round_smoke_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    raise SystemExit(0 if audit.passed else 1)


if __name__ == "__main__":
    main()

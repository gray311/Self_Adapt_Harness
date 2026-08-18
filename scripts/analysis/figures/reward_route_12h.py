#!/usr/bin/env python3
"""Plot the three-route score-vs-rollout figures.

Only measured local trajectories are drawn:

* proposer weights: the complete observed proposer campaign for the paper figure;
* context/analyzer: the longest completed frozen-weight context campaign;
* executor weights: an isolated TTT-Discover-style state, whose manifests
  require a real merged executor checkpoint after the initial point.

The companion ``*_data.json`` file is the authoritative provenance ledger for
every plotted point.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import textwrap
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get(
    "RUN_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/runs"))
ROOT = RUN / "self_adapt_harness"
TTT_STATE_ROOT = Path(os.environ.get(
    "TTT_STATE_ROOT", str(ROOT / "ttt_discover_12h")))
TTT_AHC039_STATE_ROOT = Path(os.environ.get(
    "TTT_AHC039_STATE_ROOT", str(ROOT / "ttt_discover_clean20")))
TTT_MODEL_TAG = os.environ.get("TTT_MODEL_TAG", "unknown")
TTT_BATCH_K = int(os.environ.get("TTT_BATCH_K", "32"))
TTT_EPLB_STATE_ROOT = Path(os.environ.get(
    "TTT_EPLB_STATE_ROOT", str(TTT_STATE_ROOT)))
TTT_EXTRA_STATE_ROOT = Path(os.environ.get(
    "TTT_EXTRA_STATE_ROOT", str(ROOT / "ttt_discover_sota7_extra_k8")))
_TTT_STATE_ROOT_EXPLICIT = "TTT_STATE_ROOT" in os.environ
_TTT_AHC039_STATE_ROOT_EXPLICIT = "TTT_AHC039_STATE_ROOT" in os.environ
_TTT_EPLB_STATE_ROOT_EXPLICIT = "TTT_EPLB_STATE_ROOT" in os.environ
_TTT_EXTRA_STATE_ROOT_EXPLICIT = "TTT_EXTRA_STATE_ROOT" in os.environ
_TTT_BATCH_K_EXPLICIT = "TTT_BATCH_K" in os.environ
_TTT_MODEL_TAG_EXPLICIT = "TTT_MODEL_TAG" in os.environ
LEGACY_GUARD_AUDIT = ROOT / "legacy_semantic_guard_audit.json"
_LEGACY_GUARD_CACHE: tuple[dict[str, Any], dict[tuple[str, str], bool]] | None = None
H2_WARM_START_PROVENANCE = REPO / "results/h2_warm_start_provenance.json"
HUMAN_BEST_REFERENCE = REPO / "results/human_best_references.json"
SOTA_ENDPOINT_VALIDATION = Path(os.environ.get(
    "SOTA_ENDPOINT_VALIDATION",
    str(ROOT / "sota7_endpoint_validation" / "results.json"),
))
_ENDPOINT_VALIDATION_CACHE: dict[str, Any] | None = None

TASKS = [
    ("eft__math__erdos_min_overlap", "Erdős min-overlap †", "erdos"),
    ("eft__math__circle_packing", "Circle packing (n=26)", "circle"),
    ("eft__math__hadamard_maximal_det", "Hadamard max-det", "hadamard"),
    ("eft__math__first_autocorr_ineq", "Autocorrelation I †", "ac1"),
    ("eft__math__second_autocorr_ineq", "Autocorrelation II", "ac2"),
    ("eft__ahc_simpletes__ahc039", "AHC039", "ahc039"),
]

SOTA_TASKS = [
    ("eft__math__hadamard_maximal_det", "Hadamard", "hadamard"),
    ("eft__ahc_simpletes__ahc039", "AHC039", "ahc039"),
    ("eft__ahc_simpletes__ahc058", "AHC058", "ahc058"),
    ("adrs__eplb", "EPLB", "eplb"),
    ("adrs__prism", "PRISM", "prism"),
    ("adrs__llm_sql", "LLM-SQL", "llmsql"),
    ("adrs__txn_scheduling", "Txn", "txnsched"),
]

# The five-task scope requested for the clean reward-routing
# comparison.  Keep this explicit rather than relying on a positional slice:
# task membership is part of the scientific protocol and must survive later
# additions or reorderings of SOTA_TASKS.
SOTA5_TASK_IDS = (
    "eft__ahc_simpletes__ahc039",
    "eft__ahc_simpletes__ahc058",
    "adrs__eplb",
    "adrs__prism",
    "adrs__llm_sql",
)
SOTA5_TASKS = [row for row in SOTA_TASKS if row[0] in SOTA5_TASK_IDS]

# Main-text illustrations chosen after inspecting the interim seven-task
# campaign.  They are not a replacement for the full predeclared 1x7 audit and
# must never be summarized as an unbiased four-task benchmark average.
SOTA4_TASK_IDS = (
    "eft__math__hadamard_maximal_det",
    "adrs__eplb",
    "adrs__prism",
    "adrs__txn_scheduling",
)
SOTA4_TASKS = [row for row in SOTA_TASKS if row[0] in SOTA4_TASK_IDS]

SOTA_CONTEXT_WORKSPACES = {
    "eft__math__hadamard_maximal_det": "context_sota7_extra_guarded",
    "eft__ahc_simpletes__ahc039": "context_sota5_ahc_clean",
    "eft__ahc_simpletes__ahc058": os.environ.get(
        "SOTA_AHC058_CONTEXT_WORKSPACE",
        "context_sota7_ahc058_analysis_required_v1"),
    "adrs__eplb": os.environ.get(
        "SOTA_SYS_CONTEXT_WORKSPACE", "context_sota5_sys_guarded"),
    "adrs__prism": os.environ.get(
        "SOTA_PRISM_CONTEXT_WORKSPACE", "context_sota7_rewardfix_v1"),
    "adrs__llm_sql": os.environ.get(
        "SOTA_SYS_CONTEXT_WORKSPACE", "context_sota5_sys_guarded"),
    "adrs__txn_scheduling": "context_sota7_extra_guarded",
}
SOTA_SQL_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_SQL_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota7-sql-clean-v2"),
))
SOTA_SQL_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_SQL_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota7_sql_clean_v2"),
))
SOTA_TXN_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_TXN_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota7-txn-clean-v1"),
))
SOTA_TXN_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_TXN_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota7_txn_clean_v1"),
))
SOTA_PRISM_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_PRISM_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota7-prism-clean-v1"),
))
SOTA_PRISM_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_PRISM_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota7_prism_clean_v1"),
))
SOTA_HADAMARD_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_HADAMARD_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota7-hadamard-rewardfix-v1"),
))
SOTA_HADAMARD_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_HADAMARD_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota7_hadamard_rewardfix_v1"),
))
SOTA_AHC058_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_AHC058_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota7-ahc058-rewardfix-v1"),
))
SOTA_AHC058_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_AHC058_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota7_ahc058_rewardfix_v1"),
))
SOTA_AHC039_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_AHC039_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota5-ahc039-clean-v1"),
))
SOTA_AHC039_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_AHC039_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota5_ahc039_clean_v1"),
))
SOTA_EPLB_PROPOSER_OUTER_ROOT = Path(os.environ.get(
    "SOTA_EPLB_PROPOSER_OUTER_ROOT",
    str(ROOT / "outer-proposer-sota5-eplb-clean-v1"),
))
SOTA_EPLB_PROPOSER_WORKSPACE = Path(os.environ.get(
    "SOTA_EPLB_PROPOSER_WORKSPACE",
    str(ROOT / "proposer_sota5_eplb_clean_v1"),
))

SOTA_CLEAN_PROPOSER_ROUTES = {
    "eft__math__hadamard_maximal_det": (
        SOTA_HADAMARD_PROPOSER_WORKSPACE,
        SOTA_HADAMARD_PROPOSER_OUTER_ROOT,
    ),
    "eft__ahc_simpletes__ahc039": (
        SOTA_AHC039_PROPOSER_WORKSPACE,
        SOTA_AHC039_PROPOSER_OUTER_ROOT,
    ),
    "eft__ahc_simpletes__ahc058": (
        SOTA_AHC058_PROPOSER_WORKSPACE,
        SOTA_AHC058_PROPOSER_OUTER_ROOT,
    ),
    "adrs__eplb": (
        SOTA_EPLB_PROPOSER_WORKSPACE,
        SOTA_EPLB_PROPOSER_OUTER_ROOT,
    ),
    "adrs__prism": (
        SOTA_PRISM_PROPOSER_WORKSPACE,
        SOTA_PRISM_PROPOSER_OUTER_ROOT,
    ),
    "adrs__llm_sql": (
        SOTA_SQL_PROPOSER_WORKSPACE,
        SOTA_SQL_PROPOSER_OUTER_ROOT,
    ),
    "adrs__txn_scheduling": (
        SOTA_TXN_PROPOSER_WORKSPACE,
        SOTA_TXN_PROPOSER_OUTER_ROOT,
    ),
}

MATCHED_ARM_LIMITATIONS = {
    "eft__math__erdos_min_overlap": (
        "historical proposer/context A/B driver ratcheted combined_score as lower-is-better"
    ),
    "eft__math__first_autocorr_ineq": (
        "historical proposer/context A/B driver ratcheted combined_score as lower-is-better"
    ),
}

# All values are evaluator ``combined_score`` and therefore higher-is-better.
# The first element is the shared fixed-H2 score.  The second is retained for
# the legacy 2x3 figure; SOTA layouts use HUMAN_BEST below as their y=1 anchor.
ANCHORS = {
    "eft__math__erdos_min_overlap": (0.769452, 0.999974),
    "eft__math__circle_packing": (0.364237, 1.000373),
    "eft__math__hadamard_maximal_det": (0.14327485380116958, 0.576400),
    "eft__math__first_autocorr_ineq": (0.991237, 1.001437),
    "eft__math__second_autocorr_ineq": (0.954836, 1.056813),
    "eft__ahc_simpletes__ahc039": (2.476554, 557_168 / 225_000),
    "eft__ahc_simpletes__ahc058": (0.298859, 525_286_896 / 4.5e8),
    "adrs__eplb": (0.1265392786992853, 0.1270),
    "adrs__prism": (24.021666874072007, 24.70),
    "adrs__llm_sql": (0.09343955531989306, 0.7341),
    "adrs__txn_scheduling": (3610.1083032490974, 4761.90),
}

# Frozen task-level human references, expressed in evaluator combined_score.
# A direct score/human ratio stays monotone even when fixed H2 already exceeds
# the human result (EPLB, PRISM, and Txn); gap-closed normalization would not.
_HUMAN_BEST_PAYLOAD = json.loads(HUMAN_BEST_REFERENCE.read_text())
assert _HUMAN_BEST_PAYLOAD.get("direction") == "higher_is_better"
HUMAN_BEST = {
    task: float(row["human_best_combined_score"])
    for task, row in _HUMAN_BEST_PAYLOAD["tasks"].items()
}

# Published Qwen3-8B endpoints reported by TTT-Discover at 50 x 512 = 25,600
# executor rollouts.  These are context points only, not part of our reproduced
# executor curve and not used in any common-budget metric.
TTT_PUBLISHED_BUDGET = 25_600
TTT_PUBLISHED = {
    "eft__math__erdos_min_overlap": 0.380922 / 0.380932,
    "eft__math__first_autocorr_ineq": 1.505293 / 1.50525,
    "eft__math__second_autocorr_ineq": 0.9472 / 0.896280,
}


def h2_warm_start_record(task: str) -> dict[str, Any]:
    payload = json.loads(H2_WARM_START_PROVENANCE.read_text())
    return payload["tasks"][task]


def h2_warm_start_point(task: str, marker: str) -> dict[str, Any]:
    record = h2_warm_start_record(task)
    ledger = record.get("ledger") or {}
    return {
        "x": 1,
        "score": ANCHORS[task][0],
        marker: None,
        "source": "shared_h2_warm_start",
        "source_summary_all": record["source_summary_all"],
        "program_sha256": record["h2_best_program_sha256"],
        "launched": 1,
        "max_evals_per_trajectory": 20,
        "charged_evaluator_call_budget": 20,
        "recorded_evaluator_calls": int(ledger.get("evaluator_calls") or 0),
        "recorded_executor_model_calls": int(ledger.get("llm_calls") or 0),
        "recorded_sandbox_seconds": float(ledger.get("sandbox_seconds") or 0.0),
        "program_inherited": False,
    }


def normalize(task: str, score: float) -> float:
    seed, reference = ANCHORS[task]
    return (score - seed) / (reference - seed)


def display_score(task: str, combined_score: float) -> float:
    """Convert evaluator combined_score to the task's reporting scale."""
    if task == "eft__math__erdos_min_overlap":
        return 0.380922 / combined_score
    if task == "eft__math__first_autocorr_ineq":
        return 1.505293 / combined_score
    if task == "eft__math__second_autocorr_ineq":
        return combined_score * 0.896280
    if task == "eft__math__circle_packing":
        return combined_score * 2.635
    if task == "eft__ahc_simpletes__ahc039":
        return combined_score * 225_000
    if task == "eft__ahc_simpletes__ahc058":
        return combined_score * 4.5e8
    return combined_score


def format_display_score(task: str, raw_score: float) -> str:
    if task in ("eft__ahc_simpletes__ahc039", "eft__ahc_simpletes__ahc058"):
        return f"{raw_score:,.0f}"
    if task == "adrs__prism":
        # The three audited endpoints can all round to 26.26 at table
        # precision.  Keep four decimals in the zoom inset so the observed
        # ordering remains visible without changing the task reporting format.
        return f"{raw_score:.4f}"
    if task == "adrs__llm_sql":
        return f"{raw_score:.4f}"
    if task == "adrs__eplb":
        # Even six decimals rounds the live 0.1269996 context near-miss to
        # 0.127000 and visually turns it into a tie.  Preserve seven places in
        # the endpoint inset; the paper table can still use its usual scale.
        return f"{raw_score:.7f}"
    if task == "adrs__txn_scheduling":
        return f"{raw_score:,.2f}"
    return f"{raw_score:.6f}"


def reported_proposer_score(
    task: str, proposer_points: list[dict[str, Any]]
) -> float:
    """Return the locally ledgered endpoint used by the final report."""
    if len(proposer_points) <= 1:
        raise RuntimeError(f"{task} proposer campaign has no adaptive point")
    return float(proposer_points[-1]["score"])


def logged_phi_rounds(paths: list[str]) -> dict[int, dict[str, str]]:
    """Resolve rounds explicitly logged as proposer-driven."""
    out: dict[int, dict[str, str]] = {}
    rx = re.compile(r"round(\d+) propose \(phi=([^)]+)\)")
    job_rx = re.compile(r"\bjob (\d+)\b")
    timestamp_rx = re.compile(r"^\[([^]]+)\]")
    for path in sorted(set(paths)):
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        pending: int | None = None
        for line in text.splitlines():
            match = rx.search(line)
            if match:
                pending = int(match.group(1))
                # A failed retry driver can mention the same round without ever
                # obtaining a job ID.  Preserve a successful record already
                # found in another resumable driver instead of erasing its
                # compute provenance (AHC039 round720 is one such case).
                previous = out.get(pending, {})
                timestamp_match = timestamp_rx.search(line)
                out[pending] = {
                    "phi": match.group(2),
                    "driver": path,
                    "submitted_at": (
                        timestamp_match.group(1) if timestamp_match else ""
                    ),
                }
                if previous.get("job"):
                    out[pending]["job"] = previous["job"]
                    # Keep the timestamp/driver associated with the successful
                    # submission when a later retry log merely repeats the
                    # same round ID without obtaining a job.
                    out[pending]["submitted_at"] = previous.get(
                        "submitted_at", out[pending]["submitted_at"]
                    )
                    out[pending]["driver"] = previous.get("driver", path)
                continue
            job_match = job_rx.search(line)
            if pending is not None and job_match:
                out[pending]["job"] = job_match.group(1)
                pending = None
    return out


def phi_rounds() -> dict[int, dict[str, str]]:
    # The historical full-campaign curve is explicitly the fresh_all campaign.
    # Scanning every driver under RUN_ROOT creates round-ID collisions with
    # isolated context/clean-retry namespaces that use a different outer root.
    paths = glob.glob(str(ROOT / "fresh_all" / "*" / "driver*.log"))
    return logged_phi_rounds(paths)


def candidate_passes_posthoc_guard(rd: Path, task: str, k: int) -> bool:
    """Apply paper-protocol validity checks to legacy summaries.

    New runs receive these guards inside ``_eval_worker``.  Historical proposer
    rounds predate them, so the plot independently rejects the same known
    partial-success PRISM failure mode instead of grandfathering it in.
    """
    summaries = list((rd / "rollouts" / task / f"cand{k:02d}").glob(
        "*/summary.json"))
    if not summaries:
        return task not in ("adrs__eplb", "adrs__txn_scheduling")
    try:
        payload = json.loads(max(
            summaries, key=lambda path: path.stat().st_mtime).read_text())
        payload = payload[0] if isinstance(payload, list) else payload
    except Exception:
        return True
    if task == "adrs__prism":
        success_rate = float(
            (payload.get("best_metrics") or {}).get("success_rate", 0.0))
        return success_rate >= 1.0 - 1e-12
    if task in ("adrs__eplb", "adrs__txn_scheduling"):
        guard_field = {
            "adrs__eplb": "eplb_topology_guard",
            "adrs__txn_scheduling": "txn_legality_guard",
        }[task]
        guarded_online_roots = {
            "adrs__eplb": [ROOT / "outer-context-sota5-sys-guarded"],
            "adrs__txn_scheduling": [
                ROOT / "outer-context-sota7-extra-guarded",
                SOTA_TXN_PROPOSER_OUTER_ROOT,
            ],
        }
        program = str(payload.get("best_program") or "")
        digest = hashlib.sha256(program.encode()).hexdigest() if program else ""
        if rd.parent.resolve() in {
            root.resolve() for root in guarded_online_roots[task]
        }:
            proof = protocol_guard_proof()
            fields = proof.get("fields") or {}
            # These isolated rounds were launched only after the current
            # output-level topology guard passed its proof job.  Requiring the
            # legacy hash replay here would incorrectly drop every new valid
            # context candidate, because that index intentionally contains
            # only pre-guard historical programs.
            _, index = legacy_semantic_guard_audit()
            if (task, digest) in index:
                return index[(task, digest)]
            # Programs from a round created after the current proof ran were
            # checked online by this exact worker. Earlier context rounds must
            # appear in the hash-indexed replay above; fail closed otherwise.
            try:
                created = str(json.loads((rd / "round.json").read_text())["created"])
                created_at = datetime.strptime(
                    created, "%Y%m%d-%H%M%S"
                ).replace(tzinfo=ZoneInfo("America/Los_Angeles"))
                proof_at = datetime.fromisoformat(str(fields["validated_at"]))
                launched_after_proof = created_at > proof_at
            except Exception:
                launched_after_proof = False
            return bool(
                launched_after_proof
                and proof.get("matches_current_worker")
                and fields.get(guard_field) == "ok"
            )
        _, index = legacy_semantic_guard_audit()
        # Fail closed: legacy EPLB summaries predate the online guard and enter
        # the paper curve only after their exact program hash was replayed
        # through the current output-level topology checker.
        return index.get((task, digest), False)
    return True


def legacy_semantic_guard_audit(
) -> tuple[dict[str, Any], dict[tuple[str, str], bool]]:
    global _LEGACY_GUARD_CACHE
    if _LEGACY_GUARD_CACHE is not None:
        return _LEGACY_GUARD_CACHE
    payload: dict[str, Any] = {}
    index: dict[tuple[str, str], bool] = {}
    if LEGACY_GUARD_AUDIT.exists():
        try:
            payload = json.loads(LEGACY_GUARD_AUDIT.read_text())
            for row in payload.get("records") or []:
                digest = row.get("program_sha256")
                task = row.get("task")
                if digest and task:
                    index[(str(task), str(digest))] = bool(row.get("valid"))
        except Exception:
            payload, index = {}, {}
    _LEGACY_GUARD_CACHE = payload, index
    return _LEGACY_GUARD_CACHE


def authoritative_rollout_score(rd: Path, task: str, k: int) -> float | None:
    """Mirror the fixed reward loader's terminal-summary precedence."""
    root = rd / "rollouts" / task / f"cand{k:02d}"
    best: float | None = None
    saw_terminal = False
    for source in root.glob("*/summary.json"):
        try:
            payload = json.loads(source.read_text())
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, dict) or row.get("task_id") != task:
                    continue
                saw_terminal = True
                if row.get("best_score") is not None:
                    score = float(row["best_score"])
                    best = score if best is None else max(best, score)
        except Exception:
            continue
    if saw_terminal:
        return best
    # Only a genuinely interrupted run with no terminal task row may use its
    # atomic wall-safe checkpoint.
    for source in root.glob(f"*/checkpoints/{task}.json"):
        try:
            score = float(json.loads(source.read_text())["best_score"])
            best = score if best is None else max(best, score)
        except Exception:
            continue
    return best


def round_group(
    round_id: int, task: str, outer_root: Path | None = None
) -> tuple[
    list[float], int, int, int, int, int, dict[str, float | int], str
] | None:
    rd = (outer_root or (ROOT / "outer")) / f"round{round_id:03d}"
    summary = rd / "round_summary.json"
    if not summary.exists():
        return None
    try:
        group = json.loads(summary.read_text())["groups"].get(task)
    except Exception:
        return None
    if not group:
        return None
    scores = []
    for row in group.get("rows") or []:
        if not row.get("valid") or row.get("score") is None:
            continue
        k = int(row.get("k", -1))
        terminal = authoritative_rollout_score(rd, task, k)
        credited = float(row["score"])
        agrees = terminal is not None and abs(terminal - credited) <= max(
            1e-12, 1e-12 * abs(credited)
        )
        if not agrees:
            if os.environ.get("ALLOW_CONTAMINATED_REWARD_FALLBACK") == "1":
                continue
            raise RuntimeError(
                f"{rd.name}/{task}/cand{k:02d}: credited score {credited} "
                f"does not match authoritative terminal score {terminal}"
            )
        if candidate_passes_posthoc_guard(rd, task, k):
            scores.append(credited)
    # A log is created when an executor trajectory is launched, including a
    # trajectory that later fails.  This excludes invalid harness proposals
    # that never reached the executor.
    logs = list((rd / "rollout_logs").glob(f"{task}-cand*.log"))
    launched = len(logs)
    if launched == 0:
        launched = len(group.get("rows") or [])
    # ``round_summary.rows`` contains only materialized candidates.  Charging
    # that count as proposal compute would silently drop H1 attempts that
    # failed YAML parsing, review, or materialization.  ``round.json`` retains
    # all attempts and the number of model calls made inside each H1 run.
    proposed = len(group.get("rows") or [])
    h1_model_calls = 0
    reviewer_model_calls = 0
    max_evals = 0
    round_ledger = rd / "round.json"
    if round_ledger.exists():
        try:
            round_payload = json.loads(round_ledger.read_text())
            max_evals = int(round_payload.get("max_evals") or 0)
            per_task = round_payload["per_task"][task]
            candidates = per_task.get("candidates") or []
            if candidates:
                proposed = len(candidates)
                h1_model_calls = sum(
                    int(candidate.get("llm_calls") or 0)
                    for candidate in candidates
                )
                # ``rounds`` is the number of materialized frozen-M0 repair
                # responses before the reviewer accepted or rejected code.
                # A request that failed before returning a response is not
                # recoverable, so this remains a recorded lower bound.
                reviewer_model_calls = sum(
                    int(review.get("rounds") or 0)
                    for candidate in candidates
                    for review in candidate.get("review_log") or []
                )
        except Exception:
            pass
    execution_ledger = trajectory_execution_ledger(rd / "rollouts" / task)
    return (scores, launched, proposed, h1_model_calls, reviewer_model_calls,
            max_evals,
            execution_ledger, str(summary))


def failed_round_attempt(
    round_id: int, task: str, outer_root: Path
) -> tuple[
    list[float], int, int, int, int, int, dict[str, float | int], str
] | None:
    """Recover costs from a launched proposer round that never collected.

    It contributes no score evidence, but every materialized executor launch,
    H1 attempt, evaluator-call cap, model-call ledger, and outer job still
    belongs on the efficiency/cost ledger.
    """
    rd = outer_root / f"round{round_id:03d}"
    meta_path = rd / "round.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if task not in (meta.get("tasks_order") or []):
            return None
        candidates = ((meta.get("per_task") or {}).get(task) or {}).get(
            "candidates") or []
    except Exception:
        return None
    logs = list((rd / "rollout_logs").glob(f"{task}-cand*.log"))
    launched = len(logs)
    if launched == 0:
        return None
    proposed = len(candidates)
    h1_model_calls = sum(int(row.get("llm_calls") or 0) for row in candidates)
    reviewer_model_calls = sum(
        int(review.get("rounds") or 0)
        for candidate in candidates
        for review in candidate.get("review_log") or []
    )
    max_evals = int(meta.get("max_evals") or 0)
    execution_ledger = trajectory_execution_ledger(rd / "rollouts" / task)
    return ([], launched, proposed, h1_model_calls, reviewer_model_calls,
            max_evals,
            execution_ledger, str(meta_path))


def trajectory_execution_ledger(root: Path) -> dict[str, float | int]:
    """Sum the ledgers recorded by one completed summary per trajectory.

    A launched trajectory can die before writing a summary.  The returned
    counts are therefore explicitly a recorded lower bound; the separately
    stored launched*max_evals value remains the conservative charged cap.
    """
    summaries: list[Path] = []
    if root.exists():
        # Outer rounds have candXX/<timestamp>/summary.json; executor batches
        # call this helper once per kXX directory and have <timestamp>/summary.
        children = [path for path in root.iterdir() if path.is_dir()]
        direct = list(root.glob("*/summary.json"))
        if direct:
            summaries.append(max(direct, key=lambda path: path.stat().st_mtime))
        else:
            for child in children:
                candidates = list(child.glob("*/summary.json"))
                if candidates:
                    summaries.append(max(
                        candidates, key=lambda path: path.stat().st_mtime))
    evaluator_calls = 0
    executor_model_calls = 0
    sandbox_seconds = 0.0
    usable = 0
    for path in summaries:
        try:
            payload = json.loads(path.read_text())
            payload = payload[0] if isinstance(payload, list) else payload
            ledger = payload.get("ledger") or {}
            evaluator_calls += int(
                ledger.get("evaluator_calls", payload.get("evaluations", 0)) or 0)
            executor_model_calls += int(ledger.get("llm_calls") or 0)
            sandbox_seconds += float(ledger.get("sandbox_seconds") or 0.0)
            usable += int(payload.get("best_score") is not None)
        except Exception:
            continue
    return {
        "recorded_trajectory_summaries": len(summaries),
        "recorded_usable_trajectories": usable,
        "recorded_evaluator_calls": evaluator_calls,
        "recorded_executor_model_calls": executor_model_calls,
        "recorded_sandbox_seconds": sandbox_seconds,
    }


def proposer_curve_from_rounds(
    task: str,
    rounds: dict[int, dict[str, str]],
    outer_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    seed = ANCHORS[task][0]
    points = [h2_warm_start_point(task, "round")]
    sources, cumulative, best = [], 1, seed
    ordered_rounds = sorted(
        rounds.items(),
        key=lambda item: (
            item[1].get("submitted_at") or "9999",
            item[0],
        ),
    )
    last_logged_round = ordered_rounds[-1][0] if ordered_rounds else -1
    for rid, meta in ordered_rounds:
        got = round_group(rid, task, outer_root=outer_root)
        status = "completed"
        if got is None:
            # A driver logs the round and job before rollouts start.  The last
            # logged round can therefore be live even though ``round.json`` and
            # rollout logs already exist.  Only a no-summary round followed by
            # a later logged round is closed evidence of a failed attempt; the
            # live tail is excluded until its atomic ``round_summary.json`` is
            # written.  Explicit pre-retry failures are accounted through the
            # campaign run manifest rather than guessed from a mutable folder.
            if rid == last_logged_round:
                continue
            got = failed_round_attempt(rid, task, outer_root)
            status = "failed_without_round_summary"
        if got is None:
            continue
        (scores, launched, proposed, h1_model_calls, reviewer_model_calls,
         max_evals,
         execution_ledger, source) = got
        cumulative += launched
        if scores:
            best = max(best, max(scores))
        points.append({"x": cumulative, "score": best, "round": rid,
                       "phi": meta["phi"], "source": source,
                       "submitted_at": meta.get("submitted_at"),
                       "launched": launched, "proposed": proposed,
                       "h1_model_calls": h1_model_calls,
                       "reviewer_model_calls_lower_bound": reviewer_model_calls,
                       "max_evals_per_trajectory": max_evals,
                       "charged_evaluator_call_budget": launched * max_evals,
                       **execution_ledger,
                       "job": meta.get("job"), "status": status})
        sources.extend([meta["driver"], source])
    return collapse_x(points), sorted(set(sources))


def proposer_curve(task: str) -> tuple[list[dict[str, Any]], list[str]]:
    if task in ("eft__ahc_simpletes__ahc039", "adrs__eplb"):
        workspace, outer_root = SOTA_CLEAN_PROPOSER_ROUTES[task]
        driver = workspace / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, outer_root)
            if len(points) > 1:
                return points, sources
        if os.environ.get("REQUIRE_CLEAN_CORE_PROPOSER") == "1":
            raise RuntimeError(
                f"{task}: cadence-matched isolated K=8/max-evals=20 proposer "
                "replacement is not available"
            )
    if task == "eft__math__hadamard_maximal_det":
        driver = SOTA_HADAMARD_PROPOSER_WORKSPACE / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, SOTA_HADAMARD_PROPOSER_OUTER_ROOT)
            if len(points) > 1:
                return points, sources
        if os.environ.get("ALLOW_CONTAMINATED_REWARD_FALLBACK") != "1":
            raise RuntimeError(
                "clean isolated Hadamard proposer lineage is not available; "
                "historical rounds credited terminal harness failures with the "
                "seed checkpoint before GRPO training"
            )
    if task == "eft__ahc_simpletes__ahc058":
        driver = SOTA_AHC058_PROPOSER_WORKSPACE / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, SOTA_AHC058_PROPOSER_OUTER_ROOT)
            if len(points) > 1:
                return points, sources
        if os.environ.get("ALLOW_CONTAMINATED_REWARD_FALLBACK") != "1":
            raise RuntimeError(
                "clean isolated AHC058 proposer lineage is not available; "
                "historical round511 credited a terminal harness failure with "
                "the seed checkpoint before GRPO training"
            )
    if task == "adrs__prism":
        driver = SOTA_PRISM_PROPOSER_WORKSPACE / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, SOTA_PRISM_PROPOSER_OUTER_ROOT)
            if len(points) > 1:
                return points, sources
        if os.environ.get("ALLOW_CONTAMINATED_PRISM_PROPOSER") != "1":
            raise RuntimeError(
                "clean isolated PRISM proposer lineage is not available; refusing "
                "round410--417 because round410/cand04 had success_rate=0.98, "
                "was selected as best_k, and its harness became the later base"
            )
    if task == "adrs__llm_sql":
        driver = SOTA_SQL_PROPOSER_WORKSPACE / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, SOTA_SQL_PROPOSER_OUTER_ROOT)
            if len(points) > 1:
                return points, sources
        if os.environ.get("ALLOW_CONTAMINATED_SQL_PROPOSER") != "1":
            raise RuntimeError(
                "clean isolated SQL proposer lineage is not available; refusing "
                "to fall back to leaked historical round471--475"
            )
    if task == "adrs__txn_scheduling":
        driver = SOTA_TXN_PROPOSER_WORKSPACE / "driver.log"
        clean_rounds = logged_phi_rounds([str(driver)]) if driver.exists() else {}
        if clean_rounds:
            points, sources = proposer_curve_from_rounds(
                task, clean_rounds, SOTA_TXN_PROPOSER_OUTER_ROOT)
            if len(points) > 1:
                return points, sources
        if os.environ.get("ALLOW_CONTAMINATED_TXN_PROPOSER") != "1":
            raise RuntimeError(
                "clean isolated Txn proposer lineage is not available; refusing "
                "round460--463 because its serialized seed was the illegal "
                "one-element round450 program"
            )
        # Diagnostics-only fallback: this lineage produced legal endpoints but
        # inherited the invalid round450 program in its H1 prompt, so it is not
        # admissible efficiency or final-report evidence.
        legacy_driver = ROOT / "fresh_all" / task / "driver.log"
        rounds = logged_phi_rounds(
            [str(legacy_driver)]) if legacy_driver.exists() else {}
        return proposer_curve_from_rounds(task, rounds, ROOT / "outer")
    if task == "eft__math__hadamard_maximal_det":
        # Diagnostics-only fallback.  The historical endpoint was reached by
        # the subsequent adaptive_ab
        # continuation and then stress-tested by sota_push.  Charge both
        # continuations (including their plateau rounds) so the blue curve
        # reaches the provenance-backed 0.573283 endpoint without pretending
        # that it came from the shorter fresh_all segment alone.
        paths = (
            glob.glob(str(ROOT / "fresh_all" / task / "driver*.log"))
            + glob.glob(str(ROOT / "adaptive_ab" / task / "driver*.log"))
            + glob.glob(str(ROOT / "sota_push" / task / "driver*.log"))
        )
        rounds = logged_phi_rounds(paths)
        return proposer_curve_from_rounds(task, rounds, ROOT / "outer")
    return proposer_curve_from_rounds(task, phi_rounds(), ROOT / "outer")


def matched_proposer_curve(task: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Independent local-ratchet arm whose proposer is updated after each round."""
    workspace = "arm_proposer"
    seed = ANCHORS[task][0]
    points = [h2_warm_start_point(task, "round")]
    sources = [str(ROOT / workspace / "driver.log")]
    cumulative, best = 1, seed
    for rid, job in context_round_records(workspace):
        got = round_group(rid, task)
        if got is None:
            continue
        (scores, launched, proposed, h1_model_calls, reviewer_model_calls,
         max_evals,
         execution_ledger, source) = got
        cumulative += launched
        if scores:
            best = max(best, max(scores))
        points.append({"x": cumulative, "score": best, "round": rid,
                       "source": source, "launched": launched,
                       "proposed": proposed,
                       "h1_model_calls": h1_model_calls,
                       "reviewer_model_calls_lower_bound": reviewer_model_calls,
                       "max_evals_per_trajectory": max_evals,
                       "charged_evaluator_call_budget": launched * max_evals,
                       **execution_ledger,
                       "job": job})
        sources.append(source)
    return collapse_x(points), sorted(set(sources))


def context_round_records(workspace: str) -> list[tuple[int, str | None]]:
    log = ROOT / workspace / "driver.log"
    if not log.exists():
        return []
    text = log.read_text(errors="ignore")
    rounds = [int(x) for x in re.findall(r"round(\d+) over", text)]
    jobs = re.findall(r"\[ctx\]\s+job (\d+)", text)
    return [(rid, jobs[i] if i < len(jobs) else None)
            for i, rid in enumerate(rounds)]


def analyst_brief_count(job: str | None, task: str) -> tuple[int, str | None]:
    if not job:
        return 0, None
    path = Path("/lustre/fsw/portfolios/av/users/yingzim/logs/slurm") / f"sah-outer-{job}.out"
    if not path.exists():
        return 0, str(path)
    marker = f"[propose] {task}: analysis brief attached"
    return path.read_text(errors="ignore").count(marker), str(path)


def context_outer_root(workspace: str) -> tuple[Path, str | None]:
    """Resolve an isolated context campaign's round root from its ledger."""
    manifest = ROOT / workspace / "run_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
            return Path(payload["outer_root"]), str(manifest)
        except Exception:
            pass
    return ROOT / "outer", None


def context_curve(
    task: str, preferred_workspace: str | None = None
) -> tuple[list[dict[str, Any]], list[str], str]:
    candidates = []
    workspaces = ((preferred_workspace,) if preferred_workspace else
                  ("context_v2", "arm_context_long"))
    # Prefer the newer six-task matched arm on a tie when no source is pinned.
    for priority, ws in enumerate(workspaces):
        outer_root, run_manifest = context_outer_root(ws)
        completed = [(r, j) for r, j in context_round_records(ws)
                     if round_group(r, task, outer_root) is not None]
        candidates.append((len(completed), priority, ws, completed,
                           outer_root, run_manifest))
    _, _, workspace, rounds, outer_root, run_manifest = max(candidates)
    seed = ANCHORS[task][0]
    points = [h2_warm_start_point(task, "round")]
    sources = [str(ROOT / workspace / "driver.log")]
    if run_manifest:
        sources.append(run_manifest)
    cumulative, best = 1, seed
    for index, (rid, job) in enumerate(rounds):
        got = round_group(rid, task, outer_root)
        if got is None:
            continue
        briefs, slurm_log = analyst_brief_count(job, task)
        # The cold first round has no prior feedback by construction.  Every
        # later round must prove that the analyzer actually attached context.
        if index > 0 and briefs == 0:
            raise RuntimeError(f"{workspace} round{rid}: no analyzer brief in {slurm_log}")
        (scores, launched, proposed, h1_model_calls, reviewer_model_calls,
         max_evals,
         execution_ledger, source) = got
        cumulative += launched
        if scores:
            best = max(best, max(scores))
        points.append({"x": cumulative, "score": best, "round": rid,
                       "source": source, "launched": launched,
                       "proposed": proposed,
                       "h1_model_calls": h1_model_calls,
                       "reviewer_model_calls_lower_bound": reviewer_model_calls,
                       "max_evals_per_trajectory": max_evals,
                       "charged_evaluator_call_budget": launched * max_evals,
                       **execution_ledger,
                       "analyst_briefs": briefs,
                       "analyzer_model_calls": 2 * briefs,
                       "analyzer_specialists": ["performance", "design"] if briefs else [],
                       "job": job})
        sources.append(source)
        if slurm_log:
            sources.append(slurm_log)
    return collapse_x(points), sorted(set(sources)), workspace


def executor_curve(
    task: str, tag: str, state_root: Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    state_dir = (state_root or TTT_STATE_ROOT) / tag
    curve = state_dir / "curve.jsonl"
    state_file = state_dir / "state.json"
    seed = ANCHORS[task][0]
    points = [h2_warm_start_point(task, "step")]
    sources = [str(curve), str(state_file)]
    batch_meta: dict[int, dict[str, Any]] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            batch_meta = {int(row["step"]): row for row in state.get("batches", [])}
        except Exception:
            batch_meta = {}
    if curve.exists():
        running = seed
        for line in curve.read_text().splitlines():
            try:
                row = json.loads(line)
                score = float(row["best"])
                x = int(row["cum_rollouts"])
            except Exception:
                continue
            step = row.get("step")
            meta = batch_meta.get(int(step)) if step is not None else None
            usable = int((meta.get("usable") if meta else row.get("usable")) or 0)
            train_rows = int((meta.get("train_rows") if meta else usable) or 0)
            # A partial tail batch is cost evidence, but it is not a canonical
            # K-sized reward/update batch.  Keep it in its immutable eval
            # manifest and GPU ledger while excluding it from the primary
            # curve until explicit top-ups close both the usable and replay-row
            # targets.  The strict audit independently enforces this invariant.
            if usable < TTT_BATCH_K or train_rows < TTT_BATCH_K:
                continue
            running = max(running, score)
            batch_dir = Path(meta["round_dir"]) if meta and meta.get("round_dir") else None
            batch_ledger = {
                "recorded_trajectory_summaries": 0,
                "recorded_usable_trajectories": 0,
                "recorded_evaluator_calls": int(
                    meta.get("evaluator_calls") or 0) if meta else 0,
                "recorded_executor_model_calls": 0,
                "recorded_sandbox_seconds": 0.0,
            }
            if batch_dir and batch_dir.exists():
                rows = [path for path in batch_dir.glob("k*") if path.is_dir()]
                parts = [trajectory_execution_ledger(path) for path in rows]
                batch_ledger = {
                    key: sum(float(part[key]) for part in parts)
                    for key in batch_ledger
                }
                for key in ("recorded_trajectory_summaries",
                            "recorded_usable_trajectories",
                            "recorded_evaluator_calls",
                            "recorded_executor_model_calls"):
                    batch_ledger[key] = int(batch_ledger[key])
            # ``curve.jsonl`` stores only newly launched trajectories.  Add the
            # one shared H2 warm-start trajectory shown at x=1 so the first
            # fresh K=8 batch lands at x=9 for every arm.
            points.append({"x": x + 1, "score": running,
                           "step": step, "source": str(curve),
                           "checkpoint": row.get("checkpoint"),
                           "usable": usable,
                           "train_rows": train_rows,
                           "launched": meta.get("launched") if meta else row.get("launched"),
                           "batch_best": meta.get("batch_best") if meta else row.get("batch_best"),
                           "evaluator_calls": meta.get("evaluator_calls") if meta else None,
                           "max_evals_per_trajectory": 20,
                           "charged_evaluator_call_budget": 20 * int(
                               (meta.get("launched") if meta else row.get("launched")) or 0
                           ),
                           **batch_ledger})
    sources.extend(str(p) for p in sorted(state_dir.glob("eval_*/eval_manifest.json")))
    return collapse_x(points), sorted(set(sources))


def collapse_x(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one best-so-far point per x without hiding charged restarts."""
    out: dict[int, dict[str, Any]] = {}
    for point in points:
        x = int(point["x"])
        if x not in out or float(point["score"]) >= float(out[x]["score"]):
            out[x] = point
    return [out[x] for x in sorted(out)]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            text=True).strip()
    except Exception:
        return "unknown"


def protocol_guard_proof() -> dict[str, Any]:
    path = ROOT / "protocol_guards.ok"
    worker = REPO / "src/inner/evaluation/_eval_worker.py"
    current_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    fields: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
    return {
        "path": str(path),
        "fields": fields,
        "current_eval_worker_sha256": current_sha,
        "matches_current_worker": fields.get("eval_worker_sha256") == current_sha,
    }


def legacy_guard_proof() -> dict[str, Any]:
    payload, _ = legacy_semantic_guard_audit()
    worker = REPO / "src/inner/evaluation/_eval_worker.py"
    current_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    return {
        "path": str(LEGACY_GUARD_AUDIT),
        "job": payload.get("job"),
        "counts": payload.get("counts") or {},
        "scope": payload.get("scope") or {},
        "warm_start_valid": payload.get("warm_start_valid") or {},
        "warm_start_current_guard_scores": payload.get(
            "warm_start_current_guard_scores"
        ) or {},
        "eval_worker_sha256": payload.get("eval_worker_sha256"),
        "current_eval_worker_sha256": current_sha,
        "matches_current_worker": payload.get("eval_worker_sha256") == current_sha,
    }


def endpoint_revalidation(task: str, method: str) -> dict[str, Any] | None:
    global _ENDPOINT_VALIDATION_CACHE
    if _ENDPOINT_VALIDATION_CACHE is None:
        try:
            payload = json.loads(SOTA_ENDPOINT_VALIDATION.read_text())
            _ENDPOINT_VALIDATION_CACHE = (
                payload if payload.get("status") == "complete" else {}
            )
        except Exception:
            _ENDPOINT_VALIDATION_CACHE = {}
    row = (_ENDPOINT_VALIDATION_CACHE.get("case_results") or {}).get(
        f"{task}::{method}"
    )
    return row if isinstance(row, dict) else None


def endpoint_revalidation_proof() -> dict[str, Any]:
    endpoint_revalidation("", "")
    payload = _ENDPOINT_VALIDATION_CACHE or {}
    return {
        "path": str(SOTA_ENDPOINT_VALIDATION),
        "status": payload.get("status"),
        "requested_runs": payload.get("requested_runs"),
        "all_runs_valid": payload.get("all_runs_valid"),
        "case_count": len(payload.get("case_results") or {}),
        "visual_encoding": (
            "black tick and horizontal whisker in endpoint inset = repeated "
            "evaluator mean +/- one standard deviation; colored bar = online "
            "observed locally ledgered proposer endpoint"
        ),
    }


def main() -> None:
    global TTT_STATE_ROOT, TTT_AHC039_STATE_ROOT, TTT_EPLB_STATE_ROOT
    global TTT_EXTRA_STATE_ROOT
    global TTT_BATCH_K, TTT_MODEL_TAG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default="papers/figures/score_compute_curves_12h")
    ap.add_argument(
        "--layout", choices=("2x3", "1x4", "1x5", "1x7"), default="2x3",
        help=("use the six-task audit, four selected main-text illustrations, "
              "five-task pilot, or seven priority tasks"),
    )
    args = ap.parse_args()
    sota_layout = args.layout in ("1x4", "1x5", "1x7")
    # The legacy six-task figure used a K~=32 sensitivity campaign.  The paper
    # SOTA panels use distinct cadence-matched K=8 campaigns.  Make these modes
    # mode fail-safe when invoked manually: explicit environment overrides are
    # still honored, but omission can no longer silently select the excluded
    # sensitivity run.
    if sota_layout:
        sota5_root = ROOT / "ttt_discover_sota5_k8"
        if not _TTT_STATE_ROOT_EXPLICIT:
            TTT_STATE_ROOT = sota5_root
        if not _TTT_AHC039_STATE_ROOT_EXPLICIT:
            TTT_AHC039_STATE_ROOT = ROOT / "ttt_discover_sota5_k8_ahc039"
        if not _TTT_EPLB_STATE_ROOT_EXPLICIT:
            TTT_EPLB_STATE_ROOT = TTT_STATE_ROOT
        if not _TTT_EXTRA_STATE_ROOT_EXPLICIT:
            TTT_EXTRA_STATE_ROOT = ROOT / "ttt_discover_sota7_extra_k8"
        if not _TTT_BATCH_K_EXPLICIT:
            TTT_BATCH_K = 8
        if not _TTT_MODEL_TAG_EXPLICIT:
            TTT_MODEL_TAG = "ttts5k8"
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.layout == "1x4":
        plot_tasks = SOTA4_TASKS
        fig, axes = plt.subplots(1, 4, figsize=(18.0, 8.2))
        axes_flat = list(axes)
        fig.subplots_adjust(left=0.052, right=0.992, top=0.845, bottom=0.405,
                            wspace=0.15)
    elif args.layout == "1x5":
        plot_tasks = SOTA5_TASKS
        fig, axes = plt.subplots(1, 5, figsize=(24.5, 7.4))
        axes_flat = list(axes)
        fig.subplots_adjust(left=0.044, right=0.994, top=0.84, bottom=0.405,
                            wspace=0.12)
    elif args.layout == "1x7":
        plot_tasks = SOTA_TASKS
        # Keep the seven requested tasks in one row, but use a landscape-page
        # aspect ratio.  The earlier 33.5x7.6 canvas became only ~1.6 inches
        # tall when included at paper width, making the independent y-scales,
        # endpoint zooms, and fairness ledger unreadable.  This canvas is meant
        # to be included as a sideways full-page figure.
        fig, axes = plt.subplots(1, 7, figsize=(20.0, 9.2))
        axes_flat = list(axes)
        fig.subplots_adjust(left=0.046, right=0.995, top=0.855, bottom=0.41,
                            wspace=0.16)
    else:
        plot_tasks = TASKS
        fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.4))
        axes_flat = list(axes.ravel())
        fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.205,
                            hspace=0.27, wspace=0.08)

    # All selected metrics are higher-is-better.  A direct ratio keeps y=1
    # aligned with the frozen human reference even when fixed H2 exceeds that
    # reference; gap-closed normalization would reverse direction there.
    if sota_layout:
        plot_anchors = {
            task: (seed, HUMAN_BEST.get(task, reference))
            for task, (seed, reference) in ANCHORS.items()
        }
        norm_score = lambda task, score: float(score) / plot_anchors[task][1]
        y_definition = "combined_score divided by task's Best Human value"
        y_label = "best validated score / human best"
    else:
        plot_anchors = ANCHORS
        norm_score = normalize
        y_definition = "best valid combined_score normalized by seed and published <=10B reference"
        y_label = "best validated normalized score"
    fig.suptitle(
        "Where should test-time reward go? Proposer weights vs. analyzer context vs. executor weights",
        fontsize=21, y=0.975,
    )
    h2_provenance = json.loads(H2_WARM_START_PROVENANCE.read_text())
    manifest: dict[str, Any] = {
        "schema": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "layout": args.layout,
        "view_selection": (
            {
                "role": "main-text task-level illustration",
                "tasks": list(SOTA4_TASK_IDS),
                "selection_timing": "chosen after interim inspection of the full seven-task campaign",
                "selection_rationale": (
                    "the interim proposer curve exceeds the executor reference "
                    "at each selected task's route-common trajectory budget"
                ),
                "claim_restriction": (
                    "no four-task population-average or confirmatory aggregate claim; "
                    "the complete predeclared 1x7 view and audit remain authoritative"
                ),
                "full_view": str(out_prefix.with_name(
                    out_prefix.name.replace("sota4", "sota7") + "_data.json"
                )),
            }
            if args.layout == "1x4" else
            {
                "role": "clean-fair five-task reward-routing view",
                "tasks": list(SOTA5_TASK_IDS),
                "selection_timing": (
                    "task subset fixed by the requested comparison scope before "
                    "the final clean-fair render"
                ),
                "claim_restriction": (
                    "exact subview of the full 1x7 clean-fair manifest; every reported "
                    "condition must be generated from these audited route endpoints"
                ),
                "full_view": str(out_prefix.with_name(
                    out_prefix.name.replace("clean5", "sota7").replace(
                        "sota5", "sota7"
                    ) + "_data.json"
                )),
            }
            if args.layout == "1x5" else
            {
                "role": "full predeclared priority-task comparison",
                "tasks": [task for task, _, _ in SOTA_TASKS],
            }
            if args.layout == "1x7" else None
        ),
        "x": (
            "one shared fixed-H2 warm-start trajectory plus actual newly "
            "launched executor trajectories"
        ),
        "y": y_definition,
        "anchors": plot_anchors,
        "reference_standard": {
            "name": "Best Human" if sota_layout else "published <=10B best",
            "source": (
                str(HUMAN_BEST_REFERENCE)
                if sota_layout else "task-specific published references"
            ),
            "source_sha256": (
                hashlib.sha256(HUMAN_BEST_REFERENCE.read_bytes()).hexdigest()
                if sota_layout else None
            ),
            "y_equals_one": True,
            "normalization": (
                "combined_score / human_best_combined_score"
                if sota_layout else
                "gap closed from seed to published <=10B reference"
            ),
        },
        "shared_h2_warm_start_provenance": {
            "path": str(H2_WARM_START_PROVENANCE),
            "semantics": h2_provenance.get("semantics"),
            "tasks": h2_provenance.get("tasks") or {},
        },
        "protocol_guard_proof": protocol_guard_proof(),
        "legacy_semantic_guard_proof": legacy_guard_proof(),
        "endpoint_revalidation_proof": endpoint_revalidation_proof(),
        "protocol_validity_guards": {
            "eplb": (
                "output-level replica-count and logical-group-to-node topology; "
                "rejects global packing that ignores num_groups/num_nodes"
            ),
            "prism": (
                "success_rate must equal 1.0 across all generated placement cases; "
                "partial-success cherry-picking is invalid"
            ),
            "txn": (
                "exactly three schedules; each must be an exact permutation of "
                "transaction IDs 0..99; score is recomputed after this check"
            ),
            "application": (
                "online in _eval_worker for new context/executor arms and "
                "hash-indexed post-hoc replay through the same checks for "
                "legacy proposer summaries"
            ),
        },
        "proposer_route": {
            "name": "update proposer weights",
            "ownership": "ours",
            "model": "Qwen3.5-9B",
            "executor_model": "Qwen3.5-9B frozen base",
            "proposal_serving": (
                "four GPUs reserved; H1 uses one trained-phi replica while "
                "the other replicas retain the frozen base for review/analysis "
                "and subsequent executor rollouts"
            ),
            "training": {
                "lora_rank": 64,
                "lora_alpha": 128,
                "epochs_per_update": 3,
                "learning_rate": 3e-5,
                "kl_coefficient": 0.05,
                "global_batch_size": 8,
                "adapter_state_between_updates": "continued",
                "optimizer_state_between_updates": "reinitialized",
                "scheduler_state_between_updates": "reinitialized",
            },
        },
        "context_route": {
            "name": "update analyzer/context with both weight sets frozen",
            "ownership": "ours ablation",
            "proposer_model": "Qwen3.5-9B frozen base",
            "executor_model": "Qwen3.5-9B frozen base",
            "proposal_serving": (
                "four GPUs reserved; because every replica has the same frozen "
                "weights, H1 can use up to four replicas in parallel"
            ),
            "analyzer": "two frozen-base specialist calls after the cold round",
            "training": None,
        },
        "executor_baseline": {
            "name": "budget-scaled TTT-Discover-style executor adaptation",
            "ownership": "reference baseline; not our method",
            "model": "Qwen3.5-9B",
            "fixed_harness": True,
            "target_usable_trajectories_per_update": TTT_BATCH_K,
            "x_accounting": (
                "actual launched trajectories, including failed/straggling launches; "
                "AHC039 supplemental launches are counted from per-trajectory logs"
            ),
            "training": {"lora_rank": 32, "lora_alpha": 64,
                         "learning_rate": {
                             "ahc058": 2e-5,
                             "all_other_local_tasks": 4e-5,
                         },
                         "kl_coefficient": {
                             "algorithm_engineering_ahc": 0.01,
                             "other_tasks": 0.1,
                         },
                         "global_batch_size": 4,
                         "optimizer_steps_per_update": max(1, TTT_BATCH_K // 4),
                         "adam_beta2": 0.98,
                         "weight_decay": 0.1,
                         "optimizer_state_between_updates": "reinitialized",
                         "scheduler_note": (
                             "the local FSDP cosine scheduler initializes at lr=0; "
                             "GBS=4 gives a K=8 replay two optimizer boundaries so "
                             "the first updates Adam moments but not weights and the "
                             "second makes a non-zero weight update, without replaying data"
                         ),
                         "temperature": 1.0},
            "matched_components": ["adaptive-beta entropic LOO advantage",
                                   "PUCT state reuse", "top-2 children per parent"],
            "budget_deviation": (
                f"{TTT_BATCH_K} trajectories in one group per update; "
                "official default is 8 groups x 64"
            ),
            "implementation_deviations": [
                "local FSDP replay trains a teacher-forced final edit/program rather than the official Tinker stack's complete sampled token trajectory",
                "local Adam uses beta2=0.98 and weight decay=0.1; official Adam uses beta2=0.95 and does not report weight decay",
                "local jobs preserve LoRA weights but restart Adam state at each update; the official training client is persistent",
                "local K=8 uses two GBS=4 optimizer boundaries as a scheduler workaround; official TTT-Discover takes one optimizer step on the full 512-rollout batch",
            ],
            "state_root": str(TTT_STATE_ROOT),
            "ahc039_state_root": str(TTT_AHC039_STATE_ROOT),
            "eplb_state_root": str(TTT_EPLB_STATE_ROOT),
            "extra_state_root": str(TTT_EXTRA_STATE_ROOT),
            "model_tag": TTT_MODEL_TAG,
            "official_sources": {
                "paper": "https://arxiv.org/pdf/2601.16175",
                "repository": "https://github.com/test-time-training/discover",
                "repository_commit_audited": "6c40e82dab9d5de7416ac873ad5cd3106084aaed",
                "official_batch": "8 groups x 64 rollouts x 50 steps = 25,600",
                "official_optimizer": "Adam lr=4e-5 (AHC058: 2e-5), beta1=0.9, beta2=0.95, eps=1e-8; one full-batch step",
                "official_compute_accounting": (
                    "the paper labels TTT-Discover, Best-of-N, and OpenEvolve "
                    "sampling-budget matched at 25,600 rollouts; that does not "
                    "establish equal total training compute, which this audit "
                    "therefore reports separately"
                ),
                "advantages": "https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/ttt_discover/rl/train.py",
                "puct_sampler": "https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/ttt_discover/tinker_utils/sampler.py",
            },
        },
        "comparison_policy": {
            "common_start": {
                "warm_start_measurement": (
                    "one fixed-H2/base-executor trajectory with 20 evaluator "
                    "calls, displayed at x=1 for every arm"
                ),
                "actual_program_start": (
                    "task.initial_program for the first fresh batch of every arm; "
                    "later batches follow only that arm's task-local ratchet"
                ),
                "h2_program_inherited": False,
                "executor_puct_root": (
                    "task.initial_program with its true seed score; the H2 "
                    "warm-start score is not assigned to that program"
                ),
                "nominal_first_fresh_batch_target": TTT_BATCH_K,
                "executor_first_fresh_batch_x": 1 + TTT_BATCH_K,
                "proposer_context_first_fresh_batch_x": (
                    "1 + actual materialized executor launches; invalid H1 "
                    "proposals are charged separately as proposer calls"
                ),
                "cost_treatment": (
                    "the identical warm-start is reported separately; logical "
                    "per-arm adaptation costs exclude it"
                ),
            },
            "efficiency_series": (
                "task-local proposer-weight campaigns vs. isolated context and "
                "executor arms; the strict five-task artifact requires isolated "
                "K=8/max-evals=20 proposer replacements for AHC039 and EPLB as "
                "well as the clean AHC058/PRISM/LLM-SQL lineages. An "
                "allow-incomplete live render may retain the superseded historical "
                "AHC039/EPLB curves only until their replacements materialize"
                if sota_layout else
                "matched independent proposer arm vs. independent context and executor arms"
            ),
            "task_selection_scope": (
                "these four panels were selected after interim inspection from "
                "the predeclared seven-task comparison as task-level main-text "
                "illustrations; all seven tasks and all confirmatory aggregates "
                "remain in the appendix audit, so this view is not an unbiased "
                "benchmark sample and has no standalone aggregate claim"
                if args.layout == "1x4" else
                "these five panels are the requested comparison task subset and "
                "are an exact subview of the predeclared seven-task clean-fair "
                "manifest; all reported condition values must come from the same "
                "audited endpoints"
                if args.layout == "1x5" else
                "the seven panels were preselected as priority/strength "
                "priority/strength tasks, not sampled as an unbiased benchmark "
                "population; aggregate results establish effectiveness in this "
                "declared regime and are not a population-average estimate"
            ),
            "proposer_context_feedback_matching": (
                "both H1 routes receive the task-local incumbent and the same bounded "
                "per-candidate scored-feedback format; context alone adds two frozen-"
                "model analyst calls, while proposer alone updates proposer weights"
            ),
            "x_scope": (
                "task-local launched executor trajectories only; proposer generation, "
                "analyzer inference, LoRA training, and checkpoint merge are not converted "
                "into rollout equivalents"
            ),
            "total_compute_status": (
                "x supports executor-sample-efficiency comparisons, not equal-total-GPU-"
                "compute claims; job-level GPU time is audited separately"
            ),
            "allocation_utilization_caveat": (
                "both H1 routes reserve four GPUs, but proposer-weight H1 uses "
                "one trained-phi replica whereas frozen context H1 can use up to "
                "four replicas; sacct charges every reserved GPU, including idle "
                "capacity, so logical-call and allocation-time comparisons are separate"
            ),
            "context_co_batching_caveat": (
                "AHC/system-context jobs also generated older AHC058/PRISM "
                "branches that are excluded. PRISM is replaced by rewardfix-v1; "
                "AHC058 is replaced by its analysis-required isolated recovery. "
                "task-local logical ledgers exclude them, but suite sacct GPU-hours "
                "charge the full shared jobs"
            ),
            "ahc058_context_analyzer_required_policy": {
                "excluded_workspace": "context_sota7_rewardfix_v1",
                "first_invalid_round": 1101,
                "excluded_observed_score": 1.2732476555555556,
                "reason": (
                    "the AHC058 round improved without the immutable Slurm "
                    "marker proving that an analyzer brief was attached; all "
                    "task-local descendants inherit that non-analyzer harness"
                ),
                "replacement_workspace": (
                    "context_sota7_ahc058_analysis_required_v1"
                ),
                "retry_treatment": (
                    "restore pre-round state and exclude score/x evidence, but "
                    "charge the complete rejected allocation in as-run cost"
                ),
            },
            "accepted_post_work_job_anomalies": {
                "registry": "results/sota7_accepted_job_anomalies.json",
                "scope": "five proposer/context outer jobs",
                "acceptance_rule": (
                    "all expected launch logs and terminal summaries plus "
                    "round_summary.json and next_bases.json must exist before a "
                    "post-launch shell failure can be accepted"
                ),
                "cost_treatment": (
                    "retain terminal-score evidence and charge the complete "
                    "allocation in accepted protocol GPU hours"
                ),
            },
            "visual_scaling": (
                "each panel preserves the same score/human-best ratio but uses an "
                "independently zoomed y-range; gaps are comparable within a task, "
                "not by apparent vertical pixel height across tasks"
            ),
            "causal_scope": (
                "this is a comparison of complete reward-routing systems, not a pure "
                "optimizer-target intervention: after the shared fixed-H2 anchor, "
                "proposer/context synthesize candidate harnesses through H1 while the "
                "TTT-style executor reference keeps the initial harness fixed"
            ),
            "proposer_campaign_timing": (
                "the final clean-five protocol uses isolated K=8/max-evals=20 "
                "proposer campaigns for AHC039, AHC058, EPLB, PRISM, and LLM-SQL. "
                "The earlier multi-restart AHC039 and non-plateau EPLB curves are "
                "retained only as sensitivity/as-run evidence. The new proposer, "
                "context, and executor campaigns remain unpaired in sampling seed"
            ),
            "sampling_uncertainty": (
                "historical proposer and newly run context/executor campaigns use "
                "unpaired sampling seeds; there is one campaign per arm and no "
                "confidence interval"
            ),
            "endpoint_revalidation_cost_treatment": (
                "N>=5 endpoint re-evaluation is evaluator-only, CPU-only common "
                "measurement overhead performed after adaptation; it is reported "
                "as validation and excluded from every route's adaptation GPU cost"
            ),
            "proposer_restart_accounting": (
                "the superseded historical AHC039 curve aggregates several "
                "task-local restarts and remains charged in historical as-run "
                "accounting; the strict final curve instead requires the isolated "
                "single-lineage proposer_sota5_ahc039_clean_v1 campaign"
            ),
            "model_call_accounting": (
                "trajectory ledgers recover request counts but local vLLM does "
                "not report complete per-request token totals; request-budget "
                "results are a sensitivity analysis, not FLOP equivalence"
            ),
            "reporting_alignment": (
                "shared fixed-H2 warm-start scores and blue inset endpoints use "
                "the canonical run provenance; H2 is a measured harness baseline, "
                "not an inherited task program. The earlier separate-team AHC039 "
                "559,534 result is excluded because it has no local program/x "
                "ledger; the report and curve use the locally ledgered endpoint"
            ),
            "sql_proposer_leak_policy": (
                "historical round471--475 is excluded because its serialized H1 "
                "prompt contained a curated note naming a 0.728 program and asking "
                "for verbatim adoption; the plotted/reported SQL endpoint must "
                f"come from {SOTA_SQL_PROPOSER_OUTER_ROOT.name}, whose prompts "
                "and task-local incumbent chain are audited fail-closed"
            ),
            "txn_proposer_contamination_policy": (
                "historical round460--463 is excluded even though its endpoint "
                "programs were legal: the serialized round460 H1 seed was the "
                "illegal one-element program selected by the old round450 "
                "evaluator exploit. The plotted endpoint must come from the "
                "isolated current-guard campaign starting at task.initial_program"
            ),
            "prism_proposer_contamination_policy": (
                "historical round410--417 is excluded: round410/cand04 had "
                "success_rate=0.98 but was selected as best_k=4, and its H1 "
                "harness package became the base of later proposer updates. "
                "The plotted/reported endpoint must come from the isolated "
                "current-guard campaign starting at task.initial_program"
            ),
            "terminal_summary_reward_attribution_policy": (
                "a terminal task row is authoritative: best_score=null is a "
                "failed trajectory and may not inherit the earlier seed "
                "checkpoint. Historical Hadamard/AHC058 proposer lineages and "
                "AHC058/PRISM context lineages that violated this rule are "
                "excluded and rerun from task.initial_program in isolated "
                "rewardfix-v1 namespaces"
            ),
            "known_limitations": {
                "historical_matched_arm_ratchet_direction": {
                    "affected_tasks": sorted(MATCHED_ARM_LIMITATIONS),
                    "impact": (
                        "the shared proposer/context A/B driver selected its next incumbent with "
                        "the wrong direction even though evaluator combined_score is higher-is-better"
                    ),
                    "treatment": (
                        "points are shown for transparency but excluded from clean matched-arm "
                        "aggregate claims; the driver is fixed for future runs"
                    ),
                }
            },
        },
        "published_ttt_discover_qwen3_8b": {
            "budget": TTT_PUBLISHED_BUDGET, "combined_scores": TTT_PUBLISHED,
            "included_in_metrics": False,
        },
        "tasks": {},
    }

    styles = {
        "proposer_full": dict(color="#1769aa", marker="o", linestyle="-", linewidth=2.7,
                              drawstyle="steps-post",
                              markersize=4.5,
                              label="Update proposer weights (ours)"),
        "proposer": dict(color="#2e9e5b", marker="D", linestyle="-", linewidth=2.5,
                         markersize=5.0,
                         label="Update proposer weights (controlled arm)"),
        "context": dict(color="#6f6f6f", marker="^", linestyle="-.", linewidth=2.3,
                        drawstyle="steps-post",
                        markersize=5.0, markerfacecolor="white",
                        label="Analyzer context (ours ablation; weights frozen)"),
        "executor": dict(color="#e67e22", marker="s", linestyle="--", linewidth=2.5,
                         drawstyle="steps-post",
                         markersize=5.0, markerfacecolor="white",
                         label="Executor weights (local scaled TTT-style reference)"),
    }

    for ax, (task, title, tag) in zip(axes_flat, plot_tasks):
        proposer_full, pfull_src = proposer_curve(task)
        clean_proposer_workspace: Path | None = None
        clean_proposer_outer_root: Path | None = None
        expected_clean_route = SOTA_CLEAN_PROPOSER_ROUTES.get(task)
        if expected_clean_route is not None:
            expected_workspace, expected_outer_root = expected_clean_route
            if any(
                str(expected_outer_root) in str(source)
                for source in pfull_src
            ):
                clean_proposer_workspace = expected_workspace
                clean_proposer_outer_root = expected_outer_root
        # The compact paper layouts intentionally remove the old controlled
        # matched-arm line.  Do not even parse that historical lineage here:
        # besides being irrelevant to the displayed three-route comparison,
        # some of its terminal-null rewards predate the corrected attribution
        # rule and must not become a hidden dependency of the clean SOTA plot.
        if args.layout == "2x3":
            proposer, psrc = matched_proposer_curve(task)
        else:
            proposer, psrc = [], []
        if sota_layout:
            context_workspace = SOTA_CONTEXT_WORKSPACES[task]
        else:
            context_workspace = None
        context, csrc, cws = context_curve(
            task, preferred_workspace=context_workspace
        )
        executor_root = (
            TTT_AHC039_STATE_ROOT
            if sota_layout and task == "eft__ahc_simpletes__ahc039"
            else (TTT_EPLB_STATE_ROOT
                  if sota_layout and task == "adrs__eplb"
                  else (TTT_EXTRA_STATE_ROOT
                        if sota_layout and task in (
                            "eft__math__hadamard_maximal_det",
                            "adrs__txn_scheduling",
                        )
                        else TTT_STATE_ROOT))
        )
        executor, esrc = executor_curve(task, tag, executor_root)
        proposer_reported_score = reported_proposer_score(task, proposer_full)
        all_series = {"proposer_full": proposer_full, "proposer": proposer,
                      "context": context, "executor": executor}
        endpoint_validation_rows = {
            method: endpoint_revalidation(task, method)
            for method in ("proposer", "context", "executor")
        }
        endpoint_program_hashes = [
            str(row.get("program_sha256"))
            for row in endpoint_validation_rows.values()
            if row and row.get("program_sha256")
        ]
        same_endpoint_program_all_routes = (
            len(endpoint_program_hashes) == 3
            and len(set(endpoint_program_hashes)) == 1
        )
        displayed_series = (all_series if args.layout == "2x3" else
                            {name: points for name, points in all_series.items()
                             if name != "proposer"})
        common_rollout_budget = min(
            int(points[-1]["x"]) for points in displayed_series.values()
        )
        for name, points in displayed_series.items():
            if not points:
                continue
            xs = [p["x"] for p in points]
            ys = [norm_score(task, p["score"]) for p in points]
            ax.plot(xs, ys, **styles[name])
            # Emphasize the measured endpoint, not an extrapolated asymptote.
            ax.scatter([xs[-1]], [ys[-1]], s=48, zorder=5,
                       marker=styles[name]["marker"],
                       facecolor=styles[name]["color"], edgecolor="white", linewidth=0.7)
        if task in TTT_PUBLISHED:
            ax.scatter([TTT_PUBLISHED_BUDGET],
                       [norm_score(task, TTT_PUBLISHED[task])], marker="*", s=190,
                       color="#b23b2d", edgecolor="white", linewidth=0.6, zorder=6,
                       label="TTT-Discover Qwen3-8B (published; 25.6k)")
        ax.axhline(1.0, color="#333333", linewidth=1.1, linestyle=":", alpha=0.85)
        reference_label = "human best" if sota_layout else "published ≤10B best"
        # Anchor the label to the actual y=1 reference line.  Adding 0.01 in
        # data units pushes the label far outside tightly zoomed panels (for
        # example EPLB) and makes it look like a stray figure subtitle.
        ax.annotate(
            reference_label,
            xy=(0.02, 1.0), xycoords=ax.get_yaxis_transform(),
            xytext=(0, 3), textcoords="offset points",
            ha="left", va="bottom", fontsize=9.5, color="#333333",
            annotation_clip=True,
        )
        ax.set_xscale("log")
        ax.set_xlim(left=0.85)
        if args.layout in ("1x4", "1x5", "1x7"):
            ax.axvline(
                common_rollout_budget, color="#8a8a8a", linewidth=0.9,
                linestyle=(0, (2, 2)), alpha=0.72, zorder=0,
            )
            ax.text(
                common_rollout_budget, 0.985,
                f"common B={common_rollout_budget}",
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="top", fontsize=8.2, color="#666666",
            )
        plotted_y = [norm_score(task, p["score"])
                     for pts in displayed_series.values() for p in pts]
        if sota_layout:
            ylo, yhi = min([1.0] + plotted_y), max([1.0] + plotted_y)
            yspan = max(yhi - ylo, 0.003)
            ax.set_ylim(max(0.0, ylo - 0.14 * yspan), yhi + 0.16 * yspan)
        else:
            ymax = max([1.05] + plotted_y)
            ax.set_ylim(-0.04, max(1.12, ymax + 0.08))
        display_title = title if args.layout == "2x3" else title.replace(" †", "")
        ax.set_title(display_title, fontsize=15.5)
        if same_endpoint_program_all_routes:
            ax.text(
                0.5, 0.91, "same endpoint program", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color="#333333",
                bbox={"facecolor": "white", "edgecolor": "#aaaaaa",
                      "alpha": 0.82, "pad": 1.8},
            )
        ax.grid(True, which="major", linestyle="--", alpha=0.28)
        ax.tick_params(labelsize=11)

        if args.layout in ("1x4", "1x5", "1x7"):
            endpoint_names = ("proposer_full", "context", "executor")
            endpoint_labels = ("Prop.", "Ctx.", "Exec.")
            endpoint_scores = [
                proposer_reported_score,
                all_series["context"][-1]["score"],
                all_series["executor"][-1]["score"],
            ]
            endpoint_values = [norm_score(task, score)
                               for score in endpoint_scores]
            endpoint_raw = [display_score(task, score)
                            for score in endpoint_scores]
            endpoint_colors = [styles[name]["color"] for name in endpoint_names]
            validation_values = []
            for method in ("proposer", "context", "executor"):
                row = endpoint_validation_rows[method]
                stats = (row or {}).get("statistics") or {}
                if stats.get("mean") is not None:
                    mean = norm_score(task, float(stats["mean"]))
                    std = float(stats.get("std") or 0.0) / plot_anchors[task][1]
                    validation_values.extend((mean - std, mean + std))
            endpoint_hi = max([1.0] + endpoint_values + validation_values)
            endpoint_lo = min(endpoint_values + validation_values)
            endpoint_span = max(endpoint_hi - endpoint_lo, 0.004)
            zoom_lo = max(-0.02, endpoint_lo - 0.18 * endpoint_span)
            zoom_hi = endpoint_hi + 0.12 * endpoint_span

            zoom = ax.inset_axes([0.47, 0.065, 0.51, 0.31])
            ypos = [2, 1, 0]
            zoom.barh(
                ypos,
                [value - zoom_lo for value in endpoint_values],
                left=zoom_lo,
                color=endpoint_colors,
                height=0.58,
                alpha=0.92,
            )
            for y, method in zip(ypos, ("proposer", "context", "executor")):
                row = endpoint_validation_rows[method]
                stats = (row or {}).get("statistics") or {}
                if stats.get("mean") is None:
                    continue
                mean = norm_score(task, float(stats["mean"]))
                std = float(stats.get("std") or 0.0) / plot_anchors[task][1]
                zoom.errorbar(
                    [mean], [y], xerr=[std], fmt="|", color="#111111",
                    ecolor="#111111", elinewidth=1.0, capsize=2.2,
                    markersize=7.0, markeredgewidth=1.2, zorder=7,
                )
            zoom.axvline(1.0, color="#333333", linewidth=0.9, linestyle=":")
            zoom.set_xlim(zoom_lo, zoom_hi)
            zoom.set_ylim(-0.55, 2.55)
            zoom.set_yticks(ypos, endpoint_labels, fontsize=7.5)
            zoom_reference_label = "Human"
            zoom.set_xticks([1.0], [zoom_reference_label], fontsize=7.5)
            zoom.tick_params(axis="both", length=2, pad=1)
            # A short title stays inside each narrow panel in the 1x7 layout.
            zoom.set_title("endpoint zoom", fontsize=8.0, pad=1.5)
            for y, raw, color in zip(ypos, endpoint_raw, endpoint_colors):
                raw_text = format_display_score(task, raw)
                zoom.text(
                    0.98, y, raw_text,
                    transform=zoom.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.5,
                    color="#111111", fontweight="semibold",
                )
            zoom.set_facecolor((1, 1, 1, 0.94))
            for spine in zoom.spines.values():
                spine.set_color("#9a9a9a")
                spine.set_linewidth(0.6)

        proposer_failed_outer_jobs: list[dict[str, Any]] = []
        if clean_proposer_workspace is not None:
            run_manifest = clean_proposer_workspace / "run_manifest.json"
            if run_manifest.exists():
                try:
                    proposer_failed_outer_jobs = list(
                        json.loads(run_manifest.read_text()).get(
                            "failed_outer_jobs") or [])
                except Exception:
                    proposer_failed_outer_jobs = []
        manifest["tasks"][task] = {
            "title": title, "context_workspace": cws,
            "proposer_outer_root": str(
                clean_proposer_outer_root
                if clean_proposer_outer_root is not None else ROOT / "outer"
            ),
            "proposer_workspace": str(
                clean_proposer_workspace
                if clean_proposer_workspace is not None
                else "historical task-local workspaces"
            ),
            "proposer_failed_outer_jobs": proposer_failed_outer_jobs,
            "reported_proposer_combined_score": proposer_reported_score,
            "reported_proposer_display_score": display_score(
                task, proposer_reported_score),
            "ledgered_proposer_curve_endpoint": proposer_full[-1]["score"],
            "task_specific_common_rollout_budget": common_rollout_budget,
            "reported_endpoint_has_local_compute_ledger": (
                True
            ),
            "matched_arm_valid_for_clean_aggregate": task not in MATCHED_ARM_LIMITATIONS,
            "matched_arm_limitation": MATCHED_ARM_LIMITATIONS.get(task),
            "endpoint_revalidation": endpoint_validation_rows,
            "same_endpoint_program_all_routes": same_endpoint_program_all_routes,
            "series": {
                "proposer": {"points": proposer, "sources": psrc,
                             "displayed": args.layout == "2x3"},
                "proposer_full": {"points": proposer_full, "sources": pfull_src,
                                  "displayed": True,
                                  "included_in_common_budget_metrics": (
                                      sota_layout)},
                "context": {"points": context, "sources": csrc},
                "executor": {"points": executor, "sources": esrc},
            },
        }

    axes_flat[0].set_ylabel(y_label, fontsize=13)
    if args.layout == "2x3":
        axes_flat[3].set_ylabel(y_label, fontsize=13)
        xlabel_axes = axes_flat[3:]
        legend_columns, legend_y, footer_y = 3, 0.105, 0.022
    else:
        xlabel_axes = axes_flat
        legend_columns, legend_y, footer_y = 3, 0.277, None
    if args.layout in ("1x4", "1x7"):
        fig.text(
            0.5, 0.355,
            "cumulative executor trajectories incl. shared H2 (log)",
            ha="center", va="center", fontsize=12.5,
        )
    else:
        for ax in xlabel_axes:
            ax.set_xlabel("executor trajectories incl. shared H2 (log)",
                          fontsize=11.8)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_columns, frameon=False,
               bbox_to_anchor=(0.5, legend_y), fontsize=12.5)
    if args.layout == "2x3":
        limitation_footer = (
            "† Historical matched proposer/context arms used the wrong ratchet direction "
            "for Erdős and Autocorrelation I; shown for transparency, excluded from "
            "clean aggregate claims."
        )
        fig.text(0.5, footer_y,
                 "Green/gray/orange arms own independent ratchets; the blue full campaign is ceiling context and is excluded from common-budget metrics.\n"
                 + limitation_footer + "\n"
                 "Solid endpoints are observed best-so-far values; no curve is extrapolated. "
                 "Stars are published Qwen3-8B endpoints at 25.6k rollouts. "
                 "Rollout efficiency excludes analyzer and training GPU cost; one campaign per matched arm, so no uncertainty estimate.",
                 ha="center", fontsize=8.2, color="#444444")
    else:
        cost_ax = fig.add_axes([0.055, 0.125, 0.935, 0.125])
        cost_ax.axis("off")
        route_rows = [
            [
                "Proposer weights (ours)",
                "incumbent + bounded feedback -> target K=8 H1 proposals on one trained-phi replica\n-> frozen-executor trajectories (4 GPUs reserved)",
                "terminal reward -> task-local harness ratchet + proposer LoRA\nr64/a128 (116.4M); 3 epochs; GBS 8; adapter continues",
            ],
            [
                "Analyzer context (ours ablation)",
                "bounded feedback -> 2 frozen-model analysts after the cold round\n-> target K=8 frozen-proposer H1 proposals -> frozen-executor trajectories",
                "terminal reward -> task-local harness/context ratchet only\nno gradient (proposer and executor weights frozen)",
            ],
            [
                "Executor weights (reference)",
                f"PUCT parent -> target K={TTT_BATCH_K} fixed-harness executor trajectories\n(no H1 proposer or analyzer)",
                "adaptive-beta LOO -> final-edit replay -> executor LoRA\nr32/a64 (58.2M); 1 epoch; GBS 4; adapter continues",
            ],
        ]
        cost_table = cost_ax.table(
            cellText=route_rows,
            colLabels=("Reward route", "Inference per feedback batch", "Local weight update"),
            cellLoc="left",
            colLoc="left",
            colWidths=(0.22, 0.43, 0.35),
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        cost_table.auto_set_font_size(False)
        cost_table.set_fontsize(8.5)
        for (row, col), cell in cost_table.get_celld().items():
            cell.set_edgecolor("#c8c8c8")
            cell.set_linewidth(0.55)
            if row == 0:
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(weight="semibold", color="#222222")
            elif col == 0:
                cell.set_text_props(weight="semibold")
        for row, color in enumerate(("#dcecf8", "#eeeeee", "#fbe8d6"), start=1):
            cost_table[(row, 0)].set_facecolor(color)
        footer_paragraphs = [
            (
                "Fairness: x = one shared fixed-H2 20-eval measurement plus "
                "actual launched executor trajectories; every fresh route "
                "starts from task.initial_program and gains count only after "
                "a batch finishes."
            ),
            (
                "Readout: dashed = largest task-local budget shared by all "
                "routes. Y-axes are independently zoomed but retain "
                "score/human-best units; efficiency uses common-budget "
                "score/AUC, not endpoint pixel height."
            ),
            (
                "Compute: recorded model/evaluator calls (lower bounds), "
                "sandbox time, optimizer boundaries, and sacct GPU-hours are "
                "separate ledgers. Failed/rejected/discarded operational work "
                "remains charged; proposal concurrency differs, so rollout "
                "efficiency is not FLOP efficiency."
            ),
            (
                "Validation: endpoints are observed three-transition plateaus "
                "or explicit caps (never extrapolated, not absolute limits); "
                "black inset tick/whisker = program re-evaluation mean +/- 1 SD "
                "(N>=5). One unpaired campaign/route gives no campaign-level CI."
            ),
            (
                "Selection: four task-level illustrations were chosen after "
                "interim inspection from the predeclared seven-task campaign; "
                "the full 1x7 view and aggregates remain in the appendix, and "
                "this 1x4 view has no population-average claim."
                if args.layout == "1x4" else
                "Scope: the five requested tasks are shown as an exact subview "
                "of the clean-fair 1x7 manifest. Every reported condition is "
                "derived from the same clean-route endpoints."
                if args.layout == "1x5" else
                "Scope: seven preselected strength tasks, not a population "
                "sample; blue inset is the locally ledgered reported "
                "endpoint (unledgered AHC039 559,534 excluded)."
            ),
            (
                "Reference: executor update is a local budget-scaled "
                "TTT-Discover-style baseline, not an official reproduction; "
                "final-edit replay, optimizer cadence, and K=8 differ from the "
                "published full-trajectory K=512 update. Hadamard/Txn reference "
                "qualifications are stated in text."
            ),
        ]
        footer_text = "\n".join(
            textwrap.fill(paragraph, width=270)
            for paragraph in footer_paragraphs
        )
        fig.text(
            0.5, 0.007,
            footer_text,
            ha="center", va="bottom", fontsize=6.25, color="#444444",
            linespacing=1.0,
        )

    png, pdf, data = (out_prefix.with_suffix(".png"), out_prefix.with_suffix(".pdf"),
                      out_prefix.with_name(out_prefix.name + "_data.json"))
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    data.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {png}\nwrote {pdf}\nwrote {data}")


if __name__ == "__main__":
    main()

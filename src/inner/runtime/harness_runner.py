"""Run H2 (M0 + a declarative Cordis harness package) on one EFT task.

Loads ``inner/harness/cordis.yml`` (system prompt, plugins, middleware, and
sampling), injects the frozen-executor serving endpoint and evaluation budget,
and drives one task's edit->evaluate loop through official DSH/Cordis.

The harness *definition* lives entirely in ``inner/harness/`` — this module only
wires the endpoint/budget and collects results, so the outer proposer can later
mutate the package without touching this code.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

# Ensure the repository's ``inner`` and ``cordis_runtime`` packages are importable.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inner.tasks.eft_task import EFTTask  # noqa: E402
from inner.evaluation.eval_runner import EvalOutcome  # noqa: E402
from inner.runtime.session import BudgetLedger, InnerSession, session_scope  # noqa: E402
from inner.harness.tools.discovery import (  # noqa: E402
    edit_solution,
    evaluate_solution,
    finish,
    probe_solution,
)
from cordis_runtime.runner import run_cordis  # noqa: E402
from inner.harness.middleware.generated_context import GeneratedHookTracker  # noqa: E402

CORDIS_YML = Path(__file__).resolve().parents[1] / "harness" / "cordis.yml"


@dataclass
class LLMEndpoint:
    model: str = "qwen3.5-9b"
    base_url: str = "http://127.0.0.1:8800/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = 20
    max_tokens: int = 8192
    timeout: float = 600.0
    max_retries: int = 2
    enable_thinking: bool = False  # Qwen3.5: keep outputs clean/fast (matches Weave)
    seed: Optional[int] = None


@dataclass
class H2Config:
    max_evaluator_calls: int = 10
    max_probe_calls: int = 30
    max_iterations: Optional[int] = None  # None -> use cordis.yml's value
    eval_timeout_s: Optional[float] = None
    python_exe: Optional[str] = None      # interpreter with task deps for eval subprocess
    run_timeout_s: Optional[float] = None  # whole Cordis rollout; default scales by iterations


@dataclass
class InnerResult:
    task_id: str
    source: str
    best_score: float
    seed_score: float
    best_metrics: Dict[str, float]
    best_program: str
    stop_reason: str
    ledger: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    middleware_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    decode_seed: Optional[int] = None
    score_eligible: bool = True
    trajectory: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    cordis_run_dir: Optional[str] = None
    cordis_session_log: Optional[str] = None


def _cordis_entries(cordis_yml: Path) -> List[Dict[str, Any]]:
    """Return concrete plugin rows from one complete candidate patch."""

    payload = yaml.safe_load(Path(cordis_yml).read_text()) or []
    if not isinstance(payload, list):
        raise ValueError(f"{cordis_yml}: Cordis root must be a list")
    entries: List[Dict[str, Any]] = []
    for operation in payload:
        if not isinstance(operation, dict):
            continue
        inserted = operation.get("insert")
        if inserted is not None:
            if not isinstance(inserted, list):
                raise ValueError(f"{cordis_yml}: insert must be a list")
            entries.extend(row for row in inserted if isinstance(row, dict))
        elif operation.get("name"):
            entries.append(operation)
    return entries


def _cordis_components(cordis_yml: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Read trusted SAH component metadata embedded in Cordis plugin rows."""

    found: Dict[str, Dict[str, Dict[str, Any]]] = {
        "tools": {}, "skills": {}, "middlewares": {},
    }
    singular = {"tool": "tools", "skill": "skills", "middleware": "middlewares"}
    for row in _cordis_entries(cordis_yml):
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        sah = config.get("sah") if isinstance(config.get("sah"), dict) else None
        if sah is None:
            continue
        kind = str(sah.get("kind", ""))
        bucket = singular.get(kind)
        name = str(sah.get("name", "")).strip()
        if bucket is None or not name:
            raise ValueError(f"{cordis_yml}: invalid SAH plugin metadata in {row.get('id')}")
        if name in found[bucket]:
            raise ValueError(f"{cordis_yml}: duplicate {kind} component {name!r}")
        found[bucket][name] = {
            **sah,
            "id": str(row.get("id", "")),
            "source": str(row.get("name", "")),
        }
    return found


def _expected_cordis_middlewares(cordis_yml: Path) -> List[str]:
    return sorted(_cordis_components(cordis_yml)["middlewares"])


def _expected_cordis_components(cordis_yml: Path) -> Dict[str, Dict[str, str]]:
    components = _cordis_components(cordis_yml)
    return {
        "tools": {
            name: str(row["source"]) for name, row in components["tools"].items()
        },
        "skills": {
            name: str(row["source"])
            for name, row in components["skills"].items()
            if not bool(row.get("base"))
        },
    }


def _cordis_bridge_config(cordis_yml: Path) -> Dict[str, Any]:
    rows = yaml.safe_load(Path(cordis_yml).read_text(encoding="utf-8")) or []
    matched = [
        row.get("config") or {} for row in rows
        if isinstance(row, dict) and row.get("id") == "sah-bridge"
    ]
    if len(matched) != 1 or not isinstance(matched[0], dict):
        raise ValueError("cordis.yml needs exactly one sah-bridge config")
    return matched[0]


def _required_generated_skills(
    harness_dir: Path, mounted_skills: Dict[str, str],
) -> List[str]:
    """Skills changed by *this* H1 proposal and therefore required for credit.

    A materialized package contains inherited generated skills as well as the
    additions/updates made by the current proposal. Every mounted skill is
    enacted for stable harness semantics; component lineage is the separate,
    authoritative credit boundary. Only ``added`` and ``updated`` skills are
    interventions introduced by this candidate.

    Legacy packages without usable lineage predate that distinction.  Fail
    conservatively by requiring all mounted generated skills for those packages.
    """

    manifest_path = Path(harness_dir) / "component_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        lineage = (manifest.get("lineage") or {}).get("new_skills")
        if not isinstance(lineage, list):
            raise ValueError("new_skills lineage is absent")
        required = {
            str(row.get("name"))
            for row in lineage
            if isinstance(row, dict)
            and str(row.get("status")) in {"added", "updated"}
        }
        return sorted(required.intersection(mounted_skills))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return sorted(mounted_skills)


def _cordis_runtime_component_contract(
    cordis_yml: Path,
    *,
    evaluator_budget: int,
    probe_budget: int,
    current_candidate_skills: Optional[List[str]] = None,
) -> str:
    """Render the exact Cordis composition's model-visible component contract."""

    components = _cordis_components(cordis_yml)
    current = set(current_candidate_skills or [])
    lines = [
        "# Authoritative runtime component contract",
        "<RuntimeComponentContract>",
        (
            "This block is generated by the trusted Python runtime from the exact "
            "cordis.yml and plugin snapshot loaded for this rollout. It overrides "
            "stale or contradictory proposer prose."
        ),
        f"Official evaluator budget: {int(evaluator_budget)} call(s).",
        f"Approximate probe budget: {int(probe_budget)} call(s).",
        "",
        "## Callable tools",
        "- `edit_solution` [CORE]: edit the EVOLVE-BLOCK with exact diffs or a full body.",
        "- `evaluate_solution` [CORE]: run the official evaluator under the fixed budget.",
        "- `probe_solution` [CORE]: obtain a cheap approximate score.",
        "- `finish` [CORE]: retain the best program and conclude the Cordis turn.",
    ]
    for name, row in sorted(components["tools"].items()):
        description = " ".join(str(row.get("description", "")).split())
        trigger = " ".join(str(row.get("trigger", "")).split())
        detail = description or "candidate-defined Cordis tool plugin"
        if trigger:
            detail += f" Trigger: {trigger}"
        lines.append(f"- `{name}` [GENERATED, CONDITIONAL]: {detail}")
    lines.extend([
        "",
        "Generated-tool rule: before the first edit, state whether each generated "
        "tool's trigger applies and cite task/program evidence. Use an applicable "
        "tool at its first relevant point; never call one only for compliance.",
        "",
        "## Automatically enacted skill plugins",
    ])
    base_skills = [
        name for name, row in components["skills"].items() if bool(row.get("base"))
    ]
    for name in sorted(base_skills):
        lines.append(f"- `{name}` [BASE, AUTO-ENACTED BY CORDIS]")
    for name, row in sorted(components["skills"].items()):
        if bool(row.get("base")):
            continue
        lineage = "CURRENT CANDIDATE" if name in current else "INHERITED"
        description = " ".join(str(row.get("description", "")).split())
        suffix = f": {description}" if description else ""
        lines.append(f"- `{name}` [GENERATED, AUTO-ENACTED, {lineage}]{suffix}")
    lines.extend(["", "## Automatic middleware plugins"])
    if not components["middlewares"]:
        lines.append("- No proposer-generated middleware is mounted.")
    for name, row in sorted(components["middlewares"].items()):
        hook = str(row.get("hook", "agent/pre-step"))
        description = " ".join(str(row.get("description", "")).split())
        suffix = f": {description}" if description else ""
        lines.append(f"- `{name}` [GENERATED, AUTOMATIC, hook={hook}]{suffix}")
    lines.extend([
        "",
        "Middleware runs automatically through Cordis events; never try to call it.",
        "",
        "## First-edit obligation",
        "Read and follow every auto-enacted skill section. In the same assistant "
        "step as the first edit, include a concise `Component plan:` mapping the "
        "edit to skill guidance and recording each generated-tool trigger decision.",
        "</RuntimeComponentContract>",
    ])
    return "\n".join(lines)


def _is_score_eligible(stop_reason: str, participation_issues: List[str]) -> bool:
    """Publish only a completed harness route whose proposed components enacted."""

    return stop_reason != "harness_error" and not participation_issues


def _record_cordis_tool_calls(
    session: InnerSession,
    trajectory: Optional[List[Dict[str, Any]]],
    expected: Dict[str, str],
) -> None:
    """Reconcile generated-tool audit with durable Cordis call/result pairs.

    The trusted bridge normally records lifecycle events synchronously.  The
    trajectory is an independent fallback, so only missing counts are added.
    """

    pending: Dict[str, str] = {}
    observed: Dict[str, Dict[str, int]] = {
        name: {"invocations": 0, "completed": 0, "errors": 0}
        for name in expected
    }
    for message in trajectory or []:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name", ""))
                if name not in expected:
                    continue
                call_id = str(block.get("id", ""))
                observed[name]["invocations"] += 1
                pending[call_id] = name
            elif block.get("type") == "tool_result":
                call_id = str(block.get("tool_use_id", ""))
                name = pending.pop(call_id, None)
                if name is None:
                    continue
                if block.get("is_error"):
                    observed[name]["errors"] += 1
                else:
                    observed[name]["completed"] += 1

    for name, counts in observed.items():
        current = session.tool_audit.get(name, {})
        for _ in range(max(0, counts["invocations"] - int(current.get("invocations", 0)))):
            session.record_tool_event(name, "invoked")
        for _ in range(max(0, counts["completed"] - int(current.get("completed", 0)))):
            session.record_tool_event(name, "completed")
        for _ in range(max(0, counts["errors"] - int(current.get("errors", 0)))):
            session.record_tool_event(name, "error", "Cordis tool result reported an error")


def _initial_message(
    task: EFTTask,
    seed_score: float,
    seed_valid: float,
    budget: int,
    generated_skill_playbooks: Optional[Dict[str, str]] = None,
    component_contract: Optional[str] = None,
) -> str:
    msg = (
        f"# Task\n{task.spec.strip()}\n\n"
        f"# Current program\n```python\n{task.initial_program}\n```\n\n"
        f"# Baseline\nThe seed program scores combined_score = {seed_score:.6g} "
        f"(validity {seed_valid:g}). Beat it. You have {budget} evaluations.\n\n"
    )
    parents = getattr(task, "crossover_parents", None)
    if parents:
        msg += "# Alternative high-scoring approaches (different search basins)\n"
        msg += ("These reached similar scores via DIFFERENT strategies. Consider "
                "hybridizing their ideas with the current program — crossover often "
                "escapes local optima that pure mutation cannot:\n")
        for i, par in enumerate(parents):
            msg += (f"\n## Alternative {i+1} (score {par.get('score', 0):.6g})\n"
                    f"```python\n{par.get('program', '')[:4000]}\n```\n")
        msg += "\n"
    if component_contract:
        msg += f"{component_contract.strip()}\n\n"
    msg += (
        "The discovery-optimization Cordis skill plugin is already enacted in "
        "the system prompt; read and follow it before the first edit."
    )
    if generated_skill_playbooks:
        rendered = "\n\n".join(
            f"## `{name}`\n<SkillDetails>\n{body}\n</SkillDetails>"
            for name, body in generated_skill_playbooks.items()
        )
        msg += (
            "\n\n# Automatically enacted proposer-generated skills\n"
            "The trusted runtime has inserted the complete playbooks below "
            "before your first edit. They are interventions in the candidate "
            "H2 being evaluated: follow them in this rollout. You do not need "
            "to load these already-delivered playbooks.\n\n"
            f"{rendered}\n"
        )
    msg += (
        "\nPropose an improved EVOLVE-BLOCK with edit_solution and score "
        "it with evaluate_solution."
    )
    return msg


def run_task(
    task: EFTTask,
    *,
    endpoint: LLMEndpoint,
    h2: H2Config,
    keep_trajectory: bool = True,
    checkpoint_path: Optional[str] = None,
    harness_dir: Optional[Path] = None,
    seed_outcome: Optional[EvalOutcome] = None,
) -> InnerResult:
    """Run H2 through DSH/Cordis; retain Python evaluator and ledger semantics.

    ``harness_dir`` selects a materialized candidate containing a complete
    ``cordis.yml`` and ``plugins/*.mjs``.  Every run snapshots those plugins
    into an isolated DSH profile before composition.
    """

    cordis_yml = CORDIS_YML
    if harness_dir is not None:
        harness_dir = Path(harness_dir).resolve()
        cordis_yml = harness_dir / "cordis.yml"
    if not cordis_yml.is_file():
        raise FileNotFoundError(f"candidate Cordis harness missing: {cordis_yml}")

    component_details = _cordis_components(cordis_yml)
    expected_middlewares = (
        _expected_cordis_middlewares(cordis_yml) if harness_dir is not None else []
    )
    expected_components = (
        _expected_cordis_components(cordis_yml)
        if harness_dir is not None else {"tools": {}, "skills": {}}
    )
    credit_skills = (
        _required_generated_skills(harness_dir, expected_components["skills"])
        if harness_dir is not None else []
    )
    active_skills = sorted(expected_components["skills"])

    ledger = BudgetLedger(
        max_evaluator_calls=h2.max_evaluator_calls,
        max_probe_calls=h2.max_probe_calls,
    )
    component_contract = _cordis_runtime_component_contract(
        cordis_yml,
        evaluator_budget=ledger.max_evaluator_calls,
        probe_budget=ledger.max_probe_calls,
        current_candidate_skills=credit_skills,
    )
    session = InnerSession(task=task, ledger=ledger,
                           eval_timeout_s=h2.eval_timeout_s, python_exe=h2.python_exe,
                           checkpoint_path=checkpoint_path,
                           harness_dir=str(harness_dir) if harness_dir is not None else None)
    for name, source in expected_components["tools"].items():
        session.register_tool(name, source)
    for name, source in expected_components["skills"].items():
        session.register_skill(
            name, source, required_for_credit=name in credit_skills
        )
    for name, row in component_details["middlewares"].items():
        if name in expected_middlewares:
            session.register_middleware(name, str(row.get("hook", "agent/pre-step")))
    hook_tracker = GeneratedHookTracker()

    def cordis_snapshot(iteration: int = 0) -> Dict[str, Any]:
        snapshot = hook_tracker.snapshot(
            SimpleNamespace(current_iteration=int(iteration)), session,
        )
        if not math.isfinite(float(snapshot.get("best_so_far", 0.0))):
            snapshot["best_so_far"] = None
            if isinstance(snapshot.get("state"), dict):
                snapshot["state"]["best_so_far"] = None
        snapshot.update({
            "task_id": task.task_id,
            "current_program": session.current_program[:60000],
            "best_program": (session.best_program or session.current_program)[:60000],
        })
        return snapshot

    def cordis_tool_event(name: str, event: str, error: str = "") -> Dict[str, Any]:
        session.record_tool_event(name, event, error or None)
        return {"recorded": True}

    def cordis_middleware_event(
        name: str, event: str, iteration: int = 0, error: str = "",
    ) -> Dict[str, Any]:
        session.record_middleware_event(
            name, event, iteration=int(iteration), error=error or None,
        )
        return {"recorded": True}
    seed = (
        session.seed_baseline_from_outcome(seed_outcome)
        if seed_outcome is not None
        else session.seed_baseline()
    )
    seed_score = seed.combined_score

    iterations = h2.max_iterations or 36
    run_timeout = h2.run_timeout_s or max(60.0, endpoint.timeout * iterations)
    run_artifacts: Optional[Path] = None
    if checkpoint_path:
        artifact_parent = Path(checkpoint_path).resolve().parent / "cordis"
        artifact_parent.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() else "_" for c in task.task_id)[-48:]
        run_artifacts = Path(tempfile.mkdtemp(prefix=f"{safe_task}.", dir=artifact_parent))

    stop_reason, err, trajectory = "harness_error", None, None
    cordis_result = None
    try:
        bridge_config = _cordis_bridge_config(cordis_yml)
        cordis_result = run_cordis(
            _initial_message(
                task,
                seed_score,
                seed.validity,
                h2.max_evaluator_calls,
                component_contract=component_contract,
            ),
            role="inner",
            patch=cordis_yml,
            tools={
                "edit_solution": edit_solution,
                "evaluate_solution": evaluate_solution,
                "probe_solution": probe_solution,
                "finish": finish,
                "__sah_snapshot": cordis_snapshot,
                "__sah_tool_event": cordis_tool_event,
                "__sah_middleware_event": cordis_middleware_event,
            },
            scope_factory=lambda: session_scope(session),
            model=endpoint.model,
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            max_tokens=endpoint.max_tokens,
            max_retries=endpoint.max_retries,
            temperature=float(bridge_config.get("temperature", endpoint.temperature)),
            request_max_tokens=int(bridge_config.get("maxTokens", endpoint.max_tokens)),
            top_p=float(bridge_config.get("topP", endpoint.top_p)),
            top_k=int(bridge_config.get("topK", endpoint.top_k or 20)),
            seed=endpoint.seed,
            enable_thinking=endpoint.enable_thinking,
            context_window=131072,
            timeout_s=run_timeout,
            workspace=harness_dir or CORDIS_YML.parent,
            run_dir=run_artifacts,
            max_iterations=h2.max_iterations,
        )
        stop_reason = cordis_result.stop_reason
        err = cordis_result.error
        ledger.llm_calls = cordis_result.llm_calls
        if keep_trajectory:
            trajectory = cordis_result.trajectory
        if cordis_result.returncode == 0:
            for name in active_skills:
                session.record_skill_load(
                    name,
                    before_first_edit=True,
                    delivery_mode="runtime_injection",
                )
    except Exception as exc:  # never lose the already-scored seed baseline
        err = f"{type(exc).__name__}: {exc}"

    _record_cordis_tool_calls(session, trajectory, expected_components["tools"])
    middleware_issues = session.middleware_participation_issues(
        expected_middlewares
    )
    skill_issues = session.skill_participation_issues(
        active_skills
    )
    participation_issues = [
        *(f"middleware {issue}" for issue in middleware_issues),
        *(f"skill {issue}" for issue in skill_issues),
    ]
    # A harness exception may leave the evaluator's seed checkpoint in the
    # session.  Keep that diagnostic value, but never publish it as a scored
    # executor trajectory: no completed executor action produced it.
    score_eligible = _is_score_eligible(stop_reason, participation_issues)
    if participation_issues:
        if skill_issues:
            participation_error = (
                "ComponentParticipationError: "
                + "; ".join(participation_issues)
            )
        else:
            participation_error = (
                "MiddlewareParticipationError: "
                + "; ".join(middleware_issues)
            )
        session.history_note(participation_error)
        if trajectory is not None:
            trajectory.append({
                "role": "framework",
                "content": participation_error,
            })
        err = f"{err}; {participation_error}" if err else participation_error
        if stop_reason == "completed":
            stop_reason = (
                "component_participation_failed"
                if skill_issues
                else "middleware_participation_failed"
            )

    s = session.summary()
    return InnerResult(
        task_id=task.task_id, source=task.source,
        best_score=session.best_score if session.best_score != float("-inf") else seed_score,
        seed_score=seed_score, best_metrics=session.best_metrics,
        best_program=session.best_program or task.initial_program,
        stop_reason=stop_reason, ledger=s["ledger"], steps=s["steps"],
        middleware_audit=s["middleware_audit"], tool_audit=s["tool_audit"],
        skill_audit=s["skill_audit"], decode_seed=endpoint.seed,
        score_eligible=score_eligible,
        trajectory=trajectory, error=err,
        cordis_run_dir=(str(cordis_result.run_dir) if cordis_result else None),
        cordis_session_log=(
            str(cordis_result.raw_session_log)
            if cordis_result and cordis_result.raw_session_log else None
        ),
    )

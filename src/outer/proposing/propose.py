"""Run H1 to edit private candidate H2 packages (instance-wise).

Each candidate = one full H1 agent run (inspect -> edit -> validate -> submit)
against the M_phi endpoint, conditioned on ONE task instance. The K runs for a
task share that task's user message (= one GRPO group); diversity comes from
H1's temperature (1.0, fixed in its agent.yaml) plus per-run sampling seeds.
outer_round threads run_once across all (task, k) jobs and serving replicas;
the propose-session contextvar is thread-local, so sessions never collide.
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from outer.genome import harness_spec as hs
from outer.proposing.proposer_io import H1_PACKAGE  # noqa: F401 (used by run_once)
from outer.workspace.propose_session import ProposeSession, propose_scope
from outer.compiling.materialize import materialize
from outer.proposing.forced_tool_starter import seed_starter_tool
from outer.proposing.component_scaffold import (
    TOOL_PLACEHOLDER, seed_middleware_scaffold, placeholder_bodies)
from outer.proposing.proposer_io import render_workspace


@dataclass
class CandidateRecord:
    k: int
    valid: bool
    errors: List[str] = field(default_factory=list)
    raw_submission: str = ""
    spec: Optional[Dict[str, Any]] = None        # validated partial spec
    effective: Optional[Dict[str, Any]] = None   # spec folded over base
    review_log: list = field(default_factory=list)  # validate_harness component audit
    changed_fields: List[str] = field(default_factory=list)
    spec_hash: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    stop_reason: str = ""
    auto_finalized: bool = False
    schema_guard: Dict[str, Any] = field(default_factory=dict)  # TOOLSCHEMA-001
    # REPAIR-001: frozen-M0 takeover of a failed slot. ``trajectory`` always
    # stays the PRIMARY (phi) trajectory — phi's training credit binds to its
    # own actions; the repair trajectory is retained separately.
    repair: Dict[str, Any] = field(default_factory=dict)
    repair_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # EXPLORE-001 (v2.5)
    behavior_policy: str = "trained_phi"
    component_only: bool = False
    # BEAM-001: which beam member this slot descended from
    parent_package: Optional[str] = None
    parent_score: Optional[float] = None


def _dump_history(agent) -> List[Dict[str, Any]]:
    out = []
    try:
        for msg in agent.history:
            try:
                out.append(msg.model_dump())
            except Exception:
                out.append({"role": str(getattr(msg, "role", "?")),
                            "content": str(getattr(msg, "content", ""))[:4000]})
    except Exception:
        pass
    return out


def _configure_transport(config, *, timeout: float, max_retries: int) -> None:
    """Apply the same request-recovery contract used by H2 executors.

    ``LLMConfig.max_retries`` belongs to the provider SDK and retries only the
    failed HTTP request.  ``AgentConfig.retry_attempts`` is a separate outer
    layer that can replay an entire agent iteration.  Keep the latter at one
    total attempt so a transient connection reset never creates an extra H1
    trajectory.
    """

    llm = config.llm_config
    llm.timeout = timeout
    llm.max_retries = max_retries
    if not bool(getattr(llm, "stream", False)):
        llm.stream_idle_timeout_ms = max(1000, int(timeout * 1000))
    config.retry_attempts = 1


REPAIR_MSG = (
    "\n\n## REPAIR SESSION (frozen-executor takeover)\n"
    "A previous proposer session left this H2 workspace INVALID and "
    "unsubmitted. Recorded failure: {failure}\n"
    "Your ONLY job is to drive THIS existing workspace to a valid submission "
    "with the smallest possible further change — do not start new ambitious "
    "components. If the current workspace files are included in this "
    "message, do NOT re-inspect them; otherwise start with "
    "harness_shell(command=\"cat agent.yaml\"). Procedure: "
    "1) validate_harness to see the current errors; 2) apply minimal fixes "
    "— delete or shrink an unfixable generated component (a REQUIRED-tool "
    "candidate must keep one mounted generated tool: shrink it, never "
    "delete the last one); if a LENGTH CAP is reported (e.g. prompt.md over "
    "10000 chars), REPLACE a long section with a shorter one "
    "(old_text/new_text) — appending is refused over the cap; "
    "3) validate_harness; 4) submit_harness."
)


def _run_h1_agent(
    *, user_message: str, base_url: str, model: str, api_key: str,
    seed: Optional[int], timeout: float, max_retries: int,
    max_iterations: Optional[int] = None,
) -> tuple:
    """One NexAU H1 agent pass over the CURRENT propose session's draft.

    Returns (trajectory, err). The caller owns the session/contextvar scope.
    """
    from nexau import Agent, AgentConfig

    config = AgentConfig.from_yaml(H1_PACKAGE / "agent.yaml")
    llm = config.llm_config
    llm.model = model
    llm.base_url = base_url
    llm.api_key = api_key
    _configure_transport(config, timeout=timeout, max_retries=max_retries)
    if max_iterations is not None:
        config.max_iterations = int(max_iterations)
    extra = getattr(llm, "extra_params", None)
    if not isinstance(extra, dict):
        extra = {}
        try:
            llm.extra_params = extra
        except Exception:
            pass
    eb = extra.setdefault("extra_body", {})
    eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if seed is not None:
        eb["seed"] = seed

    trajectory: List[Dict[str, Any]] = []
    err: Optional[str] = None
    agent = None
    try:
        agent = Agent(config=config)
        agent.run(message=user_message)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        if agent is not None:
            trajectory = _dump_history(agent)
    return trajectory, err


def _salvage_once(session, error_text: str, trajectory) -> str:
    """SCAFFOLD-006: drop a review-failing generated tool and revalidate.

    'generated tool X failed validate_harness' wedges the whole slot even
    when the rest of the workspace is good.  In a non-forced slot the
    failing tool is deleted (file + mount; the stale-declaration strip
    removes its prompt line) and validation runs once more.  Forced slots
    keep the failure: deleting their last tool would break the contract.
    """
    import re

    import yaml as _yaml
    if getattr(session, "require_new_tool", False) or session.draft_dir is None:
        return error_text
    names = set(re.findall(r"generated tool '([\w\-]+)' failed", error_text))
    if not names:
        return error_text
    root = Path(session.draft_dir)
    try:
        agent = _yaml.safe_load((root / "agent.yaml").read_text()) or {}
        agent["tools"] = [m for m in (agent.get("tools") or [])
                          if str(m.get("name")) not in names]
        (root / "agent.yaml").write_text(
            _yaml.safe_dump(agent, sort_keys=False))
        for name in names:
            for rel in (f"custom_tools/{name}.py", f"tools/{name}.tool.yaml"):
                f = root / rel
                if f.is_file():
                    f.unlink()
    except Exception:
        return error_text
    session.workspace_revision += 1
    trajectory.append({
        "role": "framework",
        "content": "[FINALIZE-001] salvage: dropped review-failing tool(s) "
                   + ", ".join(sorted(names)) + " and revalidating once.",
    })
    return session.validate_harness()


def maybe_auto_finalize(session, trajectory) -> bool:
    """FINALIZE-001: submit a finished-but-unsubmitted workspace, once.

    7/11 invalid slots in the fix3 taxonomy ended with a complete, often
    valid workspace and no submit call.  The normal submit path runs with
    every guard live (placeholder refusal, novelty guard, no-op check), so
    an untouched scaffold or a futile repeat still comes back invalid —
    this only rescues work the model actually finished, and the flag is
    recorded so analysis can separate auto-finalized slots.
    """
    if session.submitted:
        return False
    try:
        if session.validated_revision != session.workspace_revision:
            v = session.validate_harness()
            if not str(v).startswith("VALID"):
                v = _salvage_once(session, str(v), trajectory)
            if not str(v).startswith("VALID"):
                trajectory.append({
                    "role": "framework",
                    "content": "[FINALIZE-001] auto-finalize skipped: the "
                               "final workspace does not validate: "
                               + str(v)[:200],
                })
                return False
        out = session.submit_harness()
    except Exception:
        return False
    # the failure shape is "SUBMITTED but INVALID (minimum reward)" — only a
    # real acceptance counts
    if isinstance(out, str) and "Candidate H2 accepted" in out:
        trajectory.append({
            "role": "framework",
            "content": "[FINALIZE-001] iteration budget ended without "
                       "submit_harness; the workspace validated clean and "
                       "was auto-finalized: " + out[:200],
        })
        return True
    trajectory.append({
        "role": "framework",
        "content": "[FINALIZE-001] auto-finalize attempted and REFUSED: "
                   + str(out)[:200],
    })
    return False


def run_once(k: int, *, base_spec: Dict[str, Any], user_message: str,
             futile_signatures=None, component_only: bool = False,
             component_kind: str = "",
              base_url: str, model: str, api_key: str, seed: Optional[int],
              timeout: float, max_retries: int = 2,
              require_new_tool: bool = False,
              repair_base_url: Optional[str] = None,
              repair_max_iters: int = 16) -> CandidateRecord:
    trajectory: List[Dict[str, Any]] = []
    err: Optional[str] = None
    # Every parallel proposer gets a complete private copy.  H1 can inspect and
    # edit real H2 files, but can never mutate the shared accepted package or
    # another candidate's draft.
    with TemporaryDirectory(prefix=f"h1_candidate_{k:02d}_") as td:
        draft_dir = Path(td) / "h2"
        try:
            materialize(
                base_spec,
                draft_dir,
                meta={"effective": base_spec},
                # Historical parents may predate proposer-owned component
                # catalogs. Preserve their prompt byte-for-byte in the private
                # draft so H1—not runtime—must repair it before submission.
                validate_prompt=False,
            )
        except Exception as e:
            return CandidateRecord(
                k=k,
                valid=False,
                errors=[f"cannot initialize private H2 workspace: {type(e).__name__}: {e}"],
                stop_reason="workspace_error",
            )
        if component_kind in ("tool", "middleware"):
            # SCAFFOLD-001: the plumbing (schema + mount + declaration, or the
            # hook wrapper) is seeded so the proposer spends its turns on the
            # BODY.  The component-only submit guard still refuses a slot that
            # ships the placeholder unchanged.
            try:
                if component_kind == "tool":
                    seed_starter_tool(draft_dir, k, body=TOOL_PLACEHOLDER)
                else:
                    seed_middleware_scaffold(draft_dir, k)
            except Exception as e:
                print(f"[propose] WARNING: scaffold for k{k} failed ({e})")
        if require_new_tool and component_kind != "tool":
            # BAILOUT-005: forced slots start from a mounted, valid minimal
            # tool — improve/replace instead of create-from-scratch.
            try:
                seed_starter_tool(draft_dir, k)
            except Exception as e:
                return CandidateRecord(
                    k=k,
                    valid=False,
                    errors=[f"cannot seed forced-tool starter: {e}"],
                    stop_reason="workspace_error",
                )
        # PRELOAD-001: render the whole mutable workspace into the initial
        # context so the proposer edits directly instead of cat-ing files for
        # half its budget; the iteration cap shrinks accordingly.
        preload = os.environ.get("SAH_H1_PRELOAD", "1") == "1"
        h1_max_iters: Optional[int] = None
        primary_message = user_message
        session = ProposeSession(
            base_spec=base_spec,
            draft_dir=draft_dir,
            require_new_tool=require_new_tool,
            slot_index=k,
            futile_signatures=set(futile_signatures or ()),
            component_only=component_only,
            component_kind=component_kind,
        )
        if preload:
            ws_text, rendered = render_workspace(draft_dir)
            primary_message = user_message + ws_text
            session.mark_preloaded(rendered)
            h1_max_iters = int(os.environ.get("SAH_H1_MAX_ITERS", "12"))
            if component_only or require_new_tool:
                # Measured on the AC2 v2.6 run: the skill slot submitted in 7
                # tool calls (one file + mount + one line), while the tool and
                # middleware slots ran out at 11 — they must also write a
                # schema or a hook wrapper.  Give component slots the extra
                # turns instead of losing the slot.
                h1_max_iters = int(os.environ.get(
                    "SAH_H1_COMPONENT_MAX_ITERS", str(h1_max_iters + 6)))
        with propose_scope(session):
            trajectory, err = _run_h1_agent(
                user_message=primary_message, base_url=base_url, model=model,
                api_key=api_key, seed=seed, timeout=timeout,
                max_retries=max_retries, max_iterations=h1_max_iters,
            )
        primary_guard = session.schema_guard_summary()

        # REPAIR-001: a slot phi failed is taken over by the FROZEN executor
        # model on the same still-alive draft.  Capability parity keeps the
        # no-strong-model-leakage invariant; phi's training credit stays bound
        # to its own (failed) trajectory — the collector assigns it the
        # minimum reward.
        repair_info: Dict[str, Any] = {}
        repair_trajectory: List[Dict[str, Any]] = []
        primary_valid = bool(
            session.submitted and session.effective is not None
            and session.changed_fields and not session.errors
        )
        if not primary_valid and repair_base_url:
            original_errors = list(
                session.errors or ["proposer never called submit_harness"]
            )
            failure = "; ".join(original_errors)[:600]
            repair_session = ProposeSession(
                base_spec=base_spec, draft_dir=draft_dir,
                require_new_tool=require_new_tool, slot_index=k,
                futile_signatures=set(futile_signatures or ()),
                component_only=component_only,
                component_kind=component_kind,
            )
            repair_message = user_message + REPAIR_MSG.format(failure=failure)
            if preload:
                    # Re-render: the draft now carries the failed session's edits.
                ws_text, rendered = render_workspace(draft_dir)
                repair_message += ws_text
                repair_session.mark_preloaded(rendered)
            with propose_scope(repair_session):
                repair_trajectory, repair_err = _run_h1_agent(
                    user_message=repair_message,
                    base_url=repair_base_url, model=model, api_key=api_key,
                    seed=(seed + 500000) if seed is not None else None,
                    timeout=timeout, max_retries=max_retries,
                    max_iterations=repair_max_iters,
                )
            repair_valid = bool(
                repair_session.submitted
                and repair_session.effective is not None
                and repair_session.changed_fields
                and not repair_session.errors
            )
            repair_info = {
                "attempted": True,
                "succeeded": repair_valid,
                "repair_model": "qwen3.5-9b-family",
                "repair_base_url": repair_base_url,
                "original_errors": original_errors,
                "repair_error": repair_err,
                "llm_calls": sum(
                    1 for m in repair_trajectory
                    if str(m.get("role", "")).lower().endswith("assistant")
                ),
            }
            if repair_valid:
                session = repair_session

        # FINALIZE-001 must run while the draft workspace still exists —
        # the first live round showed it firing after TemporaryDirectory
        # cleanup, where validate can only report missing agent.yaml.
        #
        # CREDIT-002: it must NOT run after a repair session has edited the
        # shared draft.  When repair edits the workspace but never lands a
        # successful submit, `session` is still phi's — auto-finalizing then
        # submits the FROZEN MODEL's files under phi's identity, with
        # repair.succeeded False, so the collector's minimum-reward rule
        # (keyed on that flag) never fires and phi is trained at full reward
        # on a candidate it did not author.
        repair_touched_draft = bool(repair_info.get("attempted"))
        auto_finalized = (
            (not err) and not repair_touched_draft
            and maybe_auto_finalize(session, trajectory)
        )
        if repair_touched_draft and not session.submitted:
            trajectory.append({
                "role": "framework",
                "content": "[FINALIZE-001] skipped: a repair session edited "
                           "this draft, so finalizing it would credit phi "
                           "with the frozen model's work.",
            })

    valid = bool(session.submitted and session.effective is not None
                 and session.changed_fields and not session.errors)
    repaired = bool(repair_info.get("succeeded"))
    rec = CandidateRecord(
        k=k, valid=valid,
        raw_submission=session.raw_submission,
        spec=session.partial_spec, effective=session.effective,
        changed_fields=session.changed_fields,
        spec_hash=hs.spec_hash(session.effective) if session.effective else "",
        trajectory=trajectory,
        llm_calls=sum(1 for m in trajectory
                      if str(m.get("role", "")).lower().endswith("assistant")),
        stop_reason="repaired_submission" if repaired else
                    ("harness_error" if err else
                     ("auto_finalized" if auto_finalized and valid
                      else "submitted" if session.submitted and session.effective
                      else "invalid_submission" if session.submitted
                      else "no_submission")),
    )
    rec.auto_finalized = auto_finalized
    rec.review_log = list(session.component_audit)
    rec.schema_guard = primary_guard
    rec.repair = repair_info
    rec.repair_trajectory = repair_trajectory
    if repaired:
        rec.errors = []
    elif err:
        rec.errors = [err]
    elif not session.submitted:
        rec.errors = ["proposer never called submit_harness"]
    elif session.errors:
        rec.errors = session.errors
    return rec


def dedup_group(records: List[CandidateRecord], base_spec: Dict[str, Any]) -> None:
    """Invalidate duplicates within one task's GRPO group (in place)."""
    seen = {hs.spec_hash(base_spec)}
    for rec in sorted(records, key=lambda r: r.k):
        if not rec.valid:
            continue
        if rec.spec_hash in seen:
            rec.valid = False
            rec.errors = ["duplicate of another candidate (or of the base)"]
        else:
            seen.add(rec.spec_hash)


PROPOSAL_GATE_MODES = ("off", "all")


def parse_gate_mode(mode: str, k: int) -> tuple:
    """Return (kind, min_valid) for a SAH_PROPOSAL_GATE value.

    ``off`` -> no enforcement; ``all`` -> every charged slot valid;
    ``min<N>`` (e.g. ``min6``) -> at least N of K slots must hold a valid
    materialized candidate (REPAIR-001 relaxation: invalid slots are charged,
    trained at minimum reward, and simply do not roll out).
    """

    if mode in ("off", "all"):
        return mode, k if mode == "all" else 0
    if mode.startswith("min") and mode[3:].isdigit():
        n = int(mode[3:])
        if 1 <= n <= k:
            return "min", n
    raise ValueError(
        f"unknown SAH_PROPOSAL_GATE mode {mode!r}; expected off, all, or min<N>"
    )


def proposal_gate_issues(
    candidates: List[Dict[str, Any]], *, k: int,
) -> List[str]:
    """Fail-closed ``all`` proposal gate over one task's candidate entries.

    A causal method-comparison round may only charge intervention slots that
    hold a valid, materialized candidate H2. The fair16-v5 campaign showed why
    this must be a hard gate: 16 of 19 update_harness circle rounds proposed
    0/8 valid H2, every slot silently ran the incumbent-fallback package, and
    the rounds still completed as official curve points. Returns a list of
    per-slot issues; empty means the task passes.
    """

    issues: List[str] = []
    if len(candidates) != k:
        issues.append(
            f"charged slot count {len(candidates)} != K={k}"
        )
    for cand in sorted(candidates, key=lambda c: int(c.get("k", -1))):
        slot = int(cand.get("k", -1))
        if not cand.get("valid"):
            reason = cand.get("stop_reason") or "invalid"
            first_error = (cand.get("errors") or ["unspecified"])[0]
            issues.append(
                f"k{slot:02d}: invalid proposal ({reason}: {first_error})"
            )
        elif not cand.get("dir"):
            issues.append(f"k{slot:02d}: valid proposal was never materialized")
    return issues


def enforce_new_tool_contract(
    rec: CandidateRecord, base_spec: Dict[str, Any], *, required: bool,
) -> CandidateRecord:
    """Fail closed when an exploration slot promised a genuinely new tool.

    The force-tool arm used to exist only as natural-language guidance. A
    valid prompt-only proposal could therefore enter rollout and receive
    reward despite violating the experimental arm. This postcondition is
    checked before deduplication/materialization and mutates the record in the
    same way as other proposal validity gates.
    """

    if not required or not rec.valid:
        return rec
    added = hs.added_generated_component_names(
        base_spec, rec.effective or {}, "new_tools"
    )
    if added:
        return rec
    rec.valid = False
    rec.errors = [
        "forced-tool contract violation: candidate did not add a new mounted "
        "generated tool"
    ]
    rec.stop_reason = "forced_tool_contract_violation"
    return rec

"""Per-task harness session: state the tools read/mutate, plus the budget ledger.

The inner harness runs a NexAU agent whose tools operate on a single
:class:`InnerSession` held in a contextvar (mirrors Weave's RuntimeBridge
pattern). The session owns the current candidate program, best-so-far tracking
(the reward), and the external budget ledger (plan.md §8.4: usage is reserved
and accounted here, never self-reported by the harness).
"""
from __future__ import annotations

import contextvars
import tempfile
import textwrap
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from inner.editing import program_edit as pe
from inner.tasks.eft_task import EFTTask
from inner.evaluation.eval_runner import EvalOutcome, evaluate_program


@dataclass
class BudgetLedger:
    """External accounting of every model/evaluator call (plan.md §8.4)."""

    max_evaluator_calls: int = 10
    evaluator_calls: int = 0
    max_probe_calls: int = 30
    probe_calls: int = 0
    edit_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    sandbox_seconds: float = 0.0

    def evaluator_budget_left(self) -> int:
        return max(0, self.max_evaluator_calls - self.evaluator_calls)

    def exhausted(self) -> bool:  # alias for the SDK
        return self.evaluator_exhausted()

    def evaluator_exhausted(self) -> bool:
        return self.evaluator_calls >= self.max_evaluator_calls


@dataclass
class StepRecord:
    step: int
    kind: str  # "seed" | "edit_eval"
    edit_mode: str
    edit_note: str
    combined_score: float
    validity: float
    error: Optional[str]
    wall_s: float
    is_new_best: bool


@dataclass
class InnerSession:
    task: EFTTask
    ledger: BudgetLedger
    eval_timeout_s: Optional[float] = None
    python_exe: Optional[str] = None
    checkpoint_path: Optional[str] = None  # if set, best-so-far is written here after each new best (wall-safe)
    harness_dir: Optional[str] = None

    current_program: str = ""
    best_score: float = float("-inf")
    best_program: str = ""
    best_metrics: Dict[str, float] = field(default_factory=dict)
    history: List[StepRecord] = field(default_factory=list)
    middleware_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_gate: Optional[Dict[str, Any]] = field(default=None, repr=False)
    tool_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_audit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _custom_scratch: Optional[str] = field(default=None, repr=False)
    _pending_edit_note: str = "seed"
    _pending_edit_mode: str = "seed"

    def __post_init__(self) -> None:
        if not self.current_program:
            self.current_program = self.task.initial_program

    # -- tool-facing operations -------------------------------------------- #
    def apply_edit(self, code: str) -> str:
        """Apply an edit in either OpenEvolve format M0 knows:

        * SEARCH/REPLACE diff (``<<<<<<< SEARCH`` ... ``>>>>>>> REPLACE``) applied
          to the full current program — targeted changes, the OpenEvolve default;
        * otherwise a full rewrite of the EVOLVE-BLOCK body (fences optional).
        """
        self.ledger.edit_calls += 1
        if "<<<<<<< SEARCH" in code:
            new_prog, n = pe.apply_diff(self.current_program, code)
            if n > 0:
                self.current_program = new_prog
                self._pending_edit_mode, self._pending_edit_note = "diff", f"applied {n} diff block(s)"
                return f"edit staged (diff: {n} block(s) applied). Call evaluate_solution to score it."
            return (
                "No SEARCH section matched the current program verbatim, so nothing "
                "changed. Copy the exact lines to replace into the SEARCH block, or "
                "send the full EVOLVE-BLOCK code instead."
            )
        raw = code
        if "```" in raw:
            extracted = pe.parse_full_rewrite(raw, self.task.language)
            if extracted:
                raw = extracted
        if pe.BLOCK_START in raw and pe.BLOCK_END in raw:
            raw = pe.split_program(raw).block
        # M0 sometimes emits the whole block uniformly indented (copied from
        # inside a class/function) — spliced verbatim that is an instant
        # "IndentationError: unexpected indent (line 2)" (80 wasted evals
        # across the K16 push rounds). dedent only strips indentation COMMON
        # to every line, so correct blocks are untouched and mixed-indent
        # garbage still fails exactly as before.
        first = next((ln for ln in raw.splitlines() if ln.strip()), "")
        if first[:1] in (" ", "\t"):
            raw = textwrap.dedent(raw)
        self.current_program = pe.split_program(self.current_program).assemble(raw)
        self._pending_edit_mode, self._pending_edit_note = "full_rewrite", "spliced new EVOLVE-BLOCK"
        return "edit staged (full EVOLVE-BLOCK replaced). Call evaluate_solution to score it."

    def _eval_once(self) -> EvalOutcome:
        return evaluate_program(
            self.task, self.current_program,
            timeout_s=self.eval_timeout_s,
            python_exe=self.python_exe or __import__("sys").executable,
        )

    def evaluate(self) -> EvalOutcome:
        # EVAL_REPEATS (noisy tasks, e.g. ahc039 whose randomized solver has
        # eval std ~= the SOTA gap): average R runs of the UNCHANGED official
        # evaluator and count as ONE budget eval. This de-noises the reward /
        # best-tracking so a single lucky-high read can't win — the objective is
        # untouched (still the official metric), we just estimate it with more
        # samples. R=1 (default) is the original single-eval behavior.
        import os as _os
        reps = max(1, int(_os.environ.get("EVAL_REPEATS", "1") or "1"))
        out = self._eval_once()
        if reps > 1 and out.error is None:
            runs = [out] + [self._eval_once() for _ in range(reps - 1)]
            valid_runs = [r for r in runs if r.error is None]
            if valid_runs:
                mean_score = sum(r.combined_score for r in valid_runs) / len(valid_runs)
                mean_val = sum(r.validity for r in valid_runs) / len(valid_runs)
                wall = sum(r.wall_s for r in runs)
                out = EvalOutcome(combined_score=mean_score, validity=mean_val,
                                  error=None, wall_s=wall,
                                  metrics=valid_runs[-1].metrics)
        self.ledger.evaluator_calls += 1
        self.ledger.sandbox_seconds += out.wall_s
        # Anti-reward-hacking: an INVALID outcome can never become best, no
        # matter how high combined_score is. Some task evaluators award a huge
        # score to a solution that violates their own constraints (e.g. txn
        # returning 32258 for an illegal schedule with validity=0.0); tracking
        # best on combined_score alone would let M0 mine that defect and feed a
        # bogus positive reward into GRPO. `out.valid` = (error is None and
        # validity >= 1.0); every legitimate winner across all tasks satisfies
        # it, so this rejects only the exploit.
        is_best = out.valid and out.combined_score > self.best_score
        if (not out.valid) and out.combined_score > self.best_score:
            print(f"[session] rejected invalid high score "
                  f"(combined={out.combined_score:g} validity={out.validity:g} "
                  f"err={out.error}) — not counted as best", flush=True)
        if is_best:
            self.best_score = out.combined_score
            self.best_program = self.current_program
            self.best_metrics = out.metrics
            self._write_checkpoint()
        self.history.append(StepRecord(
            step=len(self.history), kind=self._pending_edit_mode if self._pending_edit_mode != "seed" else "seed",
            edit_mode=self._pending_edit_mode, edit_note=self._pending_edit_note,
            combined_score=out.combined_score, validity=out.validity, error=out.error,
            wall_s=out.wall_s, is_new_best=is_best,
        ))
        self._pending_edit_mode, self._pending_edit_note = "edit_eval", ""
        return out

    def probe(self, subsample: int = 2000) -> EvalOutcome:
        """Cheap approximate evaluation on subsampled data (first N CSV rows).

        Guidance only: does NOT count against the evaluation budget and does
        NOT update best-so-far (probe scores are not comparable to full ones).
        """
        self.ledger.probe_calls += 1
        out = evaluate_program(
            self.task, self.current_program,
            timeout_s=min(self.eval_timeout_s or 120.0, 120.0),
            python_exe=self.python_exe or __import__("sys").executable,
            subsample=subsample,
        )
        self.history.append(StepRecord(
            step=len(self.history), kind="probe",
            edit_mode="probe", edit_note=f"subsample={subsample}",
            combined_score=out.combined_score, validity=out.validity,
            error=out.error, wall_s=out.wall_s, is_new_best=False,
        ))
        return out

    def history_note(self, note: str) -> None:
        """Record a free-text note from a custom tool (audit trail only)."""
        self.history.append(StepRecord(
            step=len(self.history), kind="note", edit_mode="note",
            edit_note=str(note)[:400], combined_score=self.best_score,
            validity=1.0, error=None, wall_s=0.0, is_new_best=False))

    # -- generated tool/skill audit --------------------------------------- #
    def register_tool(self, name: str, source: str) -> None:
        row = self.tool_audit.setdefault(str(name), {
            "source": str(source), "mounts": 0, "invocations": 0,
            "completed": 0, "errors": 0, "last_error": None,
        })
        row["source"] = str(source)
        row["mounts"] += 1

    def record_tool_event(self, name: str, event: str, error: Optional[str] = None) -> None:
        row = self.tool_audit.setdefault(str(name), {
            "source": None, "mounts": 0, "invocations": 0,
            "completed": 0, "errors": 0, "last_error": None,
        })
        if event == "invoked":
            row["invocations"] += 1
        elif event == "completed":
            row["completed"] += 1
        elif event == "error":
            row["errors"] += 1
            row["last_error"] = str(error or "unknown custom-tool error")[:400]
        else:
            raise ValueError(f"unknown tool event: {event}")

    def register_skill(
        self, name: str, source: str, *, required_for_credit: bool = False,
    ) -> None:
        row = self.skill_audit.setdefault(str(name), {
            "source": str(source), "mounts": 0, "loads": 0,
            "loads_before_first_edit": 0, "late_loads": 0,
            "tool_loads": 0, "runtime_injections": 0,
            "required_for_execution": True,
            "required_for_credit": False,
        })
        row["source"] = str(source)
        row["mounts"] += 1
        row["required_for_credit"] = bool(required_for_credit)

    def record_skill_load(
        self, name: str, *, before_first_edit: bool,
        delivery_mode: str = "tool",
    ) -> None:
        if name in self.skill_audit:
            row = self.skill_audit[name]
            if delivery_mode == "tool":
                # ``loads`` retains its historical meaning: successful
                # executor-issued LoadSkill calls.  Runtime delivery is kept
                # separate so an audit never claims the executor chose a skill.
                row["loads"] += 1
                row["tool_loads"] += 1
            elif delivery_mode == "runtime_injection":
                row["runtime_injections"] += 1
            else:
                raise ValueError(f"unknown skill delivery mode: {delivery_mode}")
            if before_first_edit:
                row["loads_before_first_edit"] += 1
            else:
                row["late_loads"] += 1

    def skill_participation_issues(self, expected: List[str]) -> List[str]:
        """Generated skills must influence the search before its first edit.

        A mounted task-specific skill is part of the H1 proposal being scored.
        Merely packaging it is not enactment: either trusted runtime injection
        or a successful ``LoadSkill`` call must deliver the complete playbook
        before the executor starts editing the candidate program.
        """

        issues: List[str] = []
        for name in expected:
            row = self.skill_audit.get(name)
            if row is None or int(row.get("mounts", 0)) < 1:
                issues.append(f"{name}: not mounted")
                continue
            if int(row.get("loads_before_first_edit", 0)) < 1:
                loads = int(row.get("loads", 0))
                if loads:
                    issues.append(
                        f"{name}: loaded only after program editing began"
                    )
                else:
                    issues.append(f"{name}: mounted but never loaded")
        return issues

    def custom_tool_scratch(self) -> Path:
        if self._custom_scratch is None:
            safe_task = "".join(c if c.isalnum() else "_" for c in self.task.task_id)[-48:]
            self._custom_scratch = tempfile.mkdtemp(prefix=f"sah_tool_{safe_task}_")
        return Path(self._custom_scratch)

    # -- generated-middleware participation audit ------------------------- #
    def register_middleware(self, name: str, hook: str) -> None:
        row = self.middleware_audit.setdefault(str(name), {
            "hook": str(hook), "mounts": 0, "invocations": 0,
            "fires": 0, "errors": 0, "last_iteration": None,
            "last_error": None,
        })
        row["hook"] = str(hook)
        row["mounts"] += 1

    def record_middleware_event(
        self, name: str, event: str, *, iteration: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        row = self.middleware_audit.setdefault(str(name), {
            "hook": "unknown", "mounts": 0, "invocations": 0,
            "fires": 0, "errors": 0, "last_iteration": None,
            "last_error": None,
        })
        if event == "invoked":
            row["invocations"] += 1
        elif event == "fired":
            row["fires"] += 1
        elif event == "error":
            row["errors"] += 1
            row["last_error"] = str(error or "unknown middleware error")[:400]
        else:
            raise ValueError(f"unknown middleware event: {event}")
        if iteration is not None:
            row["last_iteration"] = int(iteration)

    # -- middleware tool gate (mechanical enforcement) -------------------- #
    #
    # A generated before_model hook may request that the executor's next tool
    # call come from a named subset.  The hook itself stays a pure function;
    # the trusted wrapper applies the effect here.  Enforcement happens in the
    # tool layer: a disallowed call is refused with a structured message and
    # consumes no budget.  ``finish`` is never gated, and repeated refusals
    # auto-lift the gate so a confused executor cannot be hard-locked.
    GATEABLE_TOOLS = ("probe_solution", "edit_solution", "evaluate_solution")
    GATE_AUTO_LIFT_AFTER = 2

    def _gate_stats(self, name: str) -> Dict[str, int]:
        row = self.middleware_audit.setdefault(str(name), {
            "hook": "unknown", "mounts": 0, "invocations": 0,
            "fires": 0, "errors": 0, "last_iteration": None,
            "last_error": None,
        })
        return row.setdefault("gate", {
            "enforced": 0, "satisfied": 0, "refused": 0, "auto_lifted": 0,
        })

    def request_tool_gate(self, name: str, require_tools: Any) -> None:
        if isinstance(require_tools, str):
            require_tools = [require_tools]
        try:
            tools = tuple(dict.fromkeys(str(t) for t in require_tools))
        except TypeError:
            raise ValueError("require_tools must be a list of tool names")
        if not tools or any(t not in self.GATEABLE_TOOLS for t in tools):
            raise ValueError(
                "require_tools must be a non-empty subset of "
                f"{list(self.GATEABLE_TOOLS)}; got {list(tools)!r}"
            )
        self.tool_gate = {"require": tools, "set_by": str(name), "refusals": 0}
        self._gate_stats(name)["enforced"] += 1

    def check_tool_gate(self, tool_name: str) -> Optional[str]:
        gate = self.tool_gate
        if not gate or tool_name == "finish":
            return None
        name = gate["set_by"]
        if tool_name in gate["require"]:
            self.tool_gate = None
            self._gate_stats(name)["satisfied"] += 1
            return None
        gate["refusals"] += 1
        self._gate_stats(name)["refused"] += 1
        required = " or ".join(gate["require"])
        if gate["refusals"] >= self.GATE_AUTO_LIFT_AFTER:
            self.tool_gate = None
            self._gate_stats(name)["auto_lifted"] += 1
            self.history_note(
                f"[tool-gate:{name}] auto-lifted after repeated refusals"
            )
        return (
            f"GATED by middleware {name!r}: call {required} first. "
            "This call consumed no budget."
        )

    def middleware_participation_issues(self, expected: List[str]) -> List[str]:
        issues: List[str] = []
        for name in expected:
            row = self.middleware_audit.get(name)
            if row is None or int(row.get("mounts", 0)) < 1:
                issues.append(f"{name}: not mounted")
                continue
            if int(row.get("invocations", 0)) < 1:
                issues.append(f"{name}: mounted but never invoked")
            if int(row.get("errors", 0)) > 0:
                issues.append(
                    f"{name}: {row['errors']} execution error(s); "
                    f"last={row.get('last_error')!r}"
                )
        return issues

    def seed_baseline(self) -> EvalOutcome:
        """Evaluate the seed once to initialise best-so-far (not charged to budget)."""
        self._pending_edit_mode, self._pending_edit_note = "seed", "seed program"
        out = self.evaluate()
        self.ledger.evaluator_calls -= 1  # seed eval is the harness baseline, not agent budget
        return out

    def seed_baseline_from_outcome(self, out: EvalOutcome) -> EvalOutcome:
        """Initialize the seed from an already audited evaluator outcome.

        Outer-loop integrations may already have evaluated the selected parent.
        Reusing that hash-bound outcome avoids charging or executing the same
        evaluator twice while preserving the exact state and history that
        :meth:`seed_baseline` would establish.
        """

        self._pending_edit_mode, self._pending_edit_note = "seed", "cached seed program"
        is_best = out.valid
        if is_best:
            self.best_score = out.combined_score
            self.best_program = self.current_program
            self.best_metrics = dict(out.metrics)
            self._write_checkpoint()
        self.history.append(StepRecord(
            step=len(self.history), kind="seed", edit_mode="seed",
            edit_note="reused hash-bound outer-loop evaluation",
            combined_score=out.combined_score, validity=out.validity,
            error=out.error, wall_s=0.0, is_new_best=is_best,
        ))
        self._pending_edit_mode, self._pending_edit_note = "edit_eval", ""
        return out

    def _write_checkpoint(self) -> None:
        """Persist best-so-far after each new best, so a wall-kill never loses it."""
        if not self.checkpoint_path:
            return
        try:
            import json
            from pathlib import Path
            p = Path(self.checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps({
                "task_id": self.task.task_id, "source": self.task.source,
                "best_score": self.best_score, "best_metrics": self.best_metrics,
                "evaluations": self.ledger.evaluator_calls,
                "best_program": self.best_program,
            }))
            tmp.replace(p)  # atomic
        except Exception:
            pass  # checkpointing must never break a run

    # -- serialization ----------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "source": self.task.source,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "ledger": asdict(self.ledger),
            "steps": [asdict(s) for s in self.history],
            "middleware_audit": self.middleware_audit,
            "tool_audit": self.tool_audit,
            "skill_audit": self.skill_audit,
        }


# --------------------------------------------------------------------------- #
# contextvar bridge so module-level NexAU tools reach the active session
# --------------------------------------------------------------------------- #
_CURRENT: "contextvars.ContextVar[Optional[InnerSession]]" = contextvars.ContextVar(
    "inner_session", default=None
)


def get_session() -> InnerSession:
    s = _CURRENT.get()
    if s is None:
        raise RuntimeError("no active InnerSession (tool called outside session_scope)")
    return s


@contextmanager
def session_scope(session: InnerSession):
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)

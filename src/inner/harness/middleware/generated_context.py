"""Stable, read-only context for proposer-generated middleware hooks.

Generated middleware must not depend on NexAU's internal hook-input shape.  The
outer loop is allowed to synthesize a small ``def before_model(hook_input)``
function, but the argument passed to that function is the mapping produced
here, not the framework object itself.

The tracker also exposes a coarse structural-family signal.  It hashes a
Python AST after erasing literal values, so parameter-only edits remain in the
same family while structural rewrites normally start a new one.  This is an
operational search signal, not a claim about mathematical equivalence.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


def _normalised_structure(program: str) -> str:
    """Return a stable structural digest, or ``invalid-python`` on syntax errors."""

    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, TypeError):
        return "invalid-python"

    class _EraseLiterals(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant):  # noqa: N802 - ast API
            marker = type(node.value).__name__
            return ast.copy_location(ast.Constant(value=marker), node)

    tree = _EraseLiterals().visit(tree)
    ast.fix_missing_locations(tree)
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _latest_non_note(history) -> Optional[Any]:
    for row in reversed(history or []):
        if getattr(row, "kind", None) != "note":
            return row
    return None


def _probes_since_last_eval(history) -> int:
    probes = 0
    for row in reversed(history or []):
        kind = getattr(row, "kind", None)
        if kind == "note":
            continue
        if kind == "probe":
            probes += 1
            continue
        break
    return probes


@dataclass
class GeneratedHookTracker:
    """Build the documented mapping passed to generated hook functions."""

    _last_structure: Optional[str] = None
    _structure_streak: int = 0
    _structures_explored: Set[str] = field(default_factory=set)
    _seen_evals: int = 0
    _last_best: Optional[float] = None
    _stalled_evals: int = 0

    def _update_structure(self, program: str) -> str:
        signature = _normalised_structure(program)
        if signature == self._last_structure:
            self._structure_streak += 1
        else:
            self._last_structure = signature
            self._structure_streak = 1
            self._structures_explored.add(signature)
        return signature

    def _update_stall(self, evals: int, best: float) -> None:
        if evals < self._seen_evals:
            self._seen_evals = 0
            self._last_best = None
            self._stalled_evals = 0
        if evals > self._seen_evals:
            newly_seen = evals - self._seen_evals
            if self._last_best is not None and best <= self._last_best:
                self._stalled_evals += newly_seen
            else:
                self._stalled_evals = 0
            self._seen_evals = evals
        self._last_best = best

    def snapshot(self, framework_input: Any, session: Any) -> Dict[str, Any]:
        """Return the stable hook context documented to the proposer."""

        iteration = int(getattr(framework_input, "current_iteration", 0) or 0)
        ledger = session.ledger
        best = float(session.best_score)
        evals = int(ledger.evaluator_calls)
        probes = int(ledger.probe_calls)
        edits = int(ledger.edit_calls)
        self._update_stall(evals, best)
        structure = self._update_structure(str(session.current_program))
        last = _latest_non_note(session.history)

        state: Dict[str, Any] = {
            "iteration": iteration,
            "current_iteration": iteration,
            "best_so_far": best,
            "evaluator_calls": evals,
            "evals_done": evals,
            "evaluations_remaining": int(ledger.evaluator_budget_left()),
            "evals_remaining": int(ledger.evaluator_budget_left()),
            "probe_calls": probes,
            "probe_attempts": probes,
            "probes_remaining": max(0, int(ledger.max_probe_calls) - probes),
            "probe_remaining": max(0, int(ledger.max_probe_calls) - probes),
            "probe_available": probes < int(ledger.max_probe_calls),
            "probes_since_eval": _probes_since_last_eval(session.history),
            "edit_calls": edits,
            "stalled_evals": self._stalled_evals,
            # Backward-compatible names.  "family" means normalised Python AST
            # structure here; it is deliberately coarser than exact source text.
            "active_tool_gate": (
                dict(session.tool_gate)
                if getattr(session, "tool_gate", None) else None
            ),
            "family_streak": self._structure_streak,
            "families_explored": sorted(self._structures_explored),
            "last_family": structure,
            "structure_streak": self._structure_streak,
            "structures_explored": sorted(self._structures_explored),
            "current_program_valid_syntax": structure != "invalid-python",
            "last_step_kind": getattr(last, "kind", None),
            "last_error": getattr(last, "error", None),
            "last_validity": getattr(last, "validity", None),
            "last_score": getattr(last, "combined_score", None),
        }
        # Direct aliases support simple hooks; ``state`` supports grouped code
        # such as ``state = hook_input.get("state", {})``.
        return {**state, "state": dict(state)}

"""Submit-reminder middleware for H1's file-native H2 editing loop.

From ``remind_from_iteration`` on, if the proposer has not yet submitted a
spec, append a bounded framework message urging it to validate + submit before
the iteration cap — an un-submitted run is an invalid candidate (minimum
reward). Fails open: any error resolves to no-changes.
"""
from __future__ import annotations

from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock

from outer.harness.tools.runtime import get_session


class SubmitReminderMiddleware(Middleware):
    """Submit reminder + mechanical bail-out escalation (BAILOUT-001).

    The first two-round smoke of the gated pipeline showed the dominant
    no-submission mode: a slot writes an ambitious generated tool, gets a
    real validator error, fixes ONE issue per turn, and repair-loops until
    the iteration cap without ever abandoning the doomed component.  From
    ``bailout_from_left`` remaining turns (after >= ``bailout_after_failures``
    failed validations) the reminder switches to an explicit
    delete-and-simplify recipe naming the exact files; from
    ``freeze_from_left`` remaining turns further generated-component edits
    are refused mechanically (deletes, prompt/agent edits, validate, and
    submit stay available, so the recipe can always complete).
    """

    def __init__(self, *, remind_from_iteration: int = 6, max_iterations: int = 8,
                 bailout_from_left: int = 10, bailout_after_failures: int = 2,
                 freeze_from_left: int = 6) -> None:
        import os
        # PRELOAD-001: the runtime may shrink the agent's iteration budget
        # (SAH_H1_MAX_ITERS) below the fixed agent.yaml params; the reminder's
        # turns-left arithmetic must follow the ACTUAL cap, and thresholds
        # scale down proportionally for short runs.
        override = os.environ.get("SAH_H1_MAX_ITERS")
        if override and override.isdigit() and int(override) < int(max_iterations):
            actual = int(override)
            scale = actual / float(max_iterations)
            remind_from_iteration = max(2, round(remind_from_iteration * scale))
            bailout_from_left = max(3, round(bailout_from_left * scale))
            freeze_from_left = max(2, round(freeze_from_left * scale))
            max_iterations = actual
        self.remind_from_iteration = int(remind_from_iteration)
        self.max_iterations = int(max_iterations)
        self.bailout_from_left = int(bailout_from_left)
        self.bailout_after_failures = int(bailout_after_failures)
        self.freeze_from_left = int(freeze_from_left)
        self._current_iteration = 0
        self._no_tool_nudges = 0

    # Passes the static gate and the mock self-test verbatim; small enough to
    # fit the late-turn freeze.  Kept as data so recipes and tests share it.
    MINIMAL_TOOL_TEMPLATE = (
        "def run(ctx, args):\n"
        "    best = ctx.best_score()\n"
        "    return {\"best_score\": best, "
        "\"budget_left\": ctx.budget_left()}\n"
    )
    # Generated-component writes below this size stay allowed for forced
    # slots during the freeze: shrinking to a minimal tool must remain
    # possible while ambitious rewrites stay blocked.
    FREEZE_MAX_WRITE_CHARS = 2400

    def _forced_tool_name(self, session) -> str:
        slot = getattr(session, "slot_index", None)
        return f"quick_probe_k{slot}" if slot is not None else "quick_probe"

    def _forced_checklist(self, session) -> str:
        """Forced-slot guidance: the starter tool is already in place.

        BAILOUT-005 pre-seeds every forced slot's draft with a mounted,
        valid minimal tool, so the reliable floor is always two calls away.
        Earlier revisions taught a from-scratch six-step build; the smokes
        showed a weak proposer cannot execute it reliably in 32 turns.
        """
        name = self._forced_tool_name(session)
        return (
            "This candidate MUST keep one new mounted generated tool. The "
            f"starter tool {name} is ALREADY written (custom_tools/{name}.py)"
            ", mounted in agent.yaml, and declared in prompt.md — the "
            "workspace validates as-is. Either improve/replace its "
            "implementation now (keep the same name, schema, and mount; a "
            "small whitelist-safe implementation), or keep the starter and "
            "make your other improvements. Then validate_harness and "
            "submit_harness. Never delete your last generated tool; never "
            "unmount it."
        )

    def _bailout_recipe(self, session) -> str:
        targets = session.generated_component_targets()
        if getattr(session, "require_new_tool", False):
            main = next(
                (path for path in targets
                 if path.startswith("custom_tools/")),
                None,
            )
            replace = (
                f"REPLACE {main} entirely (write_harness_file) with the "
                "minimal valid tool and shrink its schema to empty args"
                if main else
                "write the minimal valid tool now"
            )
            return (
                "STOP repairing the ambitious generated tool — it has failed "
                f"validation {session.failed_validations} times and there is "
                "no budget left to fix it. This slot MUST keep one new "
                "mounted generated tool (forced-tool contract), so do NOT "
                f"delete it. Instead {replace}. Exact sequence:\n"
                + self._forced_checklist(session)
                + "\nA trivial valid tool beats an unsubmitted run."
            )
        deletes = " ".join(
            f'delete_harness_file(path="{path}")' for path in targets
        ) or "(no generated files to delete)"
        return (
            "STOP repairing the failing generated component — it has failed "
            f"validation {session.failed_validations} times and there is not "
            "enough budget left to fix it. Recovery recipe, in order: "
            f"1) {deletes}; 2) remove its mount from agent.yaml and any "
            "mention from prompt.md (edit_harness_file); 3) make one minimal "
            "reliable change instead — edit_harness_file(path=\"prompt.md\", "
            "append_text=\"a concise task-specific strategy\"); "
            "4) validate_harness; 5) submit_harness. A submitted small change "
            "beats an unsubmitted ambitious one."
        )

    def _stuck_on_component(self, session) -> bool:
        return (
            session.validated_revision != session.workspace_revision
            and session.failed_validations >= self.bailout_after_failures
            and bool(session.generated_component_targets())
        )

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        try:
            session = get_session()
            iteration = int(hook_input.current_iteration)
            self._current_iteration = iteration
        except Exception:
            return HookResult.no_changes()
        if session.submitted or iteration < self.remind_from_iteration:
            return HookResult.no_changes()
        left = max(0, self.max_iterations - iteration)
        forced = getattr(session, "require_new_tool", False)
        if session.workspace_revision == 0:
            action = (
                self._forced_checklist(session) if forced else (
                    "No H2 file has changed. Do not call edit_solution, "
                    "evaluate_solution, probe_solution, finish, or LoadSkill; "
                    "those are future-executor tools and are unavailable to "
                    "H1. Use edit_harness_file now. The reliable action is "
                    '`edit_harness_file(path="prompt.md", append_text="a '
                    'concise task-specific search strategy")`; never create '
                    "solution.py."
                )
            )
        elif left <= 3:
            # DEADLINE-001: 7/11 invalid slots in the fix3 taxonomy ran out
            # of budget mid-build without ever calling submit.  In the last
            # three turns nothing matters except banking the work.
            action = (
                f"DEADLINE: only {left} tool calls remain. STOP building. "
                "Call validate_harness NOW; if it reports one small error, "
                "fix only that; then submit_harness immediately. An "
                "unsubmitted workspace scores zero for this candidate."
            )
        elif left <= self.bailout_from_left and self._stuck_on_component(session):
            action = self._bailout_recipe(session)
        elif getattr(session, "component_only", False) \
                and not session.generated_component_targets():
            # EXPLORE-001b: keep the component-only slot from drifting into a
            # prompt-only submission while there is still budget to build.
            action = (
                "This candidate is a COMPONENT-ONLY slot: prompt.md is "
                "inherited and only one short declaration line may be "
                "appended, so a prompt-only submission will be refused. "
                "Write the component now — custom_tools/<name>.py + "
                "tools/<name>.tool.yaml + agent.yaml mount (tool), or "
                "skills/<name>/SKILL.md + agent.yaml skills mount (skill) — "
                "append one declaration line, then validate_harness and "
                "submit_harness."
            )
        elif forced and session.validated_revision != session.workspace_revision \
                and not session.generated_component_targets():
            # Edits exist but no generated tool has been written yet — a
            # forced slot heading toward a contract violation.  Redirect
            # before the budget is gone.
            action = self._forced_checklist(session)
        elif session.validated_revision != session.workspace_revision:
            action = (
                "The H2 has unvalidated edits. Call validate_harness now and "
                "repair only the errors it reports."
            )
        else:
            action = "The current H2 is validated. Call submit_harness now."
        text = (
            f"Only ~{left} turn(s) remain and no H2 has been submitted. "
            f"{action} An unsubmitted run scores the minimum reward."
        )
        return HookResult.with_modifications(messages=[
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)]),
        ])

    # A proposer that answers without any tool call silently ends the NexAU
    # loop as an invalid no-submission slot (observed: a forced slot wrote a
    # 7 KB tool on turn 7, then replied in prose and died).  Retry the model
    # call once with an explicit corrective message, a few times per run.
    # Raised after the r1 early-stop collapse (trained phi cloned the short
    # preloaded-trajectory SHAPE but truncated before submit: 0/8
    # self-submissions, trajectories ending right after an edit).  Every
    # prose-only response gets redirected until the cap; env-tunable.
    import os as _os
    MAX_NO_TOOL_NUDGES = int(_os.environ.get("SAH_H1_NO_TOOL_NUDGES", "8"))

    def wrap_model_call(self, params, call_next):
        response = call_next(params)
        try:
            session = get_session()
        except Exception:
            return response
        if (
            session.submitted
            or self._no_tool_nudges >= self.MAX_NO_TOOL_NUDGES
            or response is None
            or getattr(response, "tool_calls", None)
        ):
            return response
        self._no_tool_nudges += 1
        params.messages = [
            *params.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=(
                "You stopped without calling a tool — an H1 run only ends "
                "via submit_harness, and an unsubmitted run scores the "
                "minimum reward. Call a tool now: if your work is complete, "
                "validate_harness then submit_harness; otherwise continue "
                "with the next file operation."
            ))]),
        ]
        return call_next(params)

    def wrap_tool_call(self, params, call_next):
        """Mechanically preserve enough turns for edit -> validate -> submit."""

        try:
            session = get_session()
            tool_name = str(params.tool_name)
            tool_input = dict(params.parameters)
        except Exception:
            return call_next(params)

        if (
            session.validated_revision == session.workspace_revision
            and session.workspace_revision > 0
            and tool_name != "submit_harness"
        ):
            return (
                "REFUSED: the current H2 revision is already valid. Call "
                "submit_harness now; any other action risks an unsubmitted run."
            )

        if tool_name in {"write_harness_file", "edit_harness_file"}:
            path = str(tool_input.get("path", ""))
            supplied_text = "\n".join(
                str(tool_input.get(key, ""))
                for key in ("content", "new_text", "append_text")
            )
            if path.endswith("solution.py") or "EVOLVE-BLOCK" in supplied_text:
                return (
                    "REFUSED: H1 must not copy or implement the seed solution or "
                    "an EVOLVE-BLOCK in H2. Append concise task-specific search "
                    "guidance to the already-inspected prompt.md, then validate "
                    "and submit."
                )

        if (
            self._current_iteration >= self.remind_from_iteration
            and session.workspace_revision == 0
            and tool_name == "harness_shell"
        ):
            return (
                "REFUSED: the inspection deadline has passed and no H2 file has "
                "changed. Use edit_harness_file on the already-inspected "
                "prompt.md with append_text now."
            )

        # BAILOUT-001 freeze: with the budget nearly gone and a generated
        # component still failing validation, refuse to keep growing it.
        # Deletes, agent.yaml/prompt.md edits, validate, and submit remain
        # available so the recovery recipe can always complete.  Forced-tool
        # slots must keep a mounted generated tool, so for them only LARGE
        # writes are refused — shrinking to the minimal valid tool stays open.
        left = max(0, self.max_iterations - self._current_iteration)
        if (
            left <= self.freeze_from_left
            and self._stuck_on_component(session)
            and tool_name in {"write_harness_file", "edit_harness_file"}
        ):
            path = str(tool_input.get("path", ""))
            if (path.startswith("custom_tools/")
                    or (path.startswith("tools/")
                        and path.endswith(".tool.yaml"))
                    or path.startswith("middlewares/")
                    or path.startswith("skills/")):
                supplied = sum(
                    len(str(tool_input.get(key, "")))
                    for key in ("content", "new_text", "append_text")
                )
                if not getattr(session, "require_new_tool", False):
                    return "REFUSED: " + self._bailout_recipe(session)
                if supplied > self.FREEZE_MAX_WRITE_CHARS:
                    return "REFUSED (write too large): " + \
                        self._bailout_recipe(session)
        return call_next(params)

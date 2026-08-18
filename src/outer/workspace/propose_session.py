"""Per-run proposer session and candidate-isolated H2 filesystem.

Mirrors inner/session.py: the propose driver sets the active session in a
contextvar for the duration of one H1 agent run; the tool bindings and the
submit-reminder middleware resolve it via get_session(). Thread-safe: each
proposer thread runs in its own context.
"""
from __future__ import annotations

import contextvars
import shlex
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from outer.genome import harness_spec as hs
from outer.workspace import h2_workspace as h2ws


@dataclass
class ProposeSession:
    base_spec: Dict[str, Any]
    draft_dir: Optional[Path] = None
    require_new_tool: bool = False
    slot_index: Optional[int] = None
    validations: int = 0
    submitted: bool = False
    raw_submission: str = ""
    partial_spec: Optional[Dict[str, Any]] = None
    effective: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    file_events: List[Dict[str, str]] = field(default_factory=list)
    agent_yaml_inspected: bool = False
    inspected_files: Set[str] = field(default_factory=set)
    workspace_revision: int = 0
    validated_revision: Optional[int] = None
    component_audit: List[Dict[str, Any]] = field(default_factory=list)
    schema_guard_events: List[Dict[str, Any]] = field(default_factory=list)
    failed_validations: int = 0
    # NOVELTY-001: component signatures proven causally futile in previous
    # rounds; a submission repeating one is refused (twice) with redirect
    # guidance, then accepted with a flag so the loop can never deadlock.
    futile_signatures: Set[str] = field(default_factory=set)
    novelty_refusals: int = 0
    novelty_flag: Optional[str] = None
    # EXPLORE-001 (v2.5) component-only contract: the slot inherits the
    # incumbent prompt; its only allowed prompt change is a short appended
    # declaration line, so the paired delta isolates the component itself.
    component_only: bool = False
    component_kind: str = ""      # tool | skill | middleware | params
    _component_prompt_appended: int = 0
    _COMPONENT_PROMPT_APPEND_CAP = 300

    _STARTER_BODY = (
        "def run(ctx, args):\n"
        "    best = ctx.best_score()\n"
        "    return {\"best_score\": best, \"budget_left\": ctx.budget_left()}\n"
    )

    def _unchanged_starters(self) -> Set[str]:
        """Names of generated tools still byte-identical to the placeholder.

        Identity is decided by CONTENT, not by name: a starter that the
        proposer genuinely rewrote is a real component even though it kept
        the seeded file name, and a freshly written tool that happens to be
        the placeholder is not.
        """
        if self.draft_dir is None:
            return set()
        d = Path(self.draft_dir) / "custom_tools"
        if not d.is_dir():
            return set()
        bodies = {self._STARTER_BODY}
        try:
            from outer.proposing.component_scaffold import placeholder_bodies
            bodies |= placeholder_bodies()
        except Exception:
            pass
        names = {f.stem for f in d.glob("*.py") if f.read_text() in bodies}
        mw = Path(self.draft_dir) / "middlewares"
        if mw.is_dir():
            for f in mw.glob("*.py"):
                text = f.read_text()
                if any(b.strip() in text for b in bodies if "before_model" in b):
                    names.add(f.stem)
        return names

    def _component_only_refusal(self, path: str, *, whole_file: bool,
                                old_text: str = "",
                                append_text: str = "") -> Optional[str]:
        if not self.component_only or not str(path).endswith("prompt.md"):
            return None
        if whole_file or old_text:
            return (
                "REFUSED (component-only slot): the incumbent prompt.md is "
                "inherited for this candidate; rewriting or replacing prompt "
                "text is not allowed. Append ONE short declaration line for "
                "your component (edit_harness_file with append_text), build "
                "the component itself, then validate and submit."
            )
        budget = self._COMPONENT_PROMPT_APPEND_CAP
        if len(append_text) + self._component_prompt_appended > budget:
            # SCAFFOLD-003: measured failure — one slot repeated an over-cap
            # prompt append SIXTEEN times and never touched its seeded
            # component file.  The refusal now says exactly what to do next,
            # and repeats escalate instead of recycling the same words.
            self._prompt_cap_refusals += 1
            seeded = sorted(self._unchanged_starters())
            targets = self.generated_component_targets()
            if targets:
                target = targets[0]
            elif seeded:
                d = Path(self.draft_dir) if self.draft_dir else None
                mw = d and (d / "middlewares" / f"{seeded[0]}.py").is_file()
                target = (f"middlewares/{seeded[0]}.py" if mw
                          else f"custom_tools/{seeded[0]}.py")
            else:
                # skill slots have no seeded scaffold — name the real target
                target = "skills/<slug>/SKILL.md (write_harness_file)"
            if self._prompt_cap_refusals >= 2:
                seeded_claim = (
                    "the declaration line for your component was ALREADY "
                    "seeded into it before this session and no further "
                    "prompt text is needed" if seeded else
                    "one short declaration line is the ONLY prompt text "
                    "allowed here")
                return (
                    f"REFUSED AGAIN ({self._prompt_cap_refusals}x): STOP "
                    f"editing prompt.md — {seeded_claim}. Your ONLY "
                    f"remaining job: build the component in {target}, then "
                    "validate_harness and submit_harness."
                )
            return (
                f"REFUSED (component-only slot): prompt appends are capped at "
                f"{budget} chars total ({self._component_prompt_appended} "
                "used) and the declaration line is ALREADY in prompt.md "
                "(seeded). Do not add guidance text; put substance in "
                f"{target}, then validate_harness and submit_harness."
            )
        return None
    _prompt_cap_refusals: int = field(default=0, repr=False)
    _validated_check: Optional[h2ws.WorkspaceCheck] = field(default=None, repr=False)

    def generated_component_targets(self) -> List[str]:
        """Paths of generated-component files this session wrote or edited.

        The bail-out escalation (BAILOUT-001) uses this to name the exact
        files a stuck proposer should delete before falling back to a
        minimal valid change.
        """
        targets: List[str] = []
        for event in self.file_events:
            path = str(event.get("target", ""))
            if (path.startswith("custom_tools/")
                    or (path.startswith("tools/")
                        and path.endswith(".tool.yaml"))
                    or path.startswith("middlewares/")
                    or path.startswith("skills/")) \
                    and event.get("operation") in ("write", "edit") \
                    and path not in targets:
                targets.append(path)
        return targets

    def mark_preloaded(self, relative_paths) -> None:
        """PRELOAD-001: files rendered into the proposer's initial context
        count as inspected — the read-before-edit protocol's purpose (no
        blind edits) is satisfied by construction."""
        for relative in relative_paths:
            self.inspected_files.add(str(relative))
            if str(relative) == "agent.yaml":
                self.agent_yaml_inspected = True

    def record_schema_guard(self, event: Dict[str, Any]) -> None:
        """Audit trail for the tool schema guard (TOOLSCHEMA-001)."""
        self.schema_guard_events.append(dict(event))

    def schema_guard_summary(self) -> Dict[str, Any]:
        """Compact per-slot summary for round.json attribution."""
        repaired = [e for e in self.schema_guard_events if e.get("action") == "repaired"]
        refused = [e for e in self.schema_guard_events if e.get("action") == "refused"]
        by_tool: Dict[str, int] = {}
        for event in refused:
            by_tool[event["tool"]] = by_tool.get(event["tool"], 0) + 1
        return {
            "repaired_calls": len(repaired),
            "refused_calls": len(refused),
            "escalated": any(e.get("escalated") for e in refused),
            "refusals_by_tool": by_tool,
        }

    def _contract_errors(self, effective: Dict[str, Any]) -> List[str]:
        if self.require_new_tool and not hs.added_generated_component_names(
            self.base_spec, effective, "new_tools"
        ):
            return [
                "forced-tool contract violation: this candidate must add a "
                "new mounted generated tool"
            ]
        return []

    def _check(self, spec_yaml: str):
        v = hs.parse_and_validate(spec_yaml)
        if not v.valid:
            return None, None, [], v.errors
        eff = hs.merge_with_base(v.spec, self.base_spec)
        differs, changed = hs.differs_from_base(eff, self.base_spec)
        if not differs:
            return v.spec, eff, [], ["spec is identical to the current harness (no-op)"]
        prompt_errors = hs.component_prompt_issues(eff)
        if prompt_errors:
            return v.spec, eff, changed, prompt_errors
        contract_errors = self._contract_errors(eff)
        if contract_errors:
            return v.spec, eff, changed, contract_errors
        return v.spec, eff, changed, []

    def validate(self, spec_yaml: str) -> str:
        self.validations += 1
        _, _, changed, errors = self._check(spec_yaml)
        if errors:
            return "INVALID:\n- " + "\n- ".join(errors)
        return ("VALID. This spec changes: " + ", ".join(changed) +
                ". You can refine further or submit_spec now.")

    def submit(self, spec_yaml: str) -> str:
        self.submitted = True
        self.raw_submission = spec_yaml
        spec, eff, changed, errors = self._check(spec_yaml)
        if errors:
            self.errors = errors
            return "SUBMITTED but INVALID (minimum reward):\n- " + "\n- ".join(errors)
        self.partial_spec, self.effective, self.changed_fields = spec, eff, changed
        return f"SUBMITTED. Candidate spec accepted; changes: {', '.join(changed)}."

    def _require_draft(self) -> Path:
        if self.draft_dir is None:
            raise RuntimeError("this proposer session has no H2 file workspace")
        return Path(self.draft_dir)

    def inspect_harness(self, command: str) -> str:
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        is_agent_cat = (
            len(argv) == 2
            and argv[0] == "cat"
            and Path(argv[1]).as_posix().lstrip("./") == "agent.yaml"
        )
        if not self.agent_yaml_inspected and not is_agent_cat:
            return (
                "ERROR: first inspect the actual mount graph with "
                '`harness_shell(command="cat agent.yaml")`.'
            )
        output = h2ws.inspect(self._require_draft(), command)
        if not output.startswith("ERROR:") and len(argv) == 2 and argv[0] == "cat":
            try:
                relative = h2ws.relative_path(
                    self._require_draft(), argv[1], must_exist=True
                )
                self.inspected_files.add(relative)
                if relative == "agent.yaml":
                    self.agent_yaml_inspected = True
            except ValueError:
                pass
        self.file_events.append({"operation": "inspect", "target": command})
        return output

    # PROMPTCAP-001: the genome caps system_prompt at 8000 chars, but the
    # validator only reports the breach AFTER the session has spent turns.
    # Refuse the write that would cross the cap, with the fix in the message
    # — a drift-prone proposer otherwise appends its way into an unfixable
    # state that repair budgets cannot trim back in time.
    _PROMPT_CAP = 10000

    def _prompt_cap_refusal(self, path: str, *, content: Optional[str] = None,
                            old_text: Optional[str] = None,
                            new_text: Optional[str] = None,
                            append_text: Optional[str] = None) -> Optional[str]:
        if path != "prompt.md" or self.draft_dir is None:
            return None
        target = Path(self.draft_dir) / "prompt.md"
        current = target.read_text() if target.is_file() else ""
        if content is not None:
            prospective = len(content)
        elif append_text is not None:
            prospective = len(current) + len(append_text) + 1
        elif old_text is not None and old_text in current:
            prospective = len(current) - len(old_text) + len(new_text or "")
        else:
            return None
        if prospective <= self._PROMPT_CAP:
            return None
        return (
            f"REFUSED: this edit would make prompt.md {prospective} chars — "
            f"over the {self._PROMPT_CAP}-char genome cap (current: "
            f"{len(current)}). Do NOT append more text. Instead REPLACE or "
            "TRIM an existing section with edit_harness_file(old_text=..., "
            "new_text=<shorter text>) so the total stays under the cap, then "
            "validate_harness."
        )

    def _require_read_before_existing_edit(self, path: str) -> Optional[str]:
        if not self.agent_yaml_inspected:
            return (
                "ERROR: first inspect the actual mount graph with "
                '`harness_shell(command="cat agent.yaml")`.'
            )
        try:
            relative = h2ws.relative_path(
                self._require_draft(), path, must_exist=False
            )
            exists = (self._require_draft() / relative).exists()
        except ValueError as exc:
            return f"ERROR: {exc}"
        if exists and relative not in self.inspected_files:
            return (
                f"ERROR: read existing file first with "
                f'`harness_shell(command="cat {relative}")`.'
            )
        return None

    def _record_mutation(self, output: str) -> None:
        if not output.startswith("ERROR:"):
            self.workspace_revision += 1
            self.validated_revision = None
            self._validated_check = None
            self.component_audit = []

    def write_harness_file(self, path: str, content: str) -> str:
        blocked = self._component_only_refusal(path, whole_file=True)
        if blocked:
            return blocked
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        blocked = self._prompt_cap_refusal(path, content=content)
        if blocked:
            return blocked
        output = h2ws.write_file(self._require_draft(), path, content)
        self._record_mutation(output)
        self.file_events.append({"operation": "write", "target": path,
                                 "result": output})
        return output

    def edit_harness_file(
        self,
        path: str,
        old_text: str = "",
        new_text: str = "",
        append_text: str = "",
    ) -> str:
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        blocked = self._component_only_refusal(
            path, whole_file=False, old_text=old_text,
            append_text=append_text,
        )
        if blocked:
            return blocked
        if self.component_only and str(path).endswith("prompt.md") \
                and append_text:
            self._component_prompt_appended += len(append_text)
        blocked = self._prompt_cap_refusal(
            path, old_text=old_text or None, new_text=new_text,
            append_text=append_text or None,
        )
        if blocked:
            return blocked
        output = h2ws.edit_file(
            self._require_draft(), path, old_text, new_text, append_text
        )
        self._record_mutation(output)
        self.file_events.append({"operation": "edit", "target": path,
                                 "result": output})
        return output

    def delete_harness_file(self, path: str) -> str:
        blocked = self._require_read_before_existing_edit(path)
        if blocked:
            return blocked
        output = h2ws.delete_file(self._require_draft(), path)
        self._record_mutation(output)
        self.file_events.append({"operation": "delete", "target": path,
                                 "result": output})
        return output

    def _check_workspace(self) -> h2ws.WorkspaceCheck:
        return h2ws.validate_workspace(self._require_draft(), self.base_spec)

    def validate_harness(self) -> str:
        self.validations += 1
        if not self.agent_yaml_inspected:
            return (
                "INVALID H2 WORKSPACE:\n- agent.yaml was not inspected; start "
                'with harness_shell(command="cat agent.yaml")'
            )
        if "prompt.md" not in self.inspected_files:
            return (
                "INVALID H2 WORKSPACE:\n- prompt.md was not inspected; read the "
                "executor system prompt before validating H2"
            )
        check = self._check_workspace()
        self.component_audit = list(check.component_audit)
        if not check.valid:
            self.failed_validations += 1
            return "INVALID H2 WORKSPACE:\n- " + "\n- ".join(check.errors)
        contract_errors = self._contract_errors(check.effective or {})
        if contract_errors:
            self.failed_validations += 1
            return "INVALID H2 WORKSPACE:\n- " + "\n- ".join(contract_errors)
        self.validated_revision = self.workspace_revision
        self._validated_check = check
        return (
            "VALID H2 WORKSPACE. Changes: "
            + ", ".join(check.changed_fields)
            + ". You may refine files further or call submit_harness."
        )

    def submit_harness(self) -> str:
        self.submitted = True
        # a previous refusal left self.errors set, and nothing cleared it on
        # success — so the standard weak-model sequence (submit -> "validate
        # first" -> validate -> submit ACCEPTED) still read as invalid: the
        # candidate was discarded and a repair session was launched over a
        # perfectly good draft.
        self.errors = []
        if self.validated_revision != self.workspace_revision:
            self.errors = [
                "submit_harness requires a successful validate_harness after "
                "the most recent file edit"
            ]
            return "SUBMITTED but INVALID (minimum reward):\n- " + self.errors[0]
        # Reuse the exact validated object: submit_harness is atomic and never
        # invokes a reviewer or rewrites component bytes after H1 stops.
        check = self._validated_check
        if check is None:
            self.errors = ["validated workspace result is missing"]
            return "SUBMITTED but INVALID (minimum reward):\n- " + self.errors[0]
        if not check.valid:
            self.errors = check.errors
            return "SUBMITTED but INVALID (minimum reward):\n- " + "\n- ".join(
                check.errors
            )
        # A pre-seeded starter the proposer never touched is scaffolding, not
        # a component, so it is excluded from the signature — but only while
        # its body is still the placeholder (v2.6: a rewritten starter counts).
        placeholders = self._unchanged_starters()
        sig = ";".join(sorted(
            str(f) for f in check.changed_fields
            if not str(f).startswith("system_prompt")
            and str(f).split(".", 1)[-1] not in placeholders
        ))
        if self.component_kind == "params" and self.novelty_refusals < 2:
            groups = ("sampling.", "agent.", "middleware.")
            touched = [f for f in check.changed_fields
                       if str(f).startswith(groups)
                       and not str(f).startswith("new_middlewares.")]
            if not touched:
                # PARAMS-001 (measured): given the params role, the proposer
                # edited skill_body instead — the role text alone does not
                # bind.  Refuse with the concrete edit it needs to make.
                self.novelty_refusals += 1
                self.submitted = False
                return (
                    "REFUSED (parameters slot): this candidate exists to "
                    "measure ONE runtime-parameter move in isolation, but "
                    "no field under sampling / agent / middleware changed "
                    f"(you changed: {sorted(map(str, check.changed_fields))}). "
                    "Edit agent.yaml — e.g. llm_config.temperature, "
                    "llm_config.max_tokens, max_iterations, or a mounted "
                    "middleware's params — then validate_harness and "
                    "submit_harness. Stay inside the ranges in your brief."
                )
        if (self.component_only or self.require_new_tool) and not sig \
                and placeholders and self.novelty_refusals < 2:
            # v2.6 anti-"probe first": keeping the seeded placeholder tool
            # satisfies the contract at zero effort and measures nothing.
            self.novelty_refusals += 1
            self.submitted = False
            return (
                "REFUSED (component-only slot, placeholder unchanged): the "
                "starter tool is a PLACEHOLDER that only reports the score "
                "and the remaining budget — shipping it unchanged measures "
                "nothing. Replace its body with a capability the executor "
                "will actually use on THIS task (a checker, a constructor, a "
                "local optimiser), or delete it and ship a skill or a "
                "middleware instead. Then validate_harness and "
                "submit_harness."
            )
        if self.component_only and not sig and self.novelty_refusals < 2:
            # EXPLORE-001b: a component-only slot that submits a prompt-only
            # change wasted the slot — the isolation contract exists so a
            # COMPONENT can be measured.  Refuse twice with a concrete build
            # recipe, then accept (never deadlock the gate).
            self.novelty_refusals += 1
            self.submitted = False
            return (
                "REFUSED (component-only slot, no component): this candidate "
                "exists to measure a COMPONENT in isolation, but nothing "
                "outside prompt.md changed. Build one now: for a tool — "
                "write_harness_file(\"custom_tools/<name>.py\") with a real "
                "run(ctx, args), write_harness_file(\"tools/<name>.tool.yaml\") "
                "with its schema, mount it in agent.yaml (binding "
                "inner.harness.tools.custom_runtime:custom_tool, "
                "extra_kwargs.py_path), and append ONE line to prompt.md "
                "telling the executor to call it early; for a skill — "
                "write_harness_file(\"skills/<name>/SKILL.md\") teaching a "
                "construction family the incumbent prompt does NOT already "
                "describe, mount it under skills in agent.yaml, and append "
                "one line telling the executor to LoadSkill it first. Then "
                "validate_harness and submit_harness. "
                f"({2 - self.novelty_refusals} refusal(s) left.)"
            )
        # subset match: a component proven futile is still a repeat when the
        # candidate bundles something else alongside it
        sig_set = set(sig.split(";")) if sig else set()
        repeat = next(
            (f for f in self.futile_signatures
             if f and set(f.split(";")) <= sig_set), None)
        if repeat:
            sig = repeat
            if self.novelty_refusals < 2:
                self.novelty_refusals += 1
                self.submitted = False
                return (
                    "REFUSED (repeat proposal): the component change ["
                    + sig + "] was already tried in previous rounds and its "
                    "paired causal effect was ~zero every time. Propose "
                    "something genuinely different (see the campaign evidence "
                    "in your briefing): a different construction family in "
                    "prompt.md, a different capability tool, or a different "
                    "skill. Edit, validate_harness, then submit again. "
                    f"({2 - self.novelty_refusals} refusal(s) left before a "
                    "repeat is accepted.)"
                )
            self.novelty_flag = "repeat_accepted_after_warnings"
        self.partial_spec = check.partial
        self.effective = check.effective
        self.changed_fields = check.changed_fields
        self.component_audit = list(check.component_audit)
        self.raw_submission = yaml.safe_dump(
            check.partial, sort_keys=False, allow_unicode=True, width=100
        ).strip()
        return (
            "SUBMITTED. Candidate H2 accepted; changes: "
            + ", ".join(check.changed_fields)
            + "."
        )


_CURRENT: "contextvars.ContextVar[Optional[ProposeSession]]" = contextvars.ContextVar(
    "propose_session", default=None
)


def get_session() -> ProposeSession:
    s = _CURRENT.get()
    if s is None:
        raise RuntimeError("no active ProposeSession (tool called outside propose_scope)")
    return s


@contextmanager
def propose_scope(session: ProposeSession):
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)

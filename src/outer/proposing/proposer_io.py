"""Proposer I/O — round-context builder + versioning (NOT the H1 harness).

The H1 harness itself (system prompt, tools, skill, sampling) is the declarative
NexAU package at ``src/outer/harness/`` — fixed forever (plan.md §0.3). This
module is only the surrounding glue: it builds the round-varying USER message
fed to that harness, renders feedback, and hashes the package for provenance.
Renamed from ``h1.py`` (that name wrongly implied it *was* H1).

It builds the round-varying USER message for ONE task instance (plan.md §2.2:
H_j ~ pi_phi(H | tau, H1)): the task's public spec, its seed program (excerpt),
the current best harness spec for this task, and its scores.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

H1_VERSION = "h1/2.9-taskagnostic"
H1_PACKAGE = Path(__file__).resolve().parents[1] / "harness"

_SEED_PROGRAM_CAP = 5000  # chars of the seed program shown to the proposer

USER_TEMPLATE = """# Task instance: {task_id}

## Public task description
{task_spec}

## Seed program (the executor edits the EVOLVE-BLOCK region){seed_note}
```python
{seed_program}
```

## Scores on this task (combined_score, higher is better; budget {max_evals} evals)
- seed program alone: {seed_score:.6g}
- current harness best: {base_score:.6g}{stuck_tag}{campaign_context}

# Current H2 harness
A complete private filesystem copy of the current H2 is available through the
harness file tools. Do not assume its contents from prior rounds. Your FIRST
action must be `harness_shell(command="cat agent.yaml")`. Follow mounts only as
needed. Next read `prompt.md`. For example, if you may change tools, run
`ls tools/`, then `cat` the specific schema and implementation before editing.

# Task
Design ONE H2 tailored to THIS task by inspecting and editing that filesystem.
Diagnose why the current H2 reaches only {base_score:.6g}, make one coherent
file-level change, call validate_harness, then submit_harness.

The seed code above is context only: NEVER copy or implement it as an H2 file,
generated tool, or EVOLVE-BLOCK. Two anti-anchoring rules (H1 v2.4): never
write absolute claims like "X is known to be optimal" into H2 files — they
bias every later round; and never re-implement items from the incumbent
prompt's own plan text or from the "causally futile" list above — those are
already tried and verified to change nothing. The rollout runtime supplies the actual program
to the future executor. By your sixth tool call, make an H2 edit. The reliable
default is one concise task-specific appendix to the already-inspected prompt:
`edit_harness_file(path="prompt.md", append_text="...")`, then immediately
`validate_harness()` and, when valid, `submit_harness()`."""


def build_user_message(*, task_id: str, task_spec: str, seed_program: str,
                       seed_score: float, base_score: float,
                       base_spec: Dict[str, Any], max_evals: int = 20,
                       campaign_context: str = "") -> str:
    prog = seed_program.strip()
    note = ""
    if len(prog) > _SEED_PROGRAM_CAP:
        prog = prog[:_SEED_PROGRAM_CAP]
        note = f" (truncated to {_SEED_PROGRAM_CAP} chars)"
    stuck = "  <-- STUCK AT SEED: the harness makes no progress here" \
        if abs(base_score - seed_score) < 1e-9 else ""
    return USER_TEMPLATE.format(
        task_id=task_id,
        task_spec=task_spec.strip() or "(no public description; rely on the seed program)",
        seed_note=note,
        seed_program=prog,
        seed_score=seed_score,
        base_score=base_score,
        stuck_tag=stuck,
        campaign_context=("\n" + campaign_context.rstrip()
                          if campaign_context.strip() else ""),
        max_evals=max_evals,
    )


# H1 v2.4 slot roles: the non-forced slots stop being homogeneous — each gets
# an explicit exploration mandate so a saturated incumbent cannot collapse the
# whole batch into prose-tweak clones (observed: 7 rounds x 8 slots proposing
# the same futile skill).  Roles are H1-side prompt engineering only; budgets,
# gates, and pairing are untouched.
SLOT_ROLE_MESSAGES = {
    "rewrite": (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: STRUCTURAL REWRITE\n"
        "Rewrite prompt.md WHOLESALE (write_harness_file, not append): compress "
        "what still earns its keep and propose a solution approach "
        "STRUCTURALLY DIFFERENT from the incumbent's. Derive the alternative "
        "from THIS task's own structure — read the task statement and the "
        "current program in your context, name the family the incumbent is "
        "stuck in, and propose one it cannot reach by parameter tweaks. "
        "A rewrite that merely rephrases the incumbent recipe is a wasted slot."
    ),
    "tool": (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: NEW CAPABILITY TOOL "
        "(COMPONENT-ONLY)\n"
        "Propose ONE new generated tool with real computational value for this "
        "task — a constraint checker, a candidate generator, a local "
        "optimiser, an input analyser — derived from THIS task's structure, "
        "not another score/budget probe. Keep the schema "
        "small, mount it in agent.yaml. COMPONENT-ONLY CONTRACT (mechanically "
        "enforced): the incumbent prompt.md is inherited — your ONLY prompt "
        "change is ONE short appended declaration line; large prompt edits "
        "will be refused. The paired control then differs by the component "
        "alone.\n"
        "MEASURED FACT you must design around: the executor spends its turns "
        "on the mandatory edit->evaluate loop (16.3 evaluate and 17.4 edit "
        "calls per rollout) and essentially never spends one on an OPTIONAL "
        "tool — including the CORE `probe_solution`, used 0.07 times per "
        "rollout out of a 30-call budget, and generated tools, invoked in "
        "7/55 rollouts even though 55/55 were mounted and named in its "
        "prompt. Generated SKILLS, which cost no turn because they are "
        "auto-enacted into the context, were used 17/17. So an extra "
        "optional step will not be taken. Design accordingly, best first: "
        "(a) ship the capability as an AUTOMATIC middleware that runs "
        "without costing a turn; (b) make the tool REPLACE a step the "
        "executor already takes rather than add one; (c) only as a last "
        "resort, declare a trigger that is unambiguously true on turn 1."
    ),
    "skill": (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: ALTERNATIVE-FAMILY SKILL "
        "(COMPONENT-ONLY)\n"
        "Write ONE new skill teaching a solution family DIFFERENT from the "
        "incumbent's: identify what family the incumbent program belongs to "
        "(from the task statement and the program in your context) and teach "
        "one it cannot reach by tuning — a different algorithm, "
        "representation, search strategy, or decomposition. State when to "
        "use it and how to build it concretely for THIS task. "
        "COMPONENT-ONLY CONTRACT "
        "(mechanically enforced): the incumbent prompt.md is inherited — your "
        "ONLY prompt change is ONE short appended declaration line naming the "
        "skill and telling the executor to consult it early; large prompt "
        "edits will be refused. This isolates the skill's causal effect: the "
        "paired control differs from you by the component alone."
    ),
    "middleware": (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: RUNTIME BEHAVIOUR "
        "(COMPONENT-ONLY)\n"
        "Write ONE new generated middleware (hook: before_model) that changes "
        "how the executor searches — for example: detect that the last N edits "
        "stayed in one structural family and say so; surface the remaining "
        "evaluation budget with a concrete instruction; refuse a repeated "
        "no-op edit. MIDDLEWARE IS THE ONLY ZERO-COST CHANNEL: it runs "
        "automatically before a model call and costs the executor no turn, "
        "whereas an optional tool competes with the mandatory "
        "edit->evaluate loop and is measurably almost never called. "
        "COMPONENT-ONLY CONTRACT (mechanically enforced): the incumbent "
        "prompt.md is inherited — only ONE short appended line is allowed; "
        "put the behaviour in the middleware file, mount it in agent.yaml."
    ),
}


# v2.5 slot classes (EXPLORE-001):
#   component-only slots (skill/tool roles) inherit the incumbent prompt and
#   may only append one short declaration line — the paired delta then
#   measures the component itself, not a bundled prompt rewrite;
#   the epsilon slot proposes via the FROZEN BASE replica so component-
#   authoring behavior can never vanish from the training distribution
#   (round 0 proved the base proposes components; trained phi stops).
def component_only_slot(k: int, force_ks, diversity: bool,
                        k_total: int = 8) -> bool:
    # v2.6.3: middleware joined after its slot's seeded PLACEHOLDER hook
    # (return None) sailed through submit, rode along with free prompt
    # edits, and "won" a round on the prompt's merit — the role text always
    # said COMPONENT-ONLY, but this flag never enforced it for middleware.
    return slot_role_name(k, force_ks, diversity, k_total) in (
        "skill", "tool", "middleware", "params")


def epsilon_slot(k: int, force_ks, diversity: bool,
                 k_total: int = 8) -> bool:
    """The last free NON-COMPONENT slot samples from the frozen base policy.

    It must not land on a component-only slot: those carry a submit guard
    that refuses anything but their own component kind, which contradicts
    the epsilon arm's "propose whatever you judge best".  At K=4 the last
    free slot IS the middleware slot, so picking free[-1] blindly put the
    two contracts on the same candidate.
    """
    if not diversity or k in force_ks:
        return False
    free = [i for i in range(max(int(k_total), 1)) if i not in force_ks]
    eligible = [i for i in free
                if not component_only_slot(i, force_ks, diversity, k_total)]
    return bool(eligible) and k == eligible[-1]


def slot_role_name(k: int, force_ks, diversity: bool,
                   k_total: int = 8) -> str:
    if not diversity or k in force_ks:
        return ""
    # COMPONENT-FIRST (v2.6).  With K=4 and one forced slot the previous
    # rewrite-first order produced [free, rewrite, rewrite, skill] — no tool
    # slot and no middleware slot existed at all, which is one mechanical
    # reason a small batch proposes nothing but prompts.
    order = ["tool", "skill", "middleware", "params", "rewrite", "rewrite", ""]
    free = [i for i in range(max(int(k_total), 1)) if i not in force_ks]
    try:
        idx = free.index(k)
    except ValueError:
        return ""
    return order[idx] if idx < len(order) else ""


# API contracts shown to component slots (SCAFFOLD-002).  Kept to the
# surfaces that exist in inner/runtime/harness_sdk.py and
# inner/harness/middleware/generated_context.py — inventing fields here
# would teach the proposer to write bodies that crash.
_TOOL_API_NOTE = (
    "\n\nTOOL BODY API — your file defines run(ctx, args); ctx provides:\n"
    "  ctx.get_program() / ctx.get_best_program() -> str   current/best code\n"
    "  ctx.best_score() -> float; ctx.budget_left() -> {evaluations, probes}\n"
    "  ctx.probe(subsample=2000) -> approx score dict (cheap probe budget)\n"
    "  ctx.evaluate() -> official score dict (debits the REAL eval budget)\n"
    "  ctx.stage_edit(diff_or_block) -> stage a SEARCH/REPLACE edit\n"
    "  ctx.list_task_inputs(); ctx.read_input_sample(name); "
    "ctx.read_input_df(name)\n"
    "WORKED EXAMPLE (a tool that actually helps — probe several structural\n"
    "variants and stage the best):\n"
    "  def run(ctx, args):\n"
    "      base = ctx.get_best_program()\n"
    "      results = []\n"
    "      for variant in make_variants(base):      # write this for the task\n"
    "          ctx.stage_edit(variant)\n"
    "          results.append((ctx.probe()[\"score\"], variant))\n"
    "      best = max(results)\n"
    "      ctx.stage_edit(best[1])\n"
    "      return {\"probed\": len(results), \"best_probe\": best[0]}"
)
def _params_note() -> str:
    """Render the tunable-parameter contract from the genome's own bounds.

    Generated, never hand-copied: a drifted bound would teach the proposer
    to write proposals the validator rejects.
    """
    from outer.genome import harness_spec as _hs
    lines = []
    for group, fields in (("sampling", _hs._SAMPLING_FIELDS),
                          ("agent", _hs._AGENT_FIELDS),
                          ("middleware", _hs._MIDDLEWARE_FIELDS)):
        for key, (lo, hi, is_int) in fields.items():
            kind = "int" if is_int else "float"
            lines.append(f"  {group}.{key}: {kind} in [{lo}, {hi}]")
    return (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: RUNTIME PARAMETERS "
        "(ISOLATED)\n"
        "Change the executor's OPERATING POINT, not its instructions. The "
        "incumbent prompt is inherited — append at most one short line "
        "saying what you changed and why. Editable fields and their "
        "validated ranges:\n" + "\n".join(lines) + "\n"
        "WHAT THESE CONTROL: sampling.temperature/top_p/top_k set how "
        "widely the executor explores per turn (low = exploit a known "
        "family, high = diversify); sampling.max_tokens caps how much it "
        "can write in one turn (raise it when edits are being truncated); "
        "agent.max_iterations is the TURN budget for the whole rollout "
        "(the evaluation budget is separate and fixed); "
        "middleware.stall_after / max_restarts govern the built-in "
        "stall-restart policy; budget_reminder_from_left decides how early "
        "the executor is warned about its remaining evaluations; "
        "long_tool_output_max_chars decides how much evaluator feedback "
        "survives truncation (raise it when errors are being cut off).\n"
        "Change ONE or TWO related fields, not the whole block: the paired "
        "control then measures exactly that operating-point move. State "
        "your hypothesis in the declaration line (e.g. \"temperature 0.7 "
        "-> 0.4: the incumbent already found the right family, exploit "
        "it\")."
    )


_LOOP_NOTE = (
    "\n\nWHERE YOUR MIDDLEWARE RUNS — the executor's agent loop, one "
    "iteration:\n"
    "    before_agent   (once, at start)\n"
    " -> before_model   <-- YOUR HOOK RUNS HERE, every iteration\n"
    " -> LLM call       (the executor decides its next tool call)\n"
    " -> after_model\n"
    " -> before_tool -> tool executes -> after_tool\n"
    " -> (loop back to before_model)\n"
    "    after_agent    (once, at end)\n"
    "You get the LAST word before the executor thinks: your hook sees the "
    "run state and can shape that decision. It runs on EVERY iteration and "
    "costs the executor no turn. Only `before_model` is available in this "
    "genome — you cannot intercept a tool call or read a tool result, so "
    "express everything as a pre-decision intervention."
)
_MW_API_NOTE = (
    "\n\nMIDDLEWARE BODY API — your before_model(hook_input) gets "
    "hook_input[\"state\"] with:\n"
    "  iteration, best_so_far, evals_done, evals_remaining, probes_remaining,\n"
    "  stalled_evals (evals since last improvement), family_streak (edits in\n"
    "  a row within one structural family), families_explored, last_error,\n"
    "  current_program_valid_syntax.\n"
    "TWO EFFECTS ARE AVAILABLE (this is your whole action space):\n"
    "  1. return \"...\"  -> injects that text as a framework message the "
    "executor reads before deciding. Advisory.\n"
    "  2. return {\"note\": \"...\", \"require_tools\": [\"probe_solution\"]} "
    "-> ALSO sets a TOOL GATE: the executor's next call must be one of the "
    "named tools or it is refused and told why. Gateable tools: "
    "probe_solution, edit_solution, evaluate_solution. The gate clears as "
    "soon as it is satisfied, and auto-lifts after 2 refusals so it can "
    "never deadlock the rollout.\n"
    "  return None -> no intervention this iteration.\n"
    "WHY THE GATE MATTERS: the executor spends its turns on the mandatory "
    "edit->evaluate loop and almost never takes an OPTIONAL step (the core "
    "probe_solution is used 0.07 times per rollout out of 30). A gate is "
    "the only mechanism that can make a step happen at the moment it is "
    "worth taking — and it costs no turn to decide.\n"
    "WORKED EXAMPLE (advisory + gate, against the measured failure modes):\n"
    "  def before_model(hook_input):\n"
    "      s = hook_input.get(\"state\", {})\n"
    "      # stalled inside one construction family -> force a cheap probe\n"
    "      # before the next edit instead of another blind mutation\n"
    "      if s.get(\"stalled_evals\", 0) >= 2 and s.get(\"probe_available\"):\n"
    "          return {\"note\": \"No gain for 2 evals in the same family. "
    "Probe a structurally DIFFERENT candidate before editing.\",\n"
    "                  \"require_tools\": [\"probe_solution\"]}\n"
    "      if s.get(\"evals_remaining\", 99) <= 2:\n"
    "          return \"2 evaluations left - only high-confidence edits.\"\n"
    "      return None"
)

_COMPONENT_BUDGET_NOTE = (
    "\n\nBUDGET NOTE: the workspace files are already in this message — do "
    "NOT spend turns on harness_shell inspection. A measured build costs "
    "about seven calls: write the component file, mount it in agent.yaml, "
    "append one declaration line, validate_harness, submit_harness."
)


def slot_role(k: int, force_ks, diversity: bool,
              k_total: int = 8) -> str:
    """Map slot index -> role directive ('' = free slot)."""
    name = slot_role_name(k, force_ks, diversity, k_total)
    msg = SLOT_ROLE_MESSAGES.get(name, "")
    if name == "params":
        return _params_note()
    if name in ("tool", "skill", "middleware"):
        msg += _COMPONENT_BUDGET_NOTE
    if name == "tool" or (not name and k in (force_ks or set())):
        msg += _TOOL_API_NOTE
    elif name == "middleware":
        msg += _LOOP_NOTE + _MW_API_NOTE
    if epsilon_slot(k, force_ks, diversity):
        msg += (
            "\n\n## NOTE: EXPLORATION SAMPLE\n"
            "This candidate is sampled for exploration; propose whatever "
            "coherent improvement you judge best — components (skills, "
            "tools, middleware) are as welcome as prompt changes."
        )
    return msg


def h1_hash() -> str:
    """Hash the whole H1 package (every file) for provenance."""
    h = hashlib.sha256()
    for f in sorted(H1_PACKAGE.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(H1_PACKAGE)).encode())
            h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def render_workspace(draft_dir) -> str:
    """Render the complete mutable H2 workspace into proposer context.

    PRELOAD-001: the historical H1 loop spent most of its iterations on
    ``cat`` round-trips before the first edit.  Injecting every mutable file
    up front lets the proposer state its edits directly and keeps the whole
    trajectory within a small iteration budget (edits -> <=5 validation
    cycles -> submit).  Read-before-edit is satisfied by construction — the
    session marks these files pre-inspected.

    Returns ``(rendered_text, rendered_relative_paths)``.
    """
    from pathlib import Path

    draft_dir = Path(draft_dir)
    per_file_cap = 6000
    total_cap = 32000
    ordered = [draft_dir / "agent.yaml", draft_dir / "prompt.md"]
    for pattern in ("tools/*.tool.yaml", "custom_tools/*.py",
                    "skills/*/SKILL.md", "middlewares/*.py",
                    "middlewares/*.middleware.yaml"):
        ordered.extend(sorted(draft_dir.glob(pattern)))
    parts = ["\n\n## CURRENT H2 WORKSPACE (complete — do NOT re-inspect)\n"
             "Every mutable file is included below verbatim. State your "
             "edits directly with edit_harness_file / write_harness_file, "
             "then validate_harness (repair only reported errors, at most "
             "~5 validation cycles), then submit_harness."]
    used = 0
    rendered = []
    for path in ordered:
        if not path.is_file():
            continue
        relative = path.relative_to(draft_dir).as_posix()
        text = path.read_text(errors="replace")
        clipped = text[:per_file_cap]
        suffix = "\n... [truncated]" if len(text) > per_file_cap else ""
        block = (f"\n### FILE: {relative}\n```\n{clipped}{suffix}\n```\n")
        if used + len(block) > total_cap:
            parts.append(f"\n### FILE: {relative} (omitted for length — "
                         "cat it only if you must edit it)\n")
            rendered.append(relative)
            continue
        used += len(block)
        parts.append(block)
        rendered.append(relative)
    parts.append("\n(previously mounted middleware wrappers outside the "
                 "files above are fixed runtime and not editable)\n")
    return "".join(parts), rendered


def render_feedback(fb: dict) -> str:
    """Compact previous-visit telemetry section appended to the H1 user message."""
    lines = [
        "\n\n## Telemetry from the previous visit (round %s)" % fb.get("round"),
    ]
    if fb.get("analyst_note"):
        lines.append("ANALYST NOTE: " + fb["analyst_note"])
    best = fb.get("best_score") or fb.get("base_score", 0.0)
    accepted = fb.get("accepted_improvement")
    accepted_score = fb.get("outgoing_base_score", fb.get("base_score", 0.0))
    if accepted is False and best > fb.get("base_score", 0.0):
        score_line = (
            "The raw candidate maximum was %.6g, but it was not accepted "
            "(%s); the incumbent remains %.6g."
            % (best, fb.get("program_ratchet_reason", "attribution failed"), accepted_score)
        )
    else:
        score_line = (
            "The starting score was %.6g; the best of 8 candidate harnesses reached %.6g."
            % (fb.get("base_score", 0.0), best)
        )
    lines += [
        score_line,
        "%s of the candidates made NO progress past the starting program." % fb.get("n_stuck_at_base", "?"),
        "Per-candidate outcomes (k: score, evals used, stop reason, changed fields):",
    ]
    for c in fb.get("candidates", []):
        if c.get("invalid"):
            lines.append("  k%s: INVALID SPEC (never rolled out)" % c["k"])
            continue
        if c.get("tools"):
            lines.append("  k%s: tools=%s" % (c["k"], ", ".join(c["tools"])))
        lines.append("  k%s: score=%s evals=%s stop=%s changed=%s%s" % (
            c["k"],
            ("%.6g" % c["score"]) if c.get("score") is not None else "?",
            c.get("evals"), c.get("stop"), ",".join(c.get("changed", [])),
            (" err=" + c["err"]) if c.get("err") else ""))
    lines.append(
        "Diagnose WHY those harnesses failed to progress (e.g. the strategy they pushed "
        "saturated, edits kept failing, budget was wasted on timeouts) and design a harness "
        "that overcomes that specific failure mode. Do not resubmit a near-copy of a design "
        "that already stalled.")
    return "\n".join(lines)


def render_prior_actions(actions: list) -> str:
    """Sequential-sampling context: the VALID actions already proposed for THIS
    task in THIS batch, so this sample proposes something genuinely different
    (Adaptive within-batch diversity). Only the changed axes + a compact spec
    outline are shown — no scores, no rollout outcomes (those aren't available
    yet mid-batch, and withholding them keeps the channel leak-free)."""
    if not actions:
        return ""
    lines = ["\n\n## Already proposed this batch (do NOT paraphrase these)"]
    for a in actions:
        fields = ", ".join(a.get("changed_fields") or []) or "(no changes)"
        spec = a.get("spec") or {}
        tools = [t.get("name") for t in (spec.get("new_tools") or []) if isinstance(t, dict)]
        skills = [s.get("name") for s in (spec.get("new_skills") or []) if isinstance(s, dict)]
        extra = ""
        if tools:
            extra += " new_tools=[%s]" % ", ".join(filter(None, tools))
        if skills:
            extra += " new_skills=[%s]" % ", ".join(filter(None, skills))
        lines.append("  - k%s changed: %s%s" % (a.get("k"), fields, extra))
    lines.append("Propose a design that explores a DIFFERENT axis or strategy "
                 "than every entry above.")
    return "\n".join(lines)

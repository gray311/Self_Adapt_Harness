"""Proposer I/O — round-context builder + versioning (NOT the H1 harness).

The H1 harness itself is the fixed Cordis composition at
``src/outer/harness/cordis.yml``. This
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

H1_VERSION = "h1/3.0-cordis"
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
action must be `harness_shell(command="cat cordis.yml")`. That file contains
the executor persona, runtime parameters, and every mounted Cordis plugin.
Follow only relevant `./plugins/*.mjs` entries before editing.

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
default is one concise task-specific addition inside
`system-prompt.config.persona` in `cordis.yml` using an exact replacement, then
immediately
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
        "Rewrite system-prompt.config.persona inside cordis.yml: compress what "
        "still earns its keep and propose a solution approach "
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
        "not another score/budget probe. Implement it as a native .mjs Cordis "
        "plugin, mount it with one config.sah tool row in cordis.yml, and keep "
        "its inputSchema small. COMPONENT-ONLY CONTRACT: the incumbent persona "
        "is inherited except for one short declaration naming the plugin and "
        "its trigger. The paired control then differs by the component "
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
        "Implement it as a systemPrompt.section Cordis .mjs plugin and mount it "
        "with config.sah kind/name/description/body metadata. COMPONENT-ONLY "
        "CONTRACT: the incumbent persona is inherited except for one short "
        "declaration naming the auto-enacted skill. This isolates its effect: the "
        "paired control differs from you by the component alone."
    ),
    "middleware": (
        "\n\n## EXPLORATION ROLE FOR THIS CANDIDATE: RUNTIME BEHAVIOUR "
        "(COMPONENT-ONLY)\n"
        "Write ONE native Cordis middleware plugin on `agent/pre-step` that changes "
        "how the executor searches — for example: detect that the last N edits "
        "stayed in one structural family and say so; surface the remaining "
        "evaluation budget with a concrete instruction; refuse a repeated "
        "no-op edit. MIDDLEWARE IS THE ONLY ZERO-COST CHANNEL: it runs "
        "automatically before a model call and costs the executor no turn, "
        "whereas an optional tool competes with the mandatory "
        "edit->evaluate loop and is measurably almost never called. "
        "COMPONENT-ONLY CONTRACT: keep the incumbent persona except for one "
        "short declaration; put behavior in plugins/*.mjs and mount it in "
        "cordis.yml with config.sah kind middleware and hook agent/pre-step."
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


_TOOL_API_NOTE = (
    "\n\nCORDIS TOOL PLUGIN CONTRACT — write a self-contained ESM .mjs file:\n"
    "  export const name = 'sah-tool-<name>'\n"
    "  export const inject = ['tools', 'sahBridge']\n"
    "  export function apply(ctx) { ctx.tools.register({ name: '<name>', "
    "description, parameters, output, async execute(args, exec) {...} }) }\n"
    "The plugin may use pure JavaScript and the injected narrow service:\n"
    "  await ctx.sahBridge.call('__sah_snapshot', {}, exec.signal, "
    "String(exec.callId ?? ''))\n"
    "This returns current/best program summaries, score, budget, and recent "
    "search state. No import, process/global access, dynamic code, filesystem, "
    "or network API is allowed. The tool name, description, and JSON schema "
    "must also appear in its cordis.yml config.sah metadata."
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
        "incumbent persona is inherited — add at most one short line "
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
    "\n\nWHERE YOUR MIDDLEWARE RUNS: Cordis `agent/pre-step` is a waterfall "
    "immediately before each model request. Call `const decision = await "
    "next()` exactly once`; preserve reject decisions, and return either the "
    "same decision or `{ kind: 'enter', messages: [...] }`. It costs no model "
    "turn. The only supported generated hook is `agent/pre-step`."
)
_MW_API_NOTE = (
    "\n\nCORDIS MIDDLEWARE API — inject ['sahBridge'] when runtime state is "
    "needed and call `__sah_snapshot`. To add advice, append one user-role "
    "message with a deterministic id, text content, and a plugin source record "
    "to `decision.messages`. Also call `__sah_middleware_event` with name, "
    "event (`invoked`/`fired`/`error`), and iteration so participation is "
    "auditable. Never drop downstream messages."
)

_COMPONENT_BUDGET_NOTE = (
    "\n\nBUDGET NOTE: the workspace files are already in this message — do "
    "NOT spend turns on harness_shell inspection. A measured build costs "
    "about five calls: write the plugin, edit its cordis.yml insert/metadata "
    "and one persona declaration, validate_harness, submit_harness."
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
    ordered = [draft_dir / "cordis.yml"]
    ordered.extend(sorted(draft_dir.glob("plugins/*.mjs")))
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
    parts.append("\n(The trusted sah-bridge plugin and model transport live "
                 "outside this package and are fixed runtime.)\n")
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

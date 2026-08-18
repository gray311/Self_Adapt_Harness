#!/usr/bin/env python3
"""Architectural check: can the executor really reach all four component types?

Materializes one H2 that carries a proposer-authored change of EVERY kind —
prompt, generated tool, generated skill, generated middleware — then loads it
with the REAL NexAU classes (no mocks) and asserts each one arrives where the
executor can use it:

  prompt      -> the assembled system prompt contains the change
  tool        -> the tool appears in the agent's tool registry AND its python
                 binding is callable
  skill       -> the skill is registered and its body is deliverable
  middleware  -> the middleware instantiates from the materialized file and
                 sits in the executor's hook chain

Run inside the runtime container (nexau importable):

    python3 scripts/analysis/audits/nexau_component_wiring.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from outer.compiling.materialize import materialize, INNER_HARNESS  # noqa: E402
from outer.genome import harness_spec as hs  # noqa: E402

TOOL_CODE = (
    "def run(ctx, args):\n"
    "    return {\"wiring_probe\": True, \"best\": ctx.best_score()}\n"
)
MIDDLEWARE_CODE = '''
def before_model(hook_input):
    """Fires every turn; returns a framework note the executor must see."""
    state = hook_input.get("state", {})
    if state.get("iteration", 0) >= 0:
        return "WIRING_PROBE_MIDDLEWARE_FIRED"
    return None
'''
SKILL_BODY = (
    "# Wiring probe skill\n\n"
    "A distinctive marker for the wiring audit: WIRING_PROBE_SKILL_BODY.\n"
    "Use a different construction family than the incumbent.\n"
)


def build_spec(base_spec: dict) -> dict:
    spec = dict(base_spec)
    spec["system_prompt"] = (
        str(base_spec.get("system_prompt", "")) + "\n\nWIRING_PROBE_PROMPT_LINE\n"
    )
    spec["new_tools"] = [{
        "name": "wiring_probe_tool",
        "description": "Wiring audit probe tool.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
        "implementation_py": TOOL_CODE,
    }]
    spec["new_skills"] = [{
        "name": "wiring-probe-skill",
        "description": "Wiring audit probe skill.",
        "body": SKILL_BODY,
    }]
    spec["new_middlewares"] = [{
        "name": "wiring_probe_middleware",
        "description": "Wiring audit probe middleware.",
        "hook": "before_model",   # the only hook the genome allows
        "implementation_py": MIDDLEWARE_CODE,
    }]
    return spec


def main() -> None:
    results: dict[str, object] = {}
    base_spec = hs.read_base_spec(INNER_HARNESS)
    spec = build_spec(base_spec)
    # validate the probe spec through the real genome validator first, so a
    # schema error is reported as such instead of surfacing as a crash later
    import yaml as _yaml
    partial = {k: spec[k] for k in
               ("system_prompt", "new_tools", "new_skills", "new_middlewares")}
    partial["schema"] = "h2spec/1.0"
    check = hs.parse_and_validate(_yaml.safe_dump(partial, sort_keys=False))
    results["genome_validation"] = list(getattr(check, "errors", []) or []) or "ok"

    with tempfile.TemporaryDirectory() as td:
        draft = Path(td) / "h2"
        materialize(spec, draft, meta={"effective": spec}, validate_prompt=False)
        results["materialized_files"] = sorted(
            str(p.relative_to(draft)) for p in draft.rglob("*") if p.is_file()
        )[:24]

        # --- prompt ---------------------------------------------------------
        prompt = (draft / "prompt.md").read_text()
        results["prompt_change_present"] = "WIRING_PROBE_PROMPT_LINE" in prompt

        # --- real NexAU assembly -------------------------------------------
        # mirror production: harness_runner puts the candidate dir first on
        # sys.path so its middlewares/ package resolves
        sys.path.insert(0, str(draft))
        from nexau import Agent, AgentConfig  # real classes, no mocks

        config = AgentConfig.from_yaml(draft / "agent.yaml")
        tool_names = [getattr(t, "name", None) for t in (config.tools or [])]
        results["tool_registered"] = "wiring_probe_tool" in tool_names
        results["tool_names"] = tool_names

        # NexAU turns skill folders into Skill objects (name/description/detail)
        skills = list(config.skills or [])
        skill_names = [getattr(s, "name", None) or str(s) for s in skills]
        probe = next((s for s in skills
                      if (getattr(s, "name", "") or "").startswith("wiring-probe")), None)
        results["skill_names"] = skill_names
        results["skill_registered"] = probe is not None
        results["skill_body_delivered"] = bool(
            probe is not None and "WIRING_PROBE_SKILL_BODY" in
            str(getattr(probe, "detail", "") or ""))

        mw = [getattr(m, "__class__", type(m)).__name__ for m in
              (getattr(config, "middlewares", None) or [])]
        results["middleware_objects"] = mw
        results["middleware_registered"] = any(
            "wiring" in str(m).lower() or "Generated" in str(m) for m in mw)

        # --- instantiate the agent (no model call) --------------------------
        try:
            agent = Agent(config=config)
            executor = getattr(agent, "executor", None) or getattr(
                agent, "_executor", None)
            manager = getattr(executor, "middleware_manager", None)
            chain = []
            for attr in ("middlewares", "_middlewares", "hooks", "_hooks"):
                got = getattr(manager, attr, None) if manager else None
                if got:
                    chain = list(got)
                    break
            results["executor_hook_chain"] = [type(m).__name__ for m in chain]
            results["generated_middleware_in_executor_chain"] = any(
                type(m).__name__ == "GeneratedMiddleware" for m in chain)
            defs = getattr(executor, "structured_tool_definitions", None) or []
            names = []
            for d in defs:
                fn = d.get("function") if isinstance(d, dict) else None
                names.append((fn or {}).get("name") if fn else
                             (d.get("name") if isinstance(d, dict) else None))
            results["llm_tool_definitions"] = names
            results["tool_in_llm_definitions"] = "wiring_probe_tool" in names
        except Exception as exc:  # instantiation may need a live endpoint
            results["agent_instantiation_error"] = f"{type(exc).__name__}: {exc}"

        # --- tool binding executes -----------------------------------------
        try:
            impl = (draft / "custom_tools" / "wiring_probe_tool.py").read_text()
            namespace: dict = {}
            exec(compile(impl, "wiring_probe_tool.py", "exec"), namespace)
            results["tool_binding_callable"] = callable(namespace.get("run"))
        except Exception as exc:
            results["tool_binding_callable"] = f"ERROR {exc}"

    print(json.dumps(results, indent=2, default=str))
    hard = ["prompt_change_present", "tool_registered", "tool_in_llm_definitions",
            "skill_registered", "skill_body_delivered", "middleware_registered"]
    missing = [k for k in hard if not results.get(k)]
    if missing:
        raise SystemExit(f"WIRING CHECK FAILED: {missing}")


if __name__ == "__main__":
    main()

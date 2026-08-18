#!/usr/bin/env python3
"""Fail-closed native Cordis wiring audit for every proposer-owned component.

The audit materializes a package containing a persona edit, generated tool,
generated skill, and `agent/pre-step` middleware. It then reads that package
back through the production parser and workspace validator, proving that the
Cordis composition, mounted plugin bytes, semantic genome, and safety gates all
agree. No model or evaluator call is made.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from outer.compiling.materialize import INNER_HARNESS, materialize  # noqa: E402
from outer.genome import harness_spec as hs  # noqa: E402
from outer.workspace.h2_workspace import validate_workspace  # noqa: E402


TOOL_NAME = "wiring_probe_tool"
SKILL_NAME = "wiring-probe-skill"
MIDDLEWARE_NAME = "wiring_probe_middleware"

TOOL_JS = f"""export const name = 'sah-tool-{TOOL_NAME}'
export const inject = ['tools', 'sahBridge']

export function apply(ctx) {{
  ctx.tools.register({{
    name: '{TOOL_NAME}',
    description: 'Read a compact, trusted SAH runtime snapshot.',
    parameters: {{ type: 'object', properties: {{}}, additionalProperties: false }},
    output: {{
      schema: {{ type: 'string' }},
      render: (_args, value) => [{{ type: 'text', text: value }}],
    }},
    async execute(_args, exec) {{
      const value = await ctx.sahBridge.call(
        '__sah_snapshot', {{}}, exec.signal, String(exec.callId ?? ''),
      )
      return JSON.stringify(value)
    }},
  }})
}}
"""

MIDDLEWARE_JS = f"""export const name = 'sah-middleware-{MIDDLEWARE_NAME}'

export function apply(ctx) {{
  ctx.on('agent/pre-step', async (_payload, next) => {{
    return next()
  }})
}}
"""


def build_spec(base: dict) -> dict:
    spec = json.loads(json.dumps(base))
    spec["system_prompt"] = str(base["system_prompt"]).rstrip() + f"""

# Generated Tool: {TOOL_NAME}
Conditional Cordis snapshot helper; call only when runtime state is needed.

# Generated Skill: {SKILL_NAME}
Automatically enacted Cordis strategy marker: WIRING_PROBE_SKILL_BODY.

# Generated Middleware: {MIDDLEWARE_NAME}
Runs automatically at Cordis agent/pre-step and delegates with next().
"""
    spec["new_tools"] = [{
        "name": TOOL_NAME,
        "description": "Read a compact, trusted SAH runtime snapshot.",
        "input_schema": {
            "type": "object", "properties": {}, "additionalProperties": False,
        },
        "implementation_js": TOOL_JS,
    }]
    spec["new_skills"] = [{
        "name": SKILL_NAME,
        "description": "Automatically enacted wiring audit skill.",
        "body": "# Wiring probe skill\n\nWIRING_PROBE_SKILL_BODY",
    }]
    spec["new_middlewares"] = [{
        "name": MIDDLEWARE_NAME,
        "description": "Cordis waterfall wiring audit middleware.",
        "hook": "agent/pre-step",
        "implementation_js": MIDDLEWARE_JS,
    }]
    return spec


def main() -> None:
    base = hs.read_base_spec(INNER_HARNESS)
    effective = build_spec(base)
    with tempfile.TemporaryDirectory(prefix="sah-cordis-wiring-") as temp:
        package = Path(temp) / "h2"
        materialize(effective, package, meta={"effective": effective})
        check = validate_workspace(package, base)
        parsed = hs.read_base_spec(package)
        files = sorted(
            path.relative_to(package).as_posix()
            for path in package.rglob("*") if path.is_file()
        )
        expected_files = {
            "cordis.yml",
            "plugins/discovery-optimization.mjs",
            f"plugins/sah-tool-{TOOL_NAME}.mjs",
            f"plugins/sah-skill-{SKILL_NAME}.mjs",
            f"plugins/sah-middleware-{MIDDLEWARE_NAME}.mjs",
            "component_manifest.json",
            "meta.json",
        }
        inventory = hs.generated_component_inventory(parsed)
        result = {
            "schema": parsed.get("schema"),
            "workspace_valid": check.valid,
            "workspace_errors": check.errors,
            "changed_fields": check.changed_fields,
            "files": files,
            "expected_files_present": expected_files.issubset(files),
            "inventory": inventory,
            "persona_marker_present": "WIRING_PROBE_SKILL_BODY" in parsed["system_prompt"],
            "tool_source_exact": parsed["new_tools"][0]["implementation_js"] == TOOL_JS,
            "middleware_source_exact": (
                parsed["new_middlewares"][0]["implementation_js"] == MIDDLEWARE_JS
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        required = {
            "workspace_valid": check.valid,
            "expected_files_present": result["expected_files_present"],
            "persona_marker_present": result["persona_marker_present"],
            "tool_source_exact": result["tool_source_exact"],
            "middleware_source_exact": result["middleware_source_exact"],
            "tool_inventory": inventory["new_tools"] == [TOOL_NAME],
            "skill_inventory": inventory["new_skills"] == [SKILL_NAME],
            "middleware_inventory": inventory["new_middlewares"] == [MIDDLEWARE_NAME],
        }
        failed = [name for name, passed in required.items() if not passed]
        if failed:
            raise SystemExit("CORDIS WIRING AUDIT FAILED: " + ", ".join(failed))
        print("CORDIS_COMPONENT_WIRING_OK")


if __name__ == "__main__":
    main()

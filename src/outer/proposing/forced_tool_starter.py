"""Pre-seed a mounted native Cordis tool plugin for constrained H1 slots."""
from __future__ import annotations

import json
from pathlib import Path

from outer.compiling.materialize import materialize
from outer.genome.harness_spec import read_base_spec


PLACEHOLDER_MARKER = "// SAH_CORDIS_PLACEHOLDER"


def tool_plugin(name: str, *, useful: bool = False) -> str:
    description = (
        "Return a compact runtime snapshot to help choose the next search action."
        if useful else "Placeholder Cordis capability; replace its execute body."
    )
    result = (
        "const snapshot = await ctx.sahBridge.call('__sah_snapshot', {}, exec.signal, String(exec.callId ?? ''))\n"
        "      return JSON.stringify(snapshot)"
        if useful else "return JSON.stringify({ replace_me: true })"
    )
    marker = "" if useful else PLACEHOLDER_MARKER + "\n"
    return f"""{marker}export const name = {json.dumps('sah-tool-' + name)}
export const inject = ['tools', 'sahBridge']

export function apply(ctx) {{
  ctx.tools.register({{
    name: {json.dumps(name)},
    description: {json.dumps(description)},
    parameters: {{ type: 'object', properties: {{}}, additionalProperties: false }},
    output: {{
      schema: {{ type: 'string' }},
      render: (_args, value) => [{{ type: 'text', text: value }}],
    }},
    async execute(_args, exec) {{
      {result}
    }},
  }})
}}
"""


def _free_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    index = 2
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def seed_starter_tool(draft_dir, slot_index: int, body: str | None = None) -> str:
    draft_dir = Path(draft_dir)
    effective = read_base_spec(draft_dir, verify_provenance=False)
    taken = {row["name"] for row in effective.get("new_tools") or []}
    name = _free_name(f"slot_tool_k{slot_index}", taken)
    source = (
        body if isinstance(body, str)
        and "export function apply" in body
        and PLACEHOLDER_MARKER not in body
        else tool_plugin(name, useful=body is None)
    )
    effective.setdefault("new_tools", []).append({
        "name": name,
        "description": "Pre-mounted Cordis tool slot; replace with a task-specific capability.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "implementation_js": source.rstrip() + "\n",
    })
    effective["system_prompt"] = effective["system_prompt"].rstrip() + (
        f"\n\n# Generated Tool: {name}\n"
        "Conditional Cordis capability: call it only when its documented trigger applies."
    )
    materialize(effective, draft_dir, meta={"effective": effective})
    return name

"""Pre-mounted native Cordis component scaffolding for constrained slots."""
from __future__ import annotations

import json
from pathlib import Path

from outer.compiling.materialize import materialize
from outer.genome.harness_spec import read_base_spec
from outer.proposing.forced_tool_starter import PLACEHOLDER_MARKER, tool_plugin


TOOL_PLACEHOLDER = tool_plugin("placeholder_tool")


def middleware_plugin(name: str) -> str:
    return f"""{PLACEHOLDER_MARKER}
export const name = {json.dumps('sah-middleware-' + name)}

export function apply(ctx) {{
  ctx.on('agent/pre-step', async (_payload, next) => {{
    return next()
  }})
}}
"""


MIDDLEWARE_PLACEHOLDER = middleware_plugin("placeholder_middleware")


def _free_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    index = 2
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def seed_middleware_scaffold(draft_dir, slot_index: int) -> str:
    draft_dir = Path(draft_dir)
    effective = read_base_spec(draft_dir, verify_provenance=False)
    taken = {row["name"] for row in effective.get("new_middlewares") or []}
    name = _free_name(f"slot_hook_k{slot_index}", taken)
    effective.setdefault("new_middlewares", []).append({
        "name": name,
        "hook": "agent/pre-step",
        "description": "Pre-mounted Cordis middleware slot; replace its no-op listener.",
        "implementation_js": middleware_plugin(name),
    })
    effective["system_prompt"] = effective["system_prompt"].rstrip() + (
        f"\n\n# Generated Middleware: {name}\n"
        "Runs automatically at Cordis agent/pre-step; replace the placeholder behavior."
    )
    materialize(effective, draft_dir, meta={"effective": effective})
    return name


def placeholder_bodies() -> set[str]:
    return {PLACEHOLDER_MARKER}

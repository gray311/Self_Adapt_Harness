"""Pre-seeded minimal generated tool for forced-tool slots (BAILOUT-005).

Four smoke iterations showed that prompting alone cannot make a weak base
proposer reliably create a valid generated tool from scratch inside 32
turns: slots repair-looped on gate constraints, missed the agent.yaml
mount, paced the work onto the final turns, or answered in prose and died.
Under the fail-closed proposal gate a single such slot kills the round.

The mechanical floor: at propose time a forced slot's private draft
workspace starts with a slot-unique, statically-valid, self-test-passing
minimal tool ALREADY written, mounted, and declared.  The baseline state
therefore satisfies the forced-tool contract and validation; the proposer's
job becomes "improve or replace the starter" instead of "create from
scratch", and keeping only the starter is a legal (weak) proposal that the
paired reward will judge on merit.  Nothing is submitted on the model's
behalf — it must still validate and submit.
"""
from __future__ import annotations

from pathlib import Path

import yaml

TEMPLATE = (
    "def run(ctx, args):\n"
    "    best = ctx.best_score()\n"
    "    return {\"best_score\": best, "
    "\"budget_left\": ctx.budget_left()}\n"
)

DESCRIPTION = "Report the current best score and remaining budget."


def starter_name(slot_index: int) -> str:
    return f"quick_probe_k{int(slot_index)}"


def _free_name(base: str, taken: set) -> str:
    """First unclaimed starter name.

    When a forced slot wins a round, the NEXT round's base package inherits
    its starter — re-seeding the same name would fail deterministically on
    every retry and dead-lock the lane.  Suffix letters keep the name unique
    against any inherited generation.
    """

    if base not in taken:
        return base
    for suffix in "bcdefghijklmnopqrstuvwxyz":
        candidate = f"{base}{suffix}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"no free starter name for {base!r}")


def seed_starter_tool(draft_dir: Path, slot_index: int,
                      body: str | None = None) -> str:
    """Write + mount the starter into a freshly materialized draft.

    Returns the tool name (slot-unique, adjusted past any tools inherited
    from the base package).
    """

    draft_dir = Path(draft_dir)

    agent_path = draft_dir / "agent.yaml"
    agent = yaml.safe_load(agent_path.read_text())
    mounts = agent.setdefault("tools", [])
    name = _free_name(
        starter_name(slot_index),
        {str(row.get("name")) for row in mounts},
    )

    (draft_dir / "custom_tools").mkdir(exist_ok=True)
    (draft_dir / "custom_tools" / f"{name}.py").write_text(body or TEMPLATE)

    (draft_dir / "tools" / f"{name}.tool.yaml").write_text(yaml.safe_dump({
        "type": "tool",
        "name": name,
        "description": DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }, sort_keys=False))

    mounts.append({
        "name": name,
        "yaml_path": f"./tools/{name}.tool.yaml",
        "binding": "inner.harness.tools.custom_runtime:custom_tool",
        "extra_kwargs": {"py_path": f"./custom_tools/{name}.py"},
    })
    agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))

    prompt_path = draft_dir / "prompt.md"
    prompt_path.write_text(
        prompt_path.read_text().rstrip("\n")
        + f"\n\n# Generated Tool: {name}\n{DESCRIPTION} Call it to check "
        "progress cheaply before deciding the next edit.\n"
    )
    return name

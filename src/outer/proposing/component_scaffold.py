"""Pre-seeded, already-mounted scaffolding for component slots.

Measured on two AC2 runs: the SKILL slot succeeds because its build is
cheap — one file, one mount, one declaration line, validate, submit (seven
tool calls).  The TOOL and MIDDLEWARE slots additionally have to author a
JSON schema or a hook wrapper, and they ran out of the iteration budget
(0/5 and 0/5 valid) even after the budget was raised.

So the plumbing is seeded here and the proposer only has to write the body.
The placeholder body is deliberately useless and the component-only submit
guard refuses a slot that ships it unchanged, so this buys the proposer
turns without buying it a free pass.
"""
from __future__ import annotations

from pathlib import Path

import yaml

TOOL_PLACEHOLDER = (
    "def run(ctx, args):\n"
    "    # PLACEHOLDER — replace this body with a real capability for THIS\n"
    "    # task.  Shipping it unchanged is refused.\n"
    "    return {\"replace_me\": True}\n"
)

MIDDLEWARE_PLACEHOLDER = (
    "def before_model(hook_input):\n"
    "    # PLACEHOLDER — replace this body.  Runs automatically before every\n"
    "    # model call and costs the executor no turn; return a string to\n"
    "    # inject a framework note, or None for no message.\n"
    "    return None\n"
)

_MW_WRAPPER = '''"""Generated middleware (h2spec/1.0), lifecycle-audited."""
from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput, HookResult, Middleware)
from nexau.core.messages import Message, Role, TextBlock
from inner.harness.middleware.generated_context import GeneratedHookTracker
from inner.harness.tools.runtime import get_session

# --USER-HOOK-START--
{user_code}
# --USER-HOOK-END--

class GeneratedMiddleware(Middleware):
    def __init__(self):
        self._name = {name_repr}
        self._hook = "before_model"
        self._tracker = GeneratedHookTracker()
        self._registered = False
        self._ensure_registered()

    def _ensure_registered(self):
        if self._registered:
            return
        try:
            get_session().register_middleware(self._name, self._hook)
            self._registered = True
        except Exception:
            pass

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        self._ensure_registered()
        try:
            session = get_session()
            state = self._tracker.snapshot(hook_input, session)
            note = before_model({{"state": state}})
        except Exception:
            return HookResult.no_changes()
        if not note:
            return HookResult.no_changes()
        return HookResult.with_modifications(messages=[
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=str(note))]),
        ])
'''


def _free_name(base: str, taken) -> str:
    name = base
    suffix = ord("b")
    while name in taken:
        name = f"{base}{chr(suffix)}"
        suffix += 1
    return name


def seed_middleware_scaffold(draft_dir, slot_index: int) -> str:
    """Write + mount a valid placeholder middleware; return its name."""
    draft_dir = Path(draft_dir)
    agent_path = draft_dir / "agent.yaml"
    agent = yaml.safe_load(agent_path.read_text())
    mounts = agent.setdefault("middlewares", []) or []
    existing = {str(m.get("import", "")) for m in mounts}
    name = _free_name(f"slot_hook_k{slot_index}",
                      {e.split(".")[-1].split(":")[0] for e in existing})

    mw_dir = draft_dir / "middlewares"
    mw_dir.mkdir(exist_ok=True)
    (mw_dir / f"{name}.py").write_text(
        _MW_WRAPPER.format(user_code=MIDDLEWARE_PLACEHOLDER.strip(),
                           name_repr=repr(name))
    )
    (mw_dir / f"{name}.middleware.yaml").write_text(yaml.safe_dump({
        "name": name,
        "hook": "before_model",
        "description": "Placeholder generated middleware — replace its body.",
    }, sort_keys=False))
    mounts.insert(0, {"import": f"middlewares.{name}:GeneratedMiddleware",
                      "params": {}})
    agent["middlewares"] = mounts
    agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))

    prompt_path = draft_dir / "prompt.md"
    prompt_path.write_text(
        prompt_path.read_text().rstrip("\n")
        + f"\n\n# Generated Middleware: {name}\nRuns automatically before "
        "each model call; no tool call is needed.\n"
    )
    return name


def placeholder_bodies() -> set:
    """Bodies that count as untouched scaffolding, whatever the file name."""
    return {TOOL_PLACEHOLDER, MIDDLEWARE_PLACEHOLDER}

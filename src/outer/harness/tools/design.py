"""Python bridge callables for H1's fixed Cordis action space.

The trusted ``sah-bridge`` Cordis plugin exposes these functions as model
tools; each resolves the rollout-local :class:`ProposeSession`.
"""
from __future__ import annotations

from outer.harness.tools.runtime import get_session


def harness_shell(command: str) -> str:
    """Inspect the private H2 with a read-only shell-shaped command."""
    return get_session().inspect_harness(command)


def write_harness_file(path: str, content: str) -> str:
    """Create or replace one mutable file in the private H2 package."""
    return get_session().write_harness_file(path, content)


def edit_harness_file(
    path: str,
    old_text: str = "",
    new_text: str = "",
    append_text: str = "",
) -> str:
    """Replace one exact occurrence or append guidance to a mutable H2 file."""
    return get_session().edit_harness_file(
        path, old_text, new_text, append_text
    )


def delete_harness_file(path: str) -> str:
    """Delete one mutable plugin after removing its cordis.yml mount."""
    return get_session().delete_harness_file(path)


def validate_harness() -> str:
    """Parse, safety-check, and canonically compile the current H2 workspace."""
    return get_session().validate_harness()


def submit_harness() -> str:
    """Submit the current validated H2 workspace (stop tool)."""
    return get_session().submit_harness()

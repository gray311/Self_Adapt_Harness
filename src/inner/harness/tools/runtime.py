"""Run-scoped bridge between Cordis tool calls and the active task.

The contextvar itself lives in ``inner.runtime.session``; this compatibility
module keeps the established Python callable import path stable.
"""
from __future__ import annotations

from inner.runtime.session import get_session, session_scope

__all__ = ["get_session", "session_scope"]

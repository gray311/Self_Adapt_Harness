"""Read-only compatibility dispatcher for archived h2spec/1.0 packages.

A materialized candidate carries its generated tools under
``custom_tools/<name>.py`` (each defining ``def run(ctx, args): ...``, already
static-gated and self-tested by the historical validator). Older packages bind
every such tool to this dispatcher and pass the source path as an internal
argument:

    tools:
      - name: sample_hit_probe
        yaml_path: ./tools/sample_hit_probe.tool.yaml
        binding: inner.harness.tools.custom_runtime:custom_tool
        extra_kwargs: {_sah_py_path: ./custom_tools/sample_hit_probe.py}

Live h2spec/2 candidates use native Cordis ``plugins/*.mjs`` instead. This
module remains only so archived experiment artifacts can still be inspected.
"""
from __future__ import annotations

import json
import hashlib
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

from inner.harness.tools.runtime import get_session
from inner.runtime.harness_sdk import ToolContext

_CACHE: Dict[str, Callable] = {}


def _load_run(py_path: Path) -> Callable:
    source = py_path.read_bytes()
    key = f"{py_path}:{hashlib.sha256(source).hexdigest()}"
    fn = _CACHE.get(key)
    if fn is not None:
        return fn
    ns: Dict[str, Any] = {}
    exec(compile(source, str(py_path), "exec"), ns)  # gated + self-tested upstream
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError(f"custom tool {py_path} defines no run()")
    _CACHE[key] = fn
    return fn


def _resolve_source(session, declared: str) -> Path:
    if not session.harness_dir:
        raise ValueError("active session has no candidate H2 root")
    raw = Path(declared)
    if raw.is_absolute():
        raise ValueError("absolute custom-tool paths are forbidden")
    root = Path(session.harness_dir).resolve()
    path = (root / raw).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("custom-tool path escapes candidate H2") from exc
    if len(relative.parts) != 2 or relative.parts[0] != "custom_tools" \
            or path.suffix != ".py" or not path.is_file():
        raise ValueError("custom-tool path must be custom_tools/<name>.py")
    return path


def custom_tool(
    _sah_py_path: str = "", py_path: str = "", **kwargs: Any,
) -> str:
    """Generic binding for every generated tool.

    ``_sah_py_path`` comes from trusted compiler-owned ``extra_kwargs``; the
    legacy ``py_path`` alias keeps old materialized packages readable. Model
    arguments arrive in ``kwargs``.
    """
    declared_path = _sah_py_path or py_path
    if not declared_path:
        return "custom tool error: no py_path bound"
    try:
        session = get_session()
    except Exception:
        return "custom tool error: no active session"
    name = Path(declared_path).stem
    session.record_tool_event(name, "invoked")
    try:
        source = _resolve_source(session, declared_path)
        ctx = ToolContext(session, session.custom_tool_scratch())
        out = _load_run(source)(ctx, dict(kwargs))
    except Exception as exc:
        session.record_tool_event(name, "error", str(exc))
        return "custom tool raised (ignored):\n" + traceback.format_exc(limit=3)[-600:]
    session.record_tool_event(name, "completed")
    if isinstance(out, str):
        return out[:8000]
    try:
        return json.dumps(out)[:8000]
    except Exception:
        return str(out)[:8000]

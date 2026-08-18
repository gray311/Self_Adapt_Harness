"""Static safety gates for proposer-generated Cordis plugins.

Fail-closed: any violation returns errors and the candidate is rejected.  The
canonical H1 must edit the workspace and call ``validate_harness`` again; no
post-submit model repairs or byte mutations are allowed.

Native Cordis plugins may use only the injected registries/events and narrow
``sahBridge`` service.  The Python checks retained below support reading old
experiment artifacts; live candidate validation uses :func:`check_cordis_plugin`.
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

IMPORT_WHITELIST = {
    "math", "re", "json", "itertools", "functools", "collections", "heapq",
    "bisect", "random", "statistics", "string", "typing", "dataclasses",
    "numpy", "pandas",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview", "exit", "quit",
    "getattr", "setattr", "delattr", "type", "object", "super",
    # hasattr was removed from this set after a real generated tool was
    # rejected for using it: it is read-only reflection, returns a bool,
    # and cannot smuggle attribute access the way getattr can.
}
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "multiprocessing", "threading", "signal", "resource", "pickle",
    "requests", "urllib", "http", "ftplib", "telnetlib", "builtins", "io",
}
MAX_CODE_CHARS = 6000
FORBIDDEN_ATTRIBUTES = {
    # File/network-capable entry points reachable from otherwise useful
    # numerical/data libraries.
    "load", "loads", "save", "savez", "savez_compressed", "fromfile",
    "tofile", "memmap", "read_csv", "read_json", "read_pickle",
    "read_parquet", "read_excel", "read_feather", "read_hdf", "read_sql",
    "to_csv", "to_json", "to_pickle", "to_parquet", "to_excel",
    "to_feather", "to_hdf", "to_sql", "to_clipboard",
}

MIDDLEWARE_STATE_KEYS = {
    "iteration", "current_iteration", "best_so_far", "evaluator_calls",
    "evaluations_remaining", "evals_remaining", "evals_done", "probe_calls",
    "probe_attempts", "probes_remaining", "probe_remaining",
    "probe_available", "probes_since_eval", "edit_calls", "stalled_evals",
    "family_streak", "families_explored", "last_family",
    "structure_streak", "structures_explored",
    "current_program_valid_syntax", "last_step_kind", "last_error",
    "last_validity", "last_score",
    "active_tool_gate",
}
MIDDLEWARE_CONTEXT_KEYS = MIDDLEWARE_STATE_KEYS | {"state"}


# Candidate plugins execute inside the Cordis process, so their authority is
# deliberately smaller than an ordinary Node module. They may use the injected
# Cordis registry/event APIs and the narrow ``sahBridge`` service, but cannot
# discover globals, import modules, or construct code dynamically.
MAX_CORDIS_PLUGIN_CHARS = 20000
_JS_FORBIDDEN = {
    "process", "global", "globalThis", "require", "eval", "Function",
    "constructor", "prototype", "__proto__", "Reflect", "Proxy",
    "WebAssembly", "fetch", "XMLHttpRequest", "WebSocket", "EventSource",
    "Worker", "SharedArrayBuffer", "Atomics", "Deno", "Bun",
    "child_process", "node:fs", "node:net", "node:http", "node:https",
}


def check_cordis_plugin(
    code: str, *, kind: str, name: str, hook: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Fail-closed structural/capability gate for one native ``.mjs`` plugin.

    This is intentionally a narrow source contract, not a general JavaScript
    sandbox. The official Cordis loader still parses the module before rollout;
    the workspace validator additionally performs an isolated load smoke.
    """
    errors: List[str] = []
    if not isinstance(code, str) or not code.strip():
        return False, ["plugin source must be a non-empty string"]
    if len(code) > MAX_CORDIS_PLUGIN_CHARS:
        errors.append(f"plugin exceeds {MAX_CORDIS_PLUGIN_CHARS} chars")
    if kind not in {"tool", "middleware"}:
        errors.append(f"unsupported generated plugin kind: {kind!r}")
    if "export function apply" not in code:
        errors.append("must export function apply(ctx, config?)")
    if re.search(r"(^|\n)\s*import(?:\s|\()", code):
        errors.append("imports and dynamic import() are forbidden")
    if re.search(r"\bexport\s+\*|\bexport\s*\{", code):
        errors.append("re-export syntax is forbidden")
    if re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", code):
        errors.append("escaped identifiers/strings are forbidden")
    for token in sorted(_JS_FORBIDDEN):
        if re.search(r"(?<![A-Za-z0-9_$])" + re.escape(token)
                     + r"(?![A-Za-z0-9_$])", code):
            errors.append(f"forbidden JavaScript capability: {token}")
    if re.search(r"\bthis\b", code):
        errors.append("JavaScript `this` is forbidden; use the explicit ctx argument")
    if re.search(r"\[[\"'](?:constructor|prototype|__proto__)[\"']\]", code):
        errors.append("prototype-chain access is forbidden")
    if not re.search(r"export\s+const\s+name\s*=\s*['\"][^'\"]+['\"]", code):
        errors.append("must export a literal plugin name")

    quoted_name = re.escape(name)
    if kind == "tool":
        if "tools.register" not in code:
            errors.append("tool plugin must call ctx.tools.register(...)")
        if not re.search(r"name\s*:\s*['\"]" + quoted_name + r"['\"]", code):
            errors.append(f"tool plugin must register exact tool name {name!r}")
        if "execute" not in code or "parameters" not in code:
            errors.append("tool plugin must declare parameters and execute")
    elif kind == "middleware":
        expected_hook = hook or "agent/pre-step"
        if expected_hook != "agent/pre-step":
            errors.append("only Cordis agent/pre-step middleware is supported")
        if not re.search(
            r"ctx\.on\(\s*['\"]agent/pre-step['\"]", code,
        ):
            errors.append("middleware plugin must register ctx.on('agent/pre-step', ...)")
        if not re.search(r"\bnext\s*\(", code):
            errors.append("agent/pre-step waterfall must delegate with next()")
    return (not errors), sorted(set(errors))


def check_tool_code(code: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if len(code) > MAX_CODE_CHARS:
        errors.append(f"code exceeds {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]

    run_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(run_defs) != 1:
        errors.append("must define exactly one top-level `def run(ctx, args):`")
    elif [a.arg for a in run_defs[0].args.args][:2] != ["ctx", "args"]:
        errors.append("run() signature must be (ctx, args)")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) \
                else [node.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    errors.append(f"forbidden import: {m}")
                elif root not in IMPORT_WHITELIST:
                    errors.append(f"import not in whitelist: {m}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append(f"forbidden private attribute access: .{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            errors.append(f"forbidden I/O-capable attribute: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")

    return (not errors), sorted(set(errors))

def check_middleware_code(code: str, hook: str) -> Tuple[bool, List[str]]:
    """Gate a generated middleware hook body. Same import/builtin restrictions
    as tools, but requires exactly one top-level `def <hook>(hook_input):` and
    is even stricter — a middleware runs IN-PROCESS on every model turn."""
    errors: List[str] = []
    if len(code) > MAX_CODE_CHARS:
        errors.append(f"code exceeds {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]
    hook_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == hook]
    if len(hook_defs) != 1:
        errors.append(f"must define exactly one top-level `def {hook}(hook_input):`")
    elif [a.arg for a in hook_defs[0].args.args][:1] != ["hook_input"]:
        errors.append(f"{hook}() signature must be (hook_input)")
    def _is_state_view(node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "hook_input" and node.func.attr == "get" \
                and node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value == "state"
        if isinstance(node, ast.BoolOp):
            return any(_is_state_view(v) for v in node.values)
        return False

    state_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _is_state_view(value):
                state_aliases.update(t.id for t in targets if isinstance(t, ast.Name))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    errors.append(f"forbidden import: {m}")
                elif root not in IMPORT_WHITELIST:
                    errors.append(f"import not in whitelist: {m}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append(f"forbidden private attribute access: .{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            errors.append(f"forbidden I/O-capable attribute: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in ({"hook_input"} | state_aliases):
            errors.append("middleware context must be read with .get(<documented key>)")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "hook_input" and node.attr != "get":
            errors.append(f"unsupported hook_input attribute: .{node.attr}; use documented .get() keys")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.attr == "get":
            base = node.func.value.id
            allowed = MIDDLEWARE_CONTEXT_KEYS if base == "hook_input" else \
                (MIDDLEWARE_STATE_KEYS if base in state_aliases else None)
            if allowed is not None:
                if not node.args or not isinstance(node.args[0], ast.Constant) \
                        or not isinstance(node.args[0].value, str):
                    errors.append(f"{base}.get() requires a literal documented key")
                elif node.args[0].value not in allowed:
                    errors.append(f"unsupported middleware context key: {node.args[0].value!r}")
    return (not errors), sorted(set(errors))

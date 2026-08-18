"""Subprocess self-test for generated tool code.

Runs ``run(MockContext, args)`` in an isolated python with a hard timeout and
resource limits. Besides the empty-argument fallback, schema-derived examples
exercise required inputs and every declared enum branch. A tool passes if each
case returns without raising and produces a JSON-representable-ish value.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
sys.path.insert(0, {src_path!r})
from inner.runtime.harness_sdk import MockContext
ns = {{}}
code = open({code_path!r}).read()
exec(compile(code, "generated_tool.py", "exec"), ns)
samples = json.loads({samples_json!r})
outcomes = []
for sample_index, args in enumerate(samples):
    ctx = MockContext({scratch!r})
    try:
        out = ns["run"](ctx, args)
    except Exception:
        import traceback
        print(json.dumps({{
            "ok": False,
            "error": "schema sample %d args=%r: %s" % (
                sample_index, args, traceback.format_exc(limit=4)[-1200:]
            ),
        }}))
        raise SystemExit(0)
    ok_types = (str, dict, list, int, float, type(None))
    if not isinstance(out, ok_types):
        print(json.dumps({{
            "ok": False,
            "error": "schema sample %d returned %s; must be str/dict/list/number"
                     % (sample_index, type(out).__name__),
        }}))
        raise SystemExit(0)
    outcomes.append({{"ctx_calls": ctx.calls, "ret_type": type(out).__name__}})
print(json.dumps({{"ok": True, "samples": len(samples), "outcomes": outcomes}}))
'''


def _schema_value(schema: Dict[str, Any]) -> Any:
    if "default" in schema:
        return schema["default"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    kind = schema.get("type")
    if kind == "integer":
        return max(int(schema.get("minimum", 2)), 1)
    if kind == "number":
        return max(float(schema.get("minimum", 1.0)), 0.5)
    if kind == "boolean":
        return True
    if kind == "string":
        return "sample"
    if kind == "array":
        item_schema = schema.get("items")
        item = _schema_value(item_schema) if isinstance(item_schema, dict) else 1.0
        count = max(int(schema.get("minItems", 2)), 2)
        return [item for _ in range(count)]
    if kind == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties") or {}
        return {
            name: _schema_value(prop)
            for name, prop in properties.items()
            if isinstance(prop, dict)
        }
    return None


def schema_selftest_cases(input_schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build bounded, schema-valid examples, including each enum alternative."""

    cases: List[Dict[str, Any]] = [{}]
    if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
        return cases
    properties = input_schema.get("properties") or {}
    if not isinstance(properties, dict) or not properties:
        return cases
    base = {
        name: _schema_value(prop)
        for name, prop in properties.items()
        if isinstance(prop, dict)
    }
    cases.append(base)
    for name, prop in properties.items():
        enum = prop.get("enum") if isinstance(prop, dict) else None
        if not isinstance(enum, list):
            continue
        for value in enum[1:]:
            cases.append({**base, name: value})
    unique: List[Dict[str, Any]] = []
    seen = set()
    for case in cases:
        encoded = json.dumps(case, sort_keys=True)
        if encoded not in seen:
            seen.add(encoded)
            unique.append(case)
    return unique


def selftest_tool_code(
    code: str,
    timeout_s: float = 30.0,
    input_schema: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Returns (passed, detail)."""
    src_path = str(Path(__file__).resolve().parents[3])  # .../src
    samples_json = json.dumps(schema_selftest_cases(input_schema))
    with tempfile.TemporaryDirectory(prefix="tool_selftest_") as td:
        code_path = Path(td) / "generated_tool.py"
        code_path.write_text(code)
        runner = _RUNNER.format(src_path=src_path, code_path=str(code_path),
                                scratch=str(Path(td) / "scratch"),
                                samples_json=samples_json)
        try:
            r = subprocess.run([sys.executable, "-c", runner],
                               capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"self-test timeout ({timeout_s}s)"
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return False, f"self-test produced no verdict; stderr: {(r.stderr or '')[-400:]}"
        try:
            d = json.loads(line[-1])
        except Exception:
            return False, f"unparseable verdict: {line[-1][:200]}"
        return bool(d.get("ok")), d.get("error") or json.dumps(
            {k: d[k] for k in ("samples", "outcomes") if k in d})

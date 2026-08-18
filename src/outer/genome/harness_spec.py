"""Typed semantic genome for proposer-owned Cordis H2 compositions.

H1 now edits a private materialized H2 filesystem.  After validation, that
directory is parsed into this representation for hashing, provenance, static
gates, and deterministic recompilation.  The genome includes proposer-owned
executor prompts, skills, generated Cordis plugins, mounts, sampling, and
iteration settings.  Core runtime bindings and the evaluation budget remain
externally fixed.

The legacy partial-YAML interface remains supported internally. Missing fields
inherit the base; generated component removal is explicit. A candidate
identical to its base is invalid.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCHEMA_VERSION = "h2spec/2.0-cordis"
_ACCEPTED_SCHEMAS = {SCHEMA_VERSION}

# Generated-plugin structural limits (code safety is enforced by static_gates
# + the reviewer self-test, NOT here — this only checks the spec shape)
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
_RESERVED_TOOL_NAMES = {"edit_solution", "evaluate_solution", "probe_solution",
                        "finish"}
_MAX_NEW_TOOLS = 3

# field -> (type, max_chars) for text fields.
# system_prompt MUST match propose_session._PROMPT_CAP: the write-time guard
# and this validation cap are two views of the SAME genome budget — a
# mismatch (write 10000 / validate 8000) burned round003 of cp10div: every
# session edited past 8000, validation refused, and 12 iterations died in
# the shrink loop as no_submission.
_TEXT_FIELDS = {
    "system_prompt": 10000,
    "skill_description": 600,
    "skill_body": 8000,
}
_TOOL_DESC_FIELDS = {  # tool name -> max chars
    "edit_solution": 1600,
    "evaluate_solution": 1000,
    "probe_solution": 1000,
    "finish": 600,
}
# numeric field -> (min, max, is_int)
_SAMPLING_FIELDS = {
    "temperature": (0.0, 1.5, False),
    "top_p": (0.05, 1.0, False),
    "top_k": (1, 100, True),
    "max_tokens": (1024, 16384, True),
}
_AGENT_FIELDS = {
    "max_iterations": (8, 80, True),
}
_MIDDLEWARE_FIELDS = {
    "budget_reminder_from_left": (0, 10, True),
    "long_tool_output_max_chars": (2000, 20000, True),
    "stall_after": (2, 30, True),
    "max_restarts": (0, 5, True),
}
_TOP_KEYS = {"schema", "system_prompt", "skill_description", "skill_body",
             "tool_descriptions", "sampling", "agent", "middleware",
             "new_tools", "remove_tools", "new_skills", "new_middlewares",
             "remove_generated"}

_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{2,39}$")
_MW_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
# Cordis' typed waterfall is the sole generated middleware hook in this genome.
_MW_HOOKS = {"agent/pre-step"}
_MAX_NEW_SKILLS = 2
_MAX_NEW_MIDDLEWARES = 2
GENERATED_COMPONENT_FIELDS = ("new_tools", "new_skills", "new_middlewares")
_REMOVE_GENERATED_KEYS = {
    "tools": "new_tools",
    "skills": "new_skills",
    "middlewares": "new_middlewares",
}
_CORE_H2_TOOLS = ("edit_solution", "evaluate_solution", "probe_solution", "finish")
_CORE_H2_SKILLS = ("discovery-optimization",)
_CORE_H2_MIDDLEWARES = (
    "sah-bridge", "budget-reminder", "stall-restart", "long-tool-output",
)

# Component declarations in the Cordis persona are executable-interface claims, not
# ordinary prose. Keep the syntax intentionally narrow so normal mentions of
# "tool" or "skill" do not become false positives, while headings such as
# ``# Generated Tool: structure_analyzer`` cannot advertise an unmounted
# component to H2.
_COMPONENT_DECLARATION_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+"
    r"(?P<generated>generated\s+)?"
    r"(?P<kind>tool|skill|middleware)\s*:\s*"
    r"`?(?P<name>[A-Za-z][A-Za-z0-9_-]{1,63})`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_GENERATED_SECTION_RE = re.compile(
    r"^generated\s+(tools?|skills?|middlewares?)\b", re.IGNORECASE
)
_COMPONENT_BULLET_RE = re.compile(
    r"^\s*[-*]\s+`(?P<name>[a-z][a-z0-9_-]{1,63})`"
)
_GENERATED_SUBHEADING_RE = re.compile(
    r"^`?(?P<name>[a-z][a-z0-9_-]{1,63})`?\b.*\bgenerated\b",
    re.IGNORECASE,
)


def _generated_section_declarations(prompt: str) -> List[Tuple[str, str]]:
    """Return component names explicitly listed in generated-* sections."""

    singular = {
        "tool": "tool", "tools": "tool",
        "skill": "skill", "skills": "skill",
        "middleware": "middleware", "middlewares": "middleware",
    }
    declarations: List[Tuple[str, str]] = []
    active: Optional[Tuple[str, int]] = None
    component_level: Optional[int] = None
    for line in prompt.splitlines():
        heading = _MARKDOWN_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if active and level <= active[1]:
                active = None
                component_level = None
            section = _GENERATED_SECTION_RE.match(title)
            if section:
                active = (singular[section.group(1).lower()], level)
                component_level = None
                continue
            if active and level > active[1]:
                declared = _GENERATED_SUBHEADING_RE.match(title)
                if declared:
                    declarations.append((active[0], declared.group("name")))
                    component_level = level
                elif component_level is not None and level <= component_level:
                    component_level = None
            continue
        if active and component_level is None:
            bullet = _COMPONENT_BULLET_RE.match(line)
            if bullet:
                declarations.append((active[0], bullet.group("name")))
    return declarations


@dataclass
class SpecValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)
    spec: Optional[Dict[str, Any]] = None  # canonical (validated, defaults NOT folded in)


def _check_num(errors: List[str], group: str, key: str, val: Any,
               lo: float, hi: float, is_int: bool) -> Any:
    if is_int:
        if isinstance(val, bool) or not isinstance(val, int):
            errors.append(f"{group}.{key}: expected int, got {type(val).__name__}")
            return None
    elif not isinstance(val, (int, float)) or isinstance(val, bool):
        errors.append(f"{group}.{key}: expected number, got {type(val).__name__}")
        return None
    if not (lo <= val <= hi):
        errors.append(f"{group}.{key}: {val} outside [{lo}, {hi}]")
        return None
    return int(val) if is_int else float(val)


def parse_and_validate(text: str) -> SpecValidation:
    """Parse a raw YAML spec (optionally inside ```yaml fences) fail-closed."""
    m = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.DOTALL)
    raw = m.group(1) if m else text
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        return SpecValidation(False, [f"yaml parse error: {str(e)[:200]}"])
    if not isinstance(data, dict):
        return SpecValidation(False, [f"spec must be a mapping, got {type(data).__name__}"])

    errors: List[str] = []
    unknown = set(data) - _TOP_KEYS
    if unknown:
        errors.append(f"unknown top-level keys (fail closed): {sorted(unknown)}")

    out: Dict[str, Any] = {"schema": SCHEMA_VERSION}
    if "schema" in data and data["schema"] not in _ACCEPTED_SCHEMAS:
        errors.append(f"schema must be one of {sorted(_ACCEPTED_SCHEMAS)}, got {data['schema']!r}")

    for key, cap in _TEXT_FIELDS.items():
        if key in data:
            v = data[key]
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{key}: must be a non-empty string")
            elif len(v) > cap:
                errors.append(f"{key}: {len(v)} chars exceeds cap {cap}")
            else:
                out[key] = v.strip()

    if "tool_descriptions" in data:
        td = data["tool_descriptions"]
        if not isinstance(td, dict):
            errors.append("tool_descriptions: must be a mapping")
        else:
            bad = set(td) - set(_TOOL_DESC_FIELDS)
            if bad:
                errors.append(f"tool_descriptions: unknown tools {sorted(bad)}")
            good = {}
            for name, cap in _TOOL_DESC_FIELDS.items():
                if name in td:
                    v = td[name]
                    if not isinstance(v, str) or not v.strip():
                        errors.append(f"tool_descriptions.{name}: must be a non-empty string")
                    elif len(v) > cap:
                        errors.append(f"tool_descriptions.{name}: {len(v)} chars exceeds cap {cap}")
                    else:
                        good[name] = v.strip()
            if good:
                out["tool_descriptions"] = good

    for group, fields_def in (("sampling", _SAMPLING_FIELDS), ("agent", _AGENT_FIELDS),
                              ("middleware", _MIDDLEWARE_FIELDS)):
        if group in data:
            g = data[group]
            if not isinstance(g, dict):
                errors.append(f"{group}: must be a mapping")
                continue
            bad = set(g) - set(fields_def)
            if bad:
                errors.append(f"{group}: unknown keys {sorted(bad)}")
            good = {}
            for key, (lo, hi, is_int) in fields_def.items():
                if key in g:
                    v = _check_num(errors, group, key, g[key], lo, hi, is_int)
                    if v is not None:
                        good[key] = v
            if good:
                out[group] = good

    # --- Cordis generative surface: plugins mounted from cordis.yml ----------
    if "remove_tools" in data:
        rt = data["remove_tools"]
        if not isinstance(rt, list) or not all(isinstance(x, str) for x in rt):
            errors.append("remove_tools: must be a list of tool names")
        else:
            # only optional built-ins may be removed; core edit/evaluate/finish stay
            removable = {"probe_solution"}
            bad = set(rt) - removable
            if bad:
                errors.append(f"remove_tools: not removable {sorted(bad)} "
                              f"(only {sorted(removable)})")
            else:
                out["remove_tools"] = sorted(set(rt))

    if "new_tools" in data:
        nt = data["new_tools"]
        if not isinstance(nt, list):
            errors.append("new_tools: must be a list")
        elif len(nt) > _MAX_NEW_TOOLS:
            errors.append(f"new_tools: {len(nt)} exceeds cap {_MAX_NEW_TOOLS}")
        else:
            seen, good_tools = set(), []
            for i, t in enumerate(nt):
                if not isinstance(t, dict):
                    errors.append(f"new_tools[{i}]: must be a mapping")
                    continue
                extra = set(t) - {"name", "description", "input_schema",
                                  "implementation_js"}
                if extra:
                    errors.append(f"new_tools[{i}]: unknown keys {sorted(extra)}")
                name = t.get("name")
                if not isinstance(name, str) or not _TOOL_NAME_RE.match(name or ""):
                    errors.append(f"new_tools[{i}].name: must match [a-z][a-z0-9_]{{2,31}}")
                elif name in _RESERVED_TOOL_NAMES:
                    errors.append(f"new_tools[{i}].name: {name!r} is reserved")
                elif name in seen:
                    errors.append(f"new_tools[{i}].name: duplicate {name!r}")
                else:
                    seen.add(name)
                desc = t.get("description")
                if not isinstance(desc, str) or not desc.strip() or len(desc) > 800:
                    errors.append(f"new_tools[{i}].description: non-empty string <=800 chars")
                code = t.get("implementation_js")
                if not isinstance(code, str) or len(code) > 20000:
                    errors.append(
                        f"new_tools[{i}].implementation_js: must be Cordis plugin code <=20000 chars"
                    )
                elif "export function apply" not in code \
                        or "tools.register" not in code:
                    errors.append(
                        f"new_tools[{i}].implementation_js: must export apply(ctx) and register a Cordis tool"
                    )
                sch = t.get("input_schema", {"type": "object", "properties": {}})
                if not isinstance(sch, dict):
                    errors.append(f"new_tools[{i}].input_schema: must be a JSON-schema mapping")
                elif sch.get("type") != "object" \
                        or not isinstance(sch.get("properties", {}), dict):
                    errors.append(
                        f"new_tools[{i}].input_schema: must describe an object "
                        "with a properties mapping"
                    )
                if name and name in seen and isinstance(code, str) and isinstance(desc, str):
                    good_tools.append({"name": name, "description": desc.strip(),
                                       "input_schema": sch,
                                       "implementation_js": code.rstrip() + "\n"})
            if good_tools:
                out["new_tools"] = good_tools

    # --- new_skills[]: extra skill playbooks (pure text, no code risk) -------
    if "new_skills" in data:
        ns = data["new_skills"]
        if not isinstance(ns, list):
            errors.append("new_skills: must be a list")
        elif len(ns) > _MAX_NEW_SKILLS:
            errors.append(f"new_skills: {len(ns)} exceeds cap {_MAX_NEW_SKILLS}")
        else:
            seen, good = set(), []
            for i, s in enumerate(ns):
                if not isinstance(s, dict):
                    errors.append(f"new_skills[{i}]: must be a mapping"); continue
                extra = set(s) - {"name", "description", "body"}
                if extra:
                    errors.append(f"new_skills[{i}]: unknown keys {sorted(extra)}")
                name = s.get("name")
                if not isinstance(name, str) or not _SKILL_NAME_RE.match(name or ""):
                    errors.append(f"new_skills[{i}].name: must match [a-z][a-z0-9-]{{2,39}}")
                elif name in seen or name == "discovery-optimization":
                    errors.append(f"new_skills[{i}].name: duplicate/reserved {name!r}")
                else:
                    seen.add(name)
                desc = s.get("description", "")
                body = s.get("body")
                if not isinstance(body, str) or not body.strip() or len(body) > 8000:
                    errors.append(f"new_skills[{i}].body: non-empty string <=8000 chars")
                if name in seen and isinstance(body, str) and body.strip():
                    good.append({"name": name, "description": str(desc).strip()[:600],
                                 "body": body.strip()})
            if good:
                out["new_skills"] = good

    # --- new_middlewares[]: generated hooks (code — same safety chain) --------
    if "new_middlewares" in data:
        nm = data["new_middlewares"]
        if not isinstance(nm, list):
            errors.append("new_middlewares: must be a list")
        elif len(nm) > _MAX_NEW_MIDDLEWARES:
            errors.append(f"new_middlewares: {len(nm)} exceeds cap {_MAX_NEW_MIDDLEWARES}")
        else:
            seen, good = set(), []
            for i, mw in enumerate(nm):
                if not isinstance(mw, dict):
                    errors.append(f"new_middlewares[{i}]: must be a mapping"); continue
                extra = set(mw) - {"name", "description", "hook", "implementation_js"}
                if extra:
                    errors.append(f"new_middlewares[{i}]: unknown keys {sorted(extra)}")
                name = mw.get("name")
                if not isinstance(name, str) or not _MW_NAME_RE.match(name or ""):
                    errors.append(f"new_middlewares[{i}].name: must match [a-z][a-z0-9_]{{2,31}}")
                elif name in seen:
                    errors.append(f"new_middlewares[{i}].name: duplicate {name!r}")
                else:
                    seen.add(name)
                hook = mw.get("hook")
                if hook not in _MW_HOOKS:
                    errors.append(f"new_middlewares[{i}].hook: must be one of {sorted(_MW_HOOKS)}")
                code = mw.get("implementation_js")
                if not isinstance(code, str) or len(code) > 20000:
                    errors.append(
                        f"new_middlewares[{i}].implementation_js: must be Cordis plugin code <=20000 chars"
                    )
                elif "export function apply" not in code \
                        or "agent/pre-step" not in code:
                    errors.append(
                        f"new_middlewares[{i}].implementation_js: must export apply(ctx) and register agent/pre-step"
                    )
                desc = mw.get("description", "")
                if name in seen and isinstance(code, str) and hook in _MW_HOOKS:
                    good.append({"name": name, "hook": hook,
                                 "description": str(desc).strip()[:600],
                                 "implementation_js": code.rstrip() + "\n"})
            if good:
                out["new_middlewares"] = good

    # Explicit deletion is required because omission means inheritance.  The
    # file-native H1 workflow derives this mapping when a plugin entry and file
    # are removed from cordis.yml/plugins.
    if "remove_generated" in data:
        rg = data["remove_generated"]
        if not isinstance(rg, dict):
            errors.append("remove_generated: must be a mapping")
        else:
            extra = set(rg) - set(_REMOVE_GENERATED_KEYS)
            if extra:
                errors.append(
                    f"remove_generated: unknown component groups {sorted(extra)}"
                )
            good_removed: Dict[str, List[str]] = {}
            for group, field_name in _REMOVE_GENERATED_KEYS.items():
                if group not in rg:
                    continue
                names = rg[group]
                if not isinstance(names, list) or not all(
                    isinstance(name, str) for name in names
                ):
                    errors.append(
                        f"remove_generated.{group}: must be a list of names"
                    )
                    continue
                pattern = _SKILL_NAME_RE if field_name == "new_skills" else _TOOL_NAME_RE
                bad = [name for name in names if not pattern.match(name)]
                if bad:
                    errors.append(
                        f"remove_generated.{group}: invalid names {sorted(set(bad))}"
                    )
                    continue
                if names:
                    good_removed[group] = sorted(set(names))
            if good_removed:
                out["remove_generated"] = good_removed

    mutated = set(out) - {"schema"}
    if not mutated:
        errors.append("spec mutates nothing (all fields missing/invalid)")

    if errors:
        return SpecValidation(False, errors)
    return SpecValidation(True, [], out)


def canonical_json(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def spec_hash(spec: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:16]


def _rows_by_name(rows: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["name"]): row
        for row in (rows or [])
        if isinstance(row, dict) and row.get("name")
    }


def merge_named_components(
    inherited: Any, proposed: Any,
) -> List[Dict[str, Any]]:
    """Return a stable, name-keyed union of generated components.

    Components are part of the ratcheted harness genome.  A proposal with a
    new name appends one component; a proposal reusing an inherited name
    explicitly updates that component in place.  Omitting a name never removes
    it.  Deep copies keep proposal review/repair from mutating the base spec.
    """

    out = json.loads(json.dumps(list(inherited or [])))
    positions = {
        str(row.get("name")): i
        for i, row in enumerate(out)
        if isinstance(row, dict) and row.get("name")
    }
    for row in proposed or []:
        copied = json.loads(json.dumps(row))
        name = str(copied["name"])
        if name in positions:
            out[positions[name]] = copied
        else:
            positions[name] = len(out)
            out.append(copied)
    return out


def changed_generated_components(
    effective: Dict[str, Any], base: Dict[str, Any], field: str,
) -> List[Dict[str, Any]]:
    """Generated components added or explicitly updated by this candidate."""

    if field not in GENERATED_COMPONENT_FIELDS:
        raise ValueError(f"not a generated component field: {field}")
    base_by_name = _rows_by_name(base.get(field))
    return [
        row for row in effective.get(field, [])
        if base_by_name.get(str(row.get("name"))) != row
    ]


def generated_component_inventory(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Compact ordered inventory used by package provenance and audits."""

    return {
        field: [str(row["name"]) for row in spec.get(field, [])]
        for field in GENERATED_COMPONENT_FIELDS
    }


def added_generated_component_names(
    base: Dict[str, Any], effective: Dict[str, Any], field: str,
) -> List[str]:
    """Names newly introduced by a candidate, excluding in-place updates."""

    if field not in GENERATED_COMPONENT_FIELDS:
        raise ValueError(f"not a generated component field: {field}")
    base_names = set(generated_component_inventory(base)[field])
    return [
        name for name in generated_component_inventory(effective)[field]
        if name not in base_names
    ]


def h2_component_catalog(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Every component name the proposer-owned H2 prompt must expose."""

    removed = set(spec.get("remove_tools", []))
    tools = [name for name in _CORE_H2_TOOLS if name not in removed]
    tools += generated_component_inventory(spec)["new_tools"]
    return {
        "tools": tools,
        "skills": list(_CORE_H2_SKILLS)
                  + generated_component_inventory(spec)["new_skills"],
        "middlewares": generated_component_inventory(spec)["new_middlewares"]
                       + list(_CORE_H2_MIDDLEWARES),
    }


def component_prompt_issues(spec: Dict[str, Any]) -> List[str]:
    """Find missing or falsely advertised executor components in the prompt.

    Presence checks make mounted components discoverable. Declaration checks
    enforce the reverse direction for explicit component headings: an H1 may
    not describe a fictional ``Generated Tool`` in prose without mounting it.
    """

    prompt = str(spec.get("system_prompt", ""))
    issues: List[str] = []
    for group, names in h2_component_catalog(spec).items():
        missing = []
        for name in names:
            pattern = r"(?<![A-Za-z0-9_-])" + re.escape(name) \
                      + r"(?![A-Za-z0-9_-])"
            if re.search(pattern, prompt) is None:
                missing.append(name)
        if missing:
            issues.append(
                f"system_prompt does not name current {group}: {missing}"
            )

    catalog = h2_component_catalog(spec)
    generated = generated_component_inventory(spec)
    kind_to_group = {
        "tool": "tools",
        "skill": "skills",
        "middleware": "middlewares",
    }
    kind_to_field = {
        "tool": "new_tools",
        "skill": "new_skills",
        "middleware": "new_middlewares",
    }
    declarations: set[Tuple[str, str, bool]] = set()
    for match in _COMPONENT_DECLARATION_RE.finditer(prompt):
        kind = match.group("kind").lower()
        name = match.group("name")
        is_generated = bool(match.group("generated"))
        declarations.add((kind, name, is_generated))
    declarations.update(
        (kind, name, True)
        for kind, name in _generated_section_declarations(prompt)
    )
    for kind, name, is_generated in sorted(declarations):
        allowed = (
            generated[kind_to_field[kind]]
            if is_generated else catalog[kind_to_group[kind]]
        )
        if name not in allowed:
            qualifier = "generated " if is_generated else ""
            issues.append(
                f"system_prompt declares unmounted {qualifier}{kind} {name!r}"
            )
    return issues


def diff_to_partial(
    effective: Dict[str, Any], base: Dict[str, Any],
) -> Dict[str, Any]:
    """Encode a full effective H2 as a minimal, round-local mutation.

    This is also the bridge from the file-native proposer workspace back to
    h2spec provenance.  Unlike the legacy omission-only representation, it
    records component deletions explicitly.
    """

    out: Dict[str, Any] = {"schema": SCHEMA_VERSION}
    for key in _TEXT_FIELDS:
        if effective.get(key, "").strip() != base.get(key, "").strip():
            out[key] = effective.get(key, "")

    changed_descs = {}
    for name in _TOOL_DESC_FIELDS:
        value = effective.get("tool_descriptions", {}).get(name, "")
        if value.strip() != base.get("tool_descriptions", {}).get(name, "").strip():
            changed_descs[name] = value
    if changed_descs:
        out["tool_descriptions"] = changed_descs

    for group in ("sampling", "agent", "middleware"):
        values = {
            key: value
            for key, value in effective.get(group, {}).items()
            if value != base.get(group, {}).get(key)
        }
        if values:
            out[group] = values

    for field in GENERATED_COMPONENT_FIELDS:
        changed = changed_generated_components(effective, base, field)
        if changed:
            out[field] = json.loads(json.dumps(changed))

    removed: Dict[str, List[str]] = {}
    reverse_groups = {field: group for group, field in _REMOVE_GENERATED_KEYS.items()}
    for field in GENERATED_COMPONENT_FIELDS:
        effective_names = set(_rows_by_name(effective.get(field)))
        names = [
            name for name in _rows_by_name(base.get(field))
            if name not in effective_names
        ]
        if names:
            removed[reverse_groups[field]] = names
    if removed:
        out["remove_generated"] = removed

    if effective.get("remove_tools", []) != base.get("remove_tools", []):
        out["remove_tools"] = list(effective.get("remove_tools", []))
    return out


def generated_component_lineage(
    base: Dict[str, Any], effective: Dict[str, Any],
) -> Dict[str, List[Dict[str, str]]]:
    """Label every materialized component as inherited, added, or updated."""

    lineage: Dict[str, List[Dict[str, str]]] = {}
    for field in GENERATED_COMPONENT_FIELDS:
        base_by_name = _rows_by_name(base.get(field))
        rows = []
        effective_names = set()
        for row in effective.get(field, []):
            name = str(row["name"])
            effective_names.add(name)
            if name not in base_by_name:
                status = "added"
            elif base_by_name[name] == row:
                status = "inherited"
            else:
                status = "updated"
            rows.append({"name": name, "status": status})
        for row in base.get(field, []):
            name = str(row["name"])
            if name not in effective_names:
                rows.append({"name": name, "status": "removed"})
        lineage[field] = rows
    return lineage


# --------------------------------------------------------------------------- #
# Cordis package extraction. ``cordis.yml`` is the composition source of truth;
# provenance JSON is only an audit cross-check and can never resurrect an
# unmounted plugin.
# --------------------------------------------------------------------------- #
_DEFAULT_TOOL_DESCRIPTIONS = {
    "edit_solution": "Change code inside the EVOLVE-BLOCK, then evaluate it.",
    "evaluate_solution": "Run the authoritative evaluator on the current program.",
    "probe_solution": "Cheap approximate score; confirm finalists with evaluate_solution.",
    "finish": "End the session and retain the best program.",
}


def _cordis_entries(document: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(document, list):
        raise ValueError("cordis.yml must be a YAML sequence")
    rows: List[Dict[str, Any]] = []
    inserted: List[Dict[str, Any]] = []
    for index, row in enumerate(document):
        if not isinstance(row, dict):
            raise ValueError(f"cordis.yml[{index}] must be a mapping")
        if "insert" in row:
            if set(row) != {"insert"} or not isinstance(row["insert"], list):
                raise ValueError(f"cordis.yml[{index}].insert must be the only key and a list")
            for inner in row["insert"]:
                if not isinstance(inner, dict):
                    raise ValueError("cordis.yml insert entries must be mappings")
                inserted.append(inner)
        else:
            rows.append(row)
    return rows, inserted


def _one_cordis_row(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    matched = [row for row in rows if row.get("id") == row_id]
    if len(matched) != 1:
        raise ValueError(f"cordis.yml must contain exactly one {row_id!r} row")
    return matched[0]


def _plugin_path(package_dir: Path, row: Dict[str, Any]) -> Path:
    value = row.get("name")
    if not isinstance(value, str) or not re.fullmatch(r"\./plugins/[a-z0-9][a-z0-9_-]*\.mjs", value):
        raise ValueError(f"Cordis plugin name must be ./plugins/<slug>.mjs, got {value!r}")
    path = package_dir / value[2:]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"mounted Cordis plugin is missing or unsafe: {value}")
    return path


def read_base_spec(
    package_dir: Path, *, verify_provenance: bool = True,
) -> Dict[str, Any]:
    """Extract one complete ``h2spec/2.0-cordis`` from a Cordis package."""
    package_dir = Path(package_dir)
    composition_path = package_dir / "cordis.yml"
    if not composition_path.is_file():
        raise ValueError(f"Cordis H2 package has no cordis.yml: {package_dir}")
    document = yaml.safe_load(composition_path.read_text(encoding="utf-8"))
    rows, inserted = _cordis_entries(document)
    system = _one_cordis_row(rows, "system-prompt")
    bridge = _one_cordis_row(rows, "sah-bridge")
    if system.get("disabled") is True or bridge.get("disabled") is True:
        raise ValueError("system-prompt and sah-bridge must remain enabled")
    system_config = system.get("config") or {}
    bridge_config = bridge.get("config") or {}
    if not isinstance(system_config, dict) or not isinstance(bridge_config, dict):
        raise ValueError("Cordis core row config values must be mappings")
    persona = system_config.get("persona")
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("system-prompt.config.persona must be non-empty")

    base_skills = []
    for row in inserted:
        sah = ((row.get("config") or {}).get("sah") or {}) \
            if isinstance(row.get("config") or {}, dict) else {}
        if isinstance(sah, dict) and sah.get("kind") == "skill" and sah.get("base") is True:
            base_skills.append((row, sah))
    if len(base_skills) != 1 or base_skills[0][1].get("name") != "discovery-optimization":
        raise ValueError("cordis.yml must mount exactly one base discovery-optimization skill")
    _plugin_path(package_dir, base_skills[0][0])
    base_skill = base_skills[0][1]
    skill_body = base_skill.get("body")
    if not isinstance(skill_body, str) or not skill_body.strip():
        raise ValueError("base skill metadata must include non-empty config.sah.body")

    tool_descriptions = dict(_DEFAULT_TOOL_DESCRIPTIONS)
    configured_descriptions = bridge_config.get("toolDescriptions") or {}
    if not isinstance(configured_descriptions, dict):
        raise ValueError("sah-bridge.config.toolDescriptions must be a mapping")
    tool_descriptions.update({str(k): str(v) for k, v in configured_descriptions.items()})
    disabled = bridge_config.get("disabledTools") or []
    if not isinstance(disabled, list) or any(name not in {"probe_solution"} for name in disabled):
        raise ValueError("sah-bridge.config.disabledTools may contain only probe_solution")

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "system_prompt": persona.strip(),
        "skill_description": str(base_skill.get("description") or "").strip(),
        "skill_body": skill_body.strip(),
        "tool_descriptions": tool_descriptions,
        "sampling": {
            "temperature": float(bridge_config.get("temperature", 0.7)),
            "top_p": float(bridge_config.get("topP", 0.95)),
            "top_k": int(bridge_config.get("topK", 20)),
            "max_tokens": int(bridge_config.get("maxTokens", 8192)),
        },
        "agent": {"max_iterations": int(bridge_config.get("maxIterations", 36))},
        "middleware": {
            "budget_reminder_from_left": int(bridge_config.get("budgetReminderFromLeft", 3)),
            "long_tool_output_max_chars": int(bridge_config.get("maxOutputChars", 8000)),
            "stall_after": int(bridge_config.get("stallAfter", 8)),
            "max_restarts": int(bridge_config.get("maxRestarts", 2)),
        },
        **({"remove_tools": sorted(set(disabled))} if disabled else {}),
    }
    out.update(_read_generated(
        package_dir, inserted, verify_provenance=verify_provenance
    ))

    # Reuse the same strict shape/range checks used for H1 submissions. A full
    # package necessarily mutates many fields, so the partial-spec no-op rule is
    # satisfied naturally.
    checked = parse_and_validate(yaml.safe_dump(out, sort_keys=False))
    if not checked.valid:
        raise ValueError("invalid Cordis H2 genome: " + "; ".join(checked.errors))
    return out


def _read_generated(
    package_dir: Path, inserted: List[Dict[str, Any]], *,
    verify_provenance: bool = True,
) -> Dict[str, Any]:
    """Recover generated components from actually mounted Cordis plugins."""
    package_dir = Path(package_dir)
    out: Dict[str, Any] = {}
    tools: List[Dict[str, Any]] = []
    skills: List[Dict[str, Any]] = []
    middlewares: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_components: set[Tuple[str, str]] = set()

    for row in inserted:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("every inserted Cordis plugin needs a non-empty id")
        if row_id in seen_ids:
            raise ValueError(f"duplicate Cordis plugin id: {row_id}")
        seen_ids.add(row_id)
        config = row.get("config") or {}
        sah = config.get("sah") if isinstance(config, dict) else None
        if not isinstance(sah, dict):
            raise ValueError(f"inserted plugin {row_id!r} lacks config.sah metadata")
        kind, name = sah.get("kind"), sah.get("name")
        if kind not in {"tool", "skill", "middleware"} or not isinstance(name, str):
            raise ValueError(f"inserted plugin {row_id!r} has invalid config.sah kind/name")
        path = _plugin_path(package_dir, row)
        source = path.read_text(encoding="utf-8")
        if sah.get("base") is True:
            continue
        key = (kind, name)
        if key in seen_components:
            raise ValueError(f"duplicate generated {kind} name: {name}")
        seen_components.add(key)
        if kind == "tool":
            schema = sah.get("inputSchema") or {"type": "object", "properties": {}}
            tools.append({
                "name": name,
                "description": str(sah.get("description") or "").strip(),
                "input_schema": schema,
                "implementation_js": source.rstrip() + "\n",
            })
        elif kind == "skill":
            body = sah.get("body")
            if not isinstance(body, str):
                raise ValueError(f"generated skill {name!r} metadata lacks body")
            skills.append({
                "name": name,
                "description": str(sah.get("description") or "").strip(),
                "body": body.strip(),
            })
        else:
            middlewares.append({
                "name": name,
                "hook": str(sah.get("hook") or ""),
                "description": str(sah.get("description") or "").strip(),
                "implementation_js": source.rstrip() + "\n",
            })
    if tools:
        out["new_tools"] = tools
    if skills:
        out["new_skills"] = skills
    if middlewares:
        out["new_middlewares"] = middlewares

    actual = generated_component_inventory(out)
    meta_rows = {field: {} for field in GENERATED_COMPONENT_FIELDS}
    meta_path = package_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            effective = meta.get("effective") if isinstance(meta, dict) else None
            if isinstance(effective, dict):
                meta_rows = {
                    field: _rows_by_name(effective.get(field))
                    for field in GENERATED_COMPONENT_FIELDS
                }
        except Exception as exc:
            raise ValueError(f"invalid package meta.json: {exc}") from exc
    if verify_provenance:
        for field in GENERATED_COMPONENT_FIELDS:
            recorded = list(meta_rows[field])
            if recorded and recorded != actual[field]:
                raise ValueError(
                    f"package component mismatch for {field}: meta={recorded}, mounted={actual[field]}"
                )
        manifest_path = package_dir / "component_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("inventory") or {}) != actual:
                raise ValueError(
                    "component_manifest.json does not match mounted Cordis package"
                )
    return out


def merge_with_base(spec: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a (validated, partial) spec over the base spec -> full effective spec."""
    eff = json.loads(json.dumps(base))  # deep copy
    for key in _TEXT_FIELDS:
        if key in spec:
            eff[key] = spec[key]
    if "tool_descriptions" in spec:
        eff.setdefault("tool_descriptions", {}).update(spec["tool_descriptions"])
    for group in ("sampling", "agent", "middleware"):
        if group in spec:
            eff.setdefault(group, {}).update(spec[group])
    # Generated components are a ratcheted, name-keyed genome.  New names are
    # appended; reusing a name explicitly updates it; omission inherits it.
    for field in GENERATED_COMPONENT_FIELDS:
        if field in spec:
            merged = merge_named_components(base.get(field), spec[field])
            if merged:
                eff[field] = merged
    for group, field in _REMOVE_GENERATED_KEYS.items():
        removed = set((spec.get("remove_generated") or {}).get(group, []))
        if not removed:
            continue
        kept = [
            row for row in eff.get(field, [])
            if str(row.get("name")) not in removed
        ]
        if kept:
            eff[field] = kept
        else:
            eff.pop(field, None)
    if "remove_tools" in spec:
        eff["remove_tools"] = spec["remove_tools"]
    eff["schema"] = SCHEMA_VERSION
    return eff


def differs_from_base(effective: Dict[str, Any], base: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Which fields does the effective spec actually change vs the base?"""
    changed: List[str] = []
    for key in _TEXT_FIELDS:
        if effective.get(key, "").strip() != base.get(key, "").strip():
            changed.append(key)
    for name in _TOOL_DESC_FIELDS:
        if (effective.get("tool_descriptions", {}).get(name, "").strip()
                != base.get("tool_descriptions", {}).get(name, "").strip()):
            changed.append(f"tool_descriptions.{name}")
    for group in ("sampling", "agent", "middleware"):
        for key, val in effective.get(group, {}).items():
            if val != base.get(group, {}).get(key):
                changed.append(f"{group}.{key}")
    for field in GENERATED_COMPONENT_FIELDS:
        for row in changed_generated_components(effective, base, field):
            changed.append(f"{field}.{row.get('name', '?')}")
        effective_names = set(_rows_by_name(effective.get(field)))
        for name in _rows_by_name(base.get(field)):
            if name not in effective_names:
                changed.append(f"{field}.{name}.removed")
    if effective.get("remove_tools", []) != base.get("remove_tools", []):
        changed.append("remove_tools")
    return (len(changed) > 0, changed)

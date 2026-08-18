"""Deterministically compile an H2 genome into a native Cordis package."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from outer.genome.harness_spec import (
    SCHEMA_VERSION,
    component_prompt_issues,
    generated_component_inventory,
)


INNER_HARNESS = Path(__file__).resolve().parents[2] / "inner" / "harness"


class _Literal(str):
    pass


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _Literal,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(value), style="|",
    ),
)


def _literalize(value: Any) -> Any:
    if isinstance(value, str) and "\n" in value:
        return _Literal(value)
    if isinstance(value, list):
        return [_literalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _literalize(item) for key, item in value.items()}
    return value


def _yaml(data: Any) -> str:
    return yaml.dump(
        _literalize(data), Dumper=_Dumper, sort_keys=False,
        allow_unicode=True, width=100, default_flow_style=False,
    )


def plugin_filename(kind: str, name: str) -> str:
    """Canonical flat plugin name used by the DSH profile snapshotter."""
    prefix = {"tool": "tool", "skill": "skill", "middleware": "middleware"}[kind]
    return f"sah-{prefix}-{name}.mjs"


def render_skill_plugin(name: str, body: str, *, order: int = 20) -> str:
    """Render a text genome as an automatically enacted prompt plugin."""
    plugin_name = f"sah-skill-{name}"
    return (
        "/** SAH Cordis skill plugin; generated deterministically. */\n"
        f"export const name = {json.dumps(plugin_name)}\n"
        "export const inject = ['systemPrompt']\n\n"
        "export function apply(ctx) {\n"
        "  ctx.systemPrompt.section({\n"
        f"    name: {json.dumps('sah:skill:' + name)},\n"
        f"    order: {int(order)},\n"
        f"    text: {json.dumps(body.strip(), ensure_ascii=False)},\n"
        "  })\n"
        "}\n"
    )


def _sah_row(kind: str, name: str, *, description: str = "",
             source_name: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "kind": kind,
        "name": name,
        "description": description,
    }
    metadata.update(extra or {})
    file_name = source_name or plugin_filename(kind, name)
    return {
        "id": f"sah-{kind}-{name}",
        "name": f"./plugins/{file_name}",
        "config": {"sah": metadata},
    }


def _cordis_document(effective: Dict[str, Any]) -> list[dict[str, Any]]:
    sampling = effective.get("sampling") or {}
    agent = effective.get("agent") or {}
    middleware = effective.get("middleware") or {}
    bridge: Dict[str, Any] = {
        "maxIterations": int(agent.get("max_iterations", 36)),
        "maxOutputChars": int(middleware.get("long_tool_output_max_chars", 8000)),
        "temperature": float(sampling.get("temperature", 0.7)),
        "topP": float(sampling.get("top_p", 0.95)),
        "topK": int(sampling.get("top_k", 20)),
        "maxTokens": int(sampling.get("max_tokens", 8192)),
        "budgetReminderFromLeft": int(middleware.get("budget_reminder_from_left", 3)),
        "stallAfter": int(middleware.get("stall_after", 8)),
        "maxRestarts": int(middleware.get("max_restarts", 2)),
        "toolDescriptions": dict(effective.get("tool_descriptions") or {}),
        "disabledTools": sorted(set(effective.get("remove_tools") or [])),
        "generatedTools": [
            row["name"] for row in effective.get("new_tools") or []
        ],
        "generatedMiddlewares": [
            row["name"] for row in effective.get("new_middlewares") or []
        ],
    }

    inserted: list[dict[str, Any]] = [
        _sah_row(
            "skill", "discovery-optimization",
            description=str(effective.get("skill_description") or ""),
            source_name="discovery-optimization.mjs",
            extra={
                "base": True,
                "body": str(effective.get("skill_body") or "").strip(),
            },
        )
    ]
    for tool in effective.get("new_tools") or []:
        inserted.append(_sah_row(
            "tool", tool["name"], description=tool.get("description", ""),
            extra={"inputSchema": tool.get("input_schema") or {
                "type": "object", "properties": {},
            }},
        ))
    for index, skill in enumerate(effective.get("new_skills") or [], start=1):
        inserted.append(_sah_row(
            "skill", skill["name"], description=skill.get("description", ""),
            extra={"body": skill["body"], "order": 20 + index},
        ))
    for middleware_row in effective.get("new_middlewares") or []:
        inserted.append(_sah_row(
            "middleware", middleware_row["name"],
            description=middleware_row.get("description", ""),
            extra={"hook": middleware_row.get("hook", "agent/pre-step")},
        ))

    return [
        {
            "id": "system-prompt",
            "config": {
                "includeHarnessIdentity": False,
                "includeRuntimeContext": False,
                "persona": str(effective["system_prompt"]).strip(),
            },
        },
        {"id": "sah-bridge", "config": bridge},
        {"insert": inserted},
    ]


def materialize(effective: Dict[str, Any], cand_dir: Path, *,
                raw_spec_text: str = "", meta: Optional[Dict[str, Any]] = None,
                validate_prompt: bool = True) -> Path:
    """Write a full, relocatable Cordis H2 package from a full merged spec."""
    if effective.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"materialize expects {SCHEMA_VERSION}, got {effective.get('schema')!r}"
        )
    prompt_issues = component_prompt_issues(effective)
    if validate_prompt and prompt_issues:
        raise ValueError(
            "invalid proposer-owned Cordis persona: " + "; ".join(prompt_issues)
        )

    # Safety is checked both in the workspace validator and here so callers
    # cannot bypass it by constructing an effective spec directly.
    from outer.safety.static_gates import check_cordis_plugin
    for row in effective.get("new_tools") or []:
        ok, errors = check_cordis_plugin(
            row["implementation_js"], kind="tool", name=row["name"],
        )
        if not ok:
            raise ValueError(
                f"generated tool {row['name']!r} failed Cordis gate: "
                + "; ".join(errors)
            )
    for row in effective.get("new_middlewares") or []:
        ok, errors = check_cordis_plugin(
            row["implementation_js"], kind="middleware", name=row["name"],
            hook=row.get("hook", "agent/pre-step"),
        )
        if not ok:
            raise ValueError(
                f"generated middleware {row['name']!r} failed Cordis gate: "
                + "; ".join(errors)
            )

    cand_dir = Path(cand_dir)
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    plugins = cand_dir / "plugins"
    plugins.mkdir(parents=True)

    # Base and generated skills are deterministic prompt plugins; tool and
    # middleware sources are the exact Cordis plugins predicted by H1.
    (plugins / "discovery-optimization.mjs").write_text(
        render_skill_plugin(
            "discovery-optimization", str(effective.get("skill_body") or ""),
            order=20,
        ),
        encoding="utf-8",
    )
    for row in effective.get("new_tools") or []:
        (plugins / plugin_filename("tool", row["name"])).write_text(
            row["implementation_js"].rstrip() + "\n", encoding="utf-8",
        )
    for index, row in enumerate(effective.get("new_skills") or [], start=1):
        (plugins / plugin_filename("skill", row["name"])).write_text(
            render_skill_plugin(row["name"], row["body"], order=20 + index),
            encoding="utf-8",
        )
    for row in effective.get("new_middlewares") or []:
        (plugins / plugin_filename("middleware", row["name"])).write_text(
            row["implementation_js"].rstrip() + "\n", encoding="utf-8",
        )

    document = _cordis_document(effective)
    (cand_dir / "cordis.yml").write_text(_yaml(document), encoding="utf-8")

    expected_inventory = generated_component_inventory(effective)
    if raw_spec_text:
        (cand_dir / "spec.yaml").write_text(
            raw_spec_text.rstrip() + "\n", encoding="utf-8",
        )
    meta_payload = dict(meta or {})
    meta_payload.setdefault("effective", effective)
    meta_payload["component_inventory"] = expected_inventory
    (cand_dir / "component_manifest.json").write_text(
        json.dumps({
            "schema": "generated-components/2.0-cordis",
            "inventory": expected_inventory,
            "lineage": meta_payload.get("component_lineage", {}),
        }, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (cand_dir / "meta.json").write_text(
        json.dumps(meta_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cand_dir

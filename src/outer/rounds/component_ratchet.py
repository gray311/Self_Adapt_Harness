"""COMPRATCHET-001: selection at the COMPONENT level, with evidence.

The program ratchet promotes whole harnesses, so a component with a positive
ISOLATED causal effect dies whenever its candidate loses the round — measured
across campaigns: the one skill with Δ+0.0439 never reached the next round's
base.  Evolution needs the missing operator: components are admitted into the
next round's base on their own merit, independently of who won.

Admission contract (every clause checkable from artifacts):
  * the candidate was a COMPONENT-ONLY slot — the incumbent prompt was
    inherited byte-for-byte, so its paired delta is attributable to the
    component alone;
  * the pair passed attribution (``attribution_valid``);
  * ``causal_delta >= CURVE_COMPONENT_RATCHET_MIN_DELTA`` (default 1e-4);
  * capacity: the genome caps per type bound the heritable library
    (tools 3 / skills 2 / middlewares 2); incumbent-carried components are
    never evicted — overflow admissions are recorded as skipped;
  * the grafted base keeps every naming invariant (declaration lines are
    appended; the prompt cap is respected or the admission is skipped).

Every admission/skip is appended to ``rounds/component_ledger.jsonl`` with
the full evidence chain (round, slot, candidate dir, delta, control/candidate
scores, authored-by), and ``next_bases.json`` records what was grafted, from
where, and why — so the evolutionary history is reconstructible after the
fact.  ``CURVE_COMPONENT_RATCHET=0`` disables the operator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from outer.genome import harness_spec as hs
from outer.compiling.materialize import materialize

_KIND_FIELDS = {
    "new_tools": "tool",
    "new_skills": "skill",
    "new_middlewares": "middleware",
}
# The fifth dimension: an operating-point move (sampling / agent budget /
# built-in middleware policy) measured in isolation is exactly as heritable
# as a component — it is a scalar field, so admission overwrites it on the
# next base and the highest-delta proposal wins a contested field.
_PARAM_GROUPS = ("sampling", "agent", "middleware")
_TYPE_CAPS = {"new_tools": 3, "new_skills": 2, "new_middlewares": 2}
_PROMPT_CAP = 10000
_DECLARATIONS = {
    "new_tools": "\n# Generated Tool: {name}\nDecide its trigger before the "
                 "first edit; call it when the trigger holds.\n",
    "new_skills": "\n# Generated Skill: {name}\nEnact the {name} skill "
                  "before the first edit.\n",
    "new_middlewares": "\n# Generated Middleware: {name}\nRuns automatically "
                       "before each model call; no tool call is needed.\n",
}


def _component_entries(spec: Dict[str, Any], names: set) -> List[Dict[str, Any]]:
    out = []
    for field in _KIND_FIELDS:
        for entry in (spec.get(field) or []):
            if isinstance(entry, dict) and str(entry.get("name")) in names:
                out.append({"field": field, "entry": entry})
    return out


def apply_component_ratchet(
    *,
    round_dir: Path,
    tid: str,
    round_index: int,
    candidates: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    improved: bool,
    best_k: Optional[int],
    next_base_entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Graft admissible components into the next base; mutate the entry.

    Returns the ratchet record (also stored on the entry) or None when the
    operator is disabled or nothing was admissible.
    """
    if os.environ.get("CURVE_COMPONENT_RATCHET", "1") != "1":
        return None
    min_delta = float(
        os.environ.get("CURVE_COMPONENT_RATCHET_MIN_DELTA", "1e-4") or 1e-4)

    row_by_k = {int(r["k"]): r for r in rows}
    admissible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for cand in candidates:
        k = int(cand.get("k", -1))
        if not cand.get("valid"):
            continue
        comp_names = {
            str(f).split(".", 1)[1] for f in (cand.get("changed_fields") or [])
            if str(f).split(".", 1)[0] in _KIND_FIELDS
        }
        param_fields = [
            str(f) for f in (cand.get("changed_fields") or [])
            if str(f).split(".", 1)[0] in _PARAM_GROUPS and "." in str(f)
        ]
        if not comp_names and not param_fields:
            continue
        row = row_by_k.get(k) or {}
        delta = row.get("causal_delta")
        evidence = {
            "round": round_index, "k": k,
            "candidate": str(round_dir / "tasks" / tid / f"cand{k:02d}"),
            "causal_delta": delta,
            "score": row.get("score"),
            "control_score": row.get("control_score"),
            "authored_by": ("repair" if cand.get("stop_reason") ==
                            "repaired_submission" else "proposer"),
        }
        if improved and best_k is not None and k == int(best_k):
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "already_in_promoted"})
            continue
        if not cand.get("component_only"):
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "not_isolated"})
            continue
        # ISOLATION-002: component_only is a SLOT LABEL decided before H1
        # runs; the session only caps prompt.md, so such a slot can still
        # rewrite skill_body (8000 chars), tool_descriptions or remove_tools.
        # Crediting the whole paired delta to the component would install a
        # possibly-inert component and record a false evidence chain, so the
        # admission requires the change set to contain NOTHING but the
        # admitted items (a bounded declaration line aside).
        confounders = sorted(
            str(f) for f in (cand.get("changed_fields") or [])
            if str(f) != "system_prompt"
            and str(f).split(".", 1)[0] not in _KIND_FIELDS
            and str(f).split(".", 1)[0] not in _PARAM_GROUPS
        )
        if confounders:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "delta_confounded_by:" + ",".join(confounders)})
            continue
        if not row.get("attribution_valid"):
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": "attribution_invalid"})
            continue
        if delta is None or float(delta) < min_delta:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": f"delta_below_{min_delta}"})
            continue
        cand_dir = Path(evidence["candidate"])
        try:
            cand_spec = hs.read_base_spec(cand_dir)
        except Exception as exc:
            skipped.append({**evidence,
                            "names": sorted(comp_names) + param_fields,
                            "reason": f"unreadable_candidate: {exc}"})
            continue
        for item in _component_entries(cand_spec, comp_names):
            admissible.append({**evidence,
                               "field": item["field"],
                               "name": str(item["entry"].get("name")),
                               "entry": item["entry"]})
        for field in param_fields:
            group, key = field.split(".", 1)
            value = (cand_spec.get(group) or {}).get(key)
            if value is None:
                continue
            admissible.append({**evidence, "field": group, "name": key,
                               "param_value": value})

    record: Dict[str, Any] = {"admitted": [], "skipped": skipped}
    if admissible:
        admissible.sort(key=lambda a: -float(a["causal_delta"]))
        base_pkg = Path(next_base_entry["package"])
        base_spec = hs.read_base_spec(base_pkg)
        merged = json.loads(json.dumps(base_spec))  # deep copy, JSON-safe
        prompt = str(merged.get("system_prompt") or "")
        existing = {
            field: {str(e.get("name")) for e in (merged.get(field) or [])
                    if isinstance(e, dict)}
            for field in _KIND_FIELDS
        }
        admitted_params: set = set()
        for adm in admissible:
            field, name = adm["field"], adm["name"]
            public = {k: v for k, v in adm.items() if k != "entry"}
            if field in _PARAM_GROUPS:
                current = (merged.get(field) or {}).get(name)
                if (field, name) in admitted_params:
                    # admissible is sorted by DESCENDING delta, so the first
                    # writer is the strongest proposal; without this guard a
                    # weaker later one overwrote it (last-write-wins) while
                    # both were recorded as admitted.
                    skipped.append({**public,
                                    "reason": "param_contested_lower_delta"})
                    continue
                if current == adm["param_value"]:
                    skipped.append({**public, "reason": "param_already_set"})
                    continue
                admitted_params.add((field, name))
                merged.setdefault(field, {})
                merged[field][name] = adm["param_value"]
                public["previous_value"] = current
                record["admitted"].append(public)
                continue
            if name in existing[field]:
                skipped.append({**public, "reason": "name_already_in_base"})
                continue
            if len(existing[field]) >= _TYPE_CAPS[field]:
                skipped.append({**public, "reason": "capacity_full"})
                continue
            declaration = _DECLARATIONS[field].format(name=name)
            if name not in prompt and \
                    len(prompt) + len(declaration) > _PROMPT_CAP - 200:
                skipped.append({**public, "reason": "prompt_cap"})
                continue
            merged.setdefault(field, [])
            merged[field] = list(merged[field]) + [adm["entry"]]
            if name not in prompt:
                prompt = prompt.rstrip("\n") + "\n" + declaration
            existing[field].add(name)
            record["admitted"].append(public)

        if record["admitted"]:
            merged["system_prompt"] = prompt
            graft_dir = round_dir / "grafted_bases" / tid
            materialize(merged, graft_dir, meta={"effective": merged},
                        validate_prompt=False)
            record["previous_package"] = str(base_pkg)
            record["grafted_package"] = str(graft_dir)
            record["spec_hash"] = hs.spec_hash(merged)
            next_base_entry["package"] = str(graft_dir)
            next_base_entry["score_basis"] = "pre_graft"
            next_base_entry["component_ratchet"] = {
                "admitted": record["admitted"],
                "skipped_count": len(skipped),
                "previous_package": str(base_pkg),
            }
            beam = next_base_entry.get("beam")
            if beam:
                beam[0] = dict(beam[0])
                beam[0]["package"] = str(graft_dir)
                beam[0]["from"] = beam[0].get("from", "incumbent") + "+graft"

    if record["admitted"] or skipped:
        ledger = round_dir.parent / "component_ledger.jsonl"
        with ledger.open("a") as fh:
            for kind, items in (("admitted", record["admitted"]),
                                ("skipped", skipped)):
                for item in items:
                    fh.write(json.dumps({
                        "event": kind, "task_id": tid,
                        "round": round_index, **item,
                    }) + "\n")
    return record if (record["admitted"] or skipped) else None

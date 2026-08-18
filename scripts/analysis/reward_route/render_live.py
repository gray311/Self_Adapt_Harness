#!/usr/bin/env python3
"""Render the inference16 effect ledger as a self-contained live HTML audit."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROUTES = (
    ("proposer", "Update proposer weights", "#1769aa", "circle"),
    ("context", "Analyzer/context only", "#6f6f6f", "triangle"),
    ("executor", "Update executor weights", "#e67e22", "square"),
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def marker(shape: str, x: float, y: float, color: str) -> str:
    if shape == "triangle":
        pts = f"{x:.1f},{y-4:.1f} {x-4:.1f},{y+3.5:.1f} {x+4:.1f},{y+3.5:.1f}"
        return f'<polygon points="{pts}" fill="white" stroke="{color}" stroke-width="2"/>'
    if shape == "square":
        return f'<rect x="{x-3.5:.1f}" y="{y-3.5:.1f}" width="7" height="7" fill="white" stroke="{color}" stroke-width="2"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="white" stroke="{color}" stroke-width="2"/>'


def chart(task: dict[str, Any]) -> str:
    width, height = 900, 340
    left, right, top, bottom = 74, 22, 24, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    series = {}
    ys = []
    for key, _, _, _ in ROUTES:
        route = task["routes"][key]
        points = [(1, float(route["anchor"]["normalized"]))]
        points.extend(
            (int(row["x_after"]), float(row["score_after_normalized"]))
            for row in route["batches"]
        )
        series[key] = points
        ys.extend(y for _, y in points)
    low, high = min(ys), max(ys)
    span = high - low
    pad = max(span * 0.12, max(abs(low), abs(high), 1.0) * 0.004)
    if span < 1e-12:
        pad = max(abs(low) * 0.02, 0.01)
    y0, y1 = low - pad, high + pad
    # Include human best only when it is close enough not to flatten the method
    # differences. Otherwise an arrow annotation reports where y=1 lies.
    human_visible = y0 <= 1.0 <= y1 or abs(1.0 - (low + high) / 2) <= 1.5 * (y1 - y0)
    if human_visible:
        y0, y1 = min(y0, 1.0 - pad * 0.2), max(y1, 1.0 + pad * 0.2)

    def px(x: float) -> float:
        return left + math.log10(x) / math.log10(305) * plot_w

    def py(y: float) -> float:
        return top + (y1 - y) / max(y1 - y0, 1e-12) * plot_h

    out = [f'<svg class="curve" viewBox="0 0 {width} {height}" role="img">']
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff"/>')
    for i in range(5):
        value = y0 + i * (y1 - y0) / 4
        y = py(value)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.4f}</text>')
    for value in (1, 17, 65, 129, 193, 257, 305):
        x = px(value)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="tick">{value}</text>')
    if human_visible:
        y = py(1.0)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="human"/>')
        out.append(f'<text x="{left+8}" y="{y-7:.1f}" class="human-label">human best</text>')
    elif 1.0 > y1:
        out.append(f'<text x="{left+plot_w-4}" y="{top+13}" text-anchor="end" class="human-label">↑ human best = 1.0 (outside zoom)</text>')
    else:
        out.append(f'<text x="{left+plot_w-4}" y="{top+plot_h-8}" text-anchor="end" class="human-label">↓ human best = 1.0 (outside zoom)</text>')
    for key, _, color, shape in ROUTES:
        points = series[key]
        coords = [(px(x), py(y)) for x, y in points]
        if len(coords) > 1:
            out.append('<polyline fill="none" stroke="{}" stroke-width="2.6" points="{}"/>'.format(
                color, " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            ))
        out.extend(marker(shape, x, y, color) for x, y in coords)
    out.append(f'<text x="{left+plot_w/2:.1f}" y="{height-10}" text-anchor="middle" class="axis-label">cumulative generated agent trajectories (log)</text>')
    out.append(f'<text transform="translate(17 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="axis-label">validated score / human best</text>')
    out.append('</svg>')
    return "".join(out)


def route_summary(route: dict[str, Any]) -> str:
    batches = route["batches"]
    if not batches:
        return "No adaptive batch has completed."
    last = batches[-1]
    applied = sum(row["outgoing_update"].get("applied") is True for row in batches)
    skipped = sum(
        row["outgoing_update"].get("opportunity")
        and row["outgoing_update"].get("eligible") is False
        for row in batches
    )
    return (
        f'{len(batches)}/19 batches · x={last["x_after"]} · '
        f'normalized best={last["score_after_normalized"]:.5f} · '
        f'{applied} evaluated updates · {skipped} skipped updates'
    )


def effect_table(task: dict[str, Any]) -> str:
    rows = []
    for key, label, _, _ in ROUTES:
        for batch in task["routes"][key]["batches"]:
            update = batch["outgoing_update"]
            nxt = batch.get("next_batch_observation")
            bundle = (batch.get("trajectory_bundle") or {}).get("sha256")
            next_cell = f'<td>{nxt["normalized_delta"]:+.6f}</td>' if nxt else '<td>—</td>'
            row_html = (
                '<tr>'
                f'<td>{esc(label)}</td><td>{batch["batch_index"]}</td>'
                f'<td>{batch["x_after"]}</td>'
                f'<td>{batch["score_after_normalized"]:.6f}</td>'
                f'<td>{batch["search_gain"] / task["human_best_combined_score"]:+.6f}</td>'
                f'<td>{esc(update["target"])}</td>'
                f'<td>{esc(update.get("applied"))}</td>'
                f'{next_cell}'
            )
            rows.append(row_html + f'<td><code>{esc((bundle or "missing")[:12])}</code></td></tr>')
    if not rows:
        return '<p class="empty">Effect rows will appear after the first adaptive batch.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr><th>Route</th><th>Batch</th>'
        '<th>x</th><th>Best / human</th><th>Search gain</th><th>Updated state</th>'
        '<th>Seen next batch</th><th>Next-batch Δ</th><th>Trajectory hash</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path,
        default=REPO / "results" / "reward_route_inference16_effects.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "results" / "artifacts_reward_route_inference16" / "index.html",
    )
    args = parser.parse_args()
    data = json.loads(args.ledger.read_text())
    status = data["status"]
    task_sections = []
    for task_id, task in data["tasks"].items():
        route_cards = ''.join(
            f'<div class="route"><span class="swatch" style="background:{color}"></span>'
            f'<strong>{esc(label)}</strong><small>{esc(route_summary(task["routes"][key]))}</small></div>'
            for key, label, color, _ in ROUTES
        )
        task_sections.append(
            f'<section><div class="section-head"><div><span class="eyebrow">{esc(task["tag"])}</span>'
            f'<h2>{esc(task["title"])}</h2></div><code>{esc(task_id)}</code></div>'
            f'<div class="routes">{route_cards}</div>{chart(task)}'
            '<details><summary>Per-batch update/effect ledger</summary>'
            f'{effect_table(task)}</details></section>'
        )
    progress = data["route_progress"]
    warning = (
        "Canonical run has not started; only the shared x=1 anchors are visible."
        if status == "not_started" else
        ("Live preview only—missing batches are never interpolated or imported from historical runs."
         if status != "complete" else
         "All 12 task-route lineages have reached the common x=305 endpoint.")
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference-16 effect audit</title><style>
:root{{--ink:#172033;--muted:#667085;--line:#dce3eb;--paper:#f5f7fa;--blue:#1769aa;--orange:#e67e22}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:42px 24px 80px}}header{{background:#101827;color:white;border-radius:20px;padding:30px 34px;box-shadow:0 16px 40px #1720331c}}
.eyebrow{{font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#75b8ec}}h1{{font-size:32px;margin:5px 0 8px}}h2{{font-size:24px;margin:2px 0}}p{{margin:6px 0}}.status{{display:inline-block;margin-top:14px;padding:6px 11px;border-radius:99px;background:#f59e0b;color:#111827;font-weight:800}}
.warning{{margin:20px 0;background:#fff8e6;border:1px solid #f2c76e;border-radius:12px;padding:13px 16px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}.stat,.route{{background:white;border:1px solid var(--line);border-radius:12px;padding:13px 15px}}.stat strong{{display:block;font-size:22px}}.stat small,.route small{{display:block;color:var(--muted)}}
section{{background:white;border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:18px;box-shadow:0 8px 24px #1720330c}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:20px}}.section-head code{{color:var(--muted);font-size:12px}}.routes{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}}.swatch{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}}
.curve{{width:100%;height:auto}}.grid{{stroke:#e7ebf0;stroke-width:1}}.tick{{font-size:11px;fill:#697386}}.axis-label{{font-size:12px;fill:#344054}}.human{{stroke:#384250;stroke-width:1.3;stroke-dasharray:4 4}}.human-label{{font-size:11px;fill:#384250;font-weight:700}}
details{{border-top:1px solid var(--line);padding-top:14px}}summary{{cursor:pointer;font-weight:750}}.table-wrap{{overflow:auto;margin-top:12px}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{border-bottom:1px solid #e8edf2;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}.empty{{color:var(--muted)}}
@media(max-width:850px){{.stats,.routes{{grid-template-columns:1fr}}.section-head{{align-items:start;flex-direction:column}}}}
</style></head><body><main>
<header><span class="eyebrow">reward-route-inference16-v1</span><h1>Trajectory-bound adaptation audit</h1>
<p>Four tasks, three update targets, 16 generated agent trajectories per batch, common endpoint x=305.</p>
<span class="status">{esc(status.replace('_',' '))}</span></header>
<div class="warning">{esc(warning)}</div>
<div class="stats"><div class="stat"><strong>{progress['complete']}/{progress['total']}</strong><small>complete routes</small></div>
<div class="stat"><strong>{progress['started']}/{progress['total']}</strong><small>started routes</small></div>
<div class="stat"><strong>16</strong><small>trajectories / batch</small></div><div class="stat"><strong>305</strong><small>common final x</small></div></div>
{''.join(task_sections)}
<p class="empty" style="margin-top:18px">Every update is bound to SHA-256 trajectory and replay evidence. “Next-batch Δ” is observational, not a weight-only causal estimate.</p>
</main></body></html>'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    tmp.write_text(document)
    os.replace(tmp, args.out)
    print(json.dumps({"status": status, "out": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()

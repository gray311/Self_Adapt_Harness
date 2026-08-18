#!/usr/bin/env python3
"""Compose the update-target mechanism with audited evolution evidence.

The compact composition keeps the Circle Packing trajectory, score curve, and
cross-model SOTA validation.  The full composition additionally keeps the
Hadamard-29 parity panel.  Source assets remain untouched.
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WHITE = "#ffffff"
NAVY = "#172033"
MUTED = "#687386"
ORANGE = "#eb6834"
ORANGE_DARK = "#a33f18"
ORANGE_FILL = "#fff5ee"
LINE = "#cbd5e1"
BLUE = "#2a78d6"
PURPLE = "#7e57c2"
GREEN = "#16803a"

FLOW_SVG_HEIGHT = 513.36
FLOW_SVG_WIDTH = 907.2
# Stop after the two feedback loops.  The four benefit cards are restated in
# the evidence bridge below, so cropping before them avoids duplicated labels.
FLOW_CROP_HEIGHT = 432.0
EVIDENCE_SVG_WIDTH = 1188.0
COMPACT_EVIDENCE_HEIGHT = 650.0
FULL_EVIDENCE_HEIGHT = 747.2


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _crop_by_svg_height(image: Image.Image, full_svg_height: float, crop_height: float) -> Image.Image:
    bottom = round(image.height * crop_height / full_svg_height)
    return image.crop((0, 0, image.width, bottom))


def _render_png(
    flow_png: Path,
    evidence_png: Path,
    output: Path,
    *,
    evidence_height: float,
    evidence_label: str,
) -> None:
    outer_width = 2520
    margin = 60
    inner_width = outer_width - 2 * margin
    label_height = 54
    bridge_height = 126
    gap = 20
    bottom_margin = 42

    flow = Image.open(flow_png).convert("RGB")
    flow = _crop_by_svg_height(flow, FLOW_SVG_HEIGHT, FLOW_CROP_HEIGHT)
    flow = _resize_width(flow, inner_width)

    evidence = Image.open(evidence_png).convert("RGB")
    evidence = _crop_by_svg_height(evidence, FULL_EVIDENCE_HEIGHT, evidence_height)
    evidence = _resize_width(evidence, inner_width)

    total_height = label_height + flow.height + gap + bridge_height + gap + evidence.height + bottom_margin
    canvas = Image.new("RGB", (outer_width, total_height), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 14), "A  ·  EVOLUTION MECHANISM", font=_font(27, True), fill=NAVY)
    y = label_height
    canvas.paste(flow, (margin, y))
    y += flow.height + gap

    draw.rounded_rectangle(
        (margin, y, outer_width - margin, y + bridge_height),
        radius=22,
        fill=ORANGE_FILL,
        outline=ORANGE,
        width=3,
    )
    draw.text((margin + 30, y + 16), evidence_label, font=_font(25, True), fill=ORANGE_DARK)
    mechanism_labels = (
        ("1", "EXPLORATION EFFICIENCY", BLUE),
        ("2", "TRACEABLE HARNESS", PURPLE),
        ("3", "REWARD-HACKING CONTROL", GREEN),
        ("4", "TRANSFER", ORANGE),
    )
    segment_width = inner_width / 4
    for index, (number, label, color) in enumerate(mechanism_labels):
        cx = margin + 30 + index * segment_width + 18
        cy = y + 68
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color)
        number_box = draw.textbbox((0, 0), number, font=_font(17, True))
        number_width = number_box[2] - number_box[0]
        number_height = number_box[3] - number_box[1]
        draw.text(
            (cx - number_width / 2, cy - number_height / 2 - 2),
            number,
            font=_font(17, True),
            fill=WHITE,
        )
        draw.text((cx + 29, cy - 13), label, font=_font(18, True), fill=color)
    bridge = (
        "Named harness changes  →  visible program improvements  →  "
        "score trajectory  →  independently validated transfer"
    )
    draw.text((margin + 30, y + 98), bridge, font=_font(18), fill=MUTED)
    y += bridge_height + gap

    canvas.paste(evidence, (margin, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(220, 220), optimize=True)


def _svg_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _render_svg(
    flow_svg: Path,
    evidence_svg: Path,
    output: Path,
    *,
    evidence_height: float,
    evidence_label: str,
) -> None:
    width = 1188.0
    margin = 24.0
    inner_width = width - 2 * margin
    label_height = 28.0
    gap = 10.0
    bridge_height = 70.0
    bottom_margin = 18.0

    flow_height = inner_width * FLOW_CROP_HEIGHT / FLOW_SVG_WIDTH
    rendered_evidence_height = inner_width * evidence_height / EVIDENCE_SVG_WIDTH
    evidence_y = label_height + flow_height + gap + bridge_height + gap
    total_height = evidence_y + rendered_evidence_height + bottom_margin

    flow_data = _svg_data(flow_svg)
    evidence_data = _svg_data(evidence_svg)
    safe_label = html.escape(evidence_label)
    bridge = html.escape(
        "Named harness changes → visible program improvements → "
        "score trajectory → independently validated transfer"
    )
    mechanism_specs = (
        ("1", "EXPLORATION EFFICIENCY", BLUE),
        ("2", "TRACEABLE HARNESS", PURPLE),
        ("3", "REWARD-HACKING CONTROL", GREEN),
        ("4", "TRANSFER", ORANGE),
    )
    mechanism_svg = []
    segment_width = inner_width / 4
    bridge_top = label_height + flow_height + gap
    for index, (number, label, color) in enumerate(mechanism_specs):
        cx = margin + 15 + index * segment_width + 7
        cy = bridge_top + 39
        mechanism_svg.append(
            f'''  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="7.2" fill="{color}"/>
  <text x="{cx:.2f}" y="{cy + 2.8:.2f}" text-anchor="middle"
        font-family="DejaVu Sans, Arial, sans-serif" font-size="7.6" font-weight="700" fill="#ffffff">{number}</text>
  <text x="{cx + 11:.2f}" y="{cy + 2.8:.2f}"
        font-family="DejaVu Sans, Arial, sans-serif" font-size="7.9" font-weight="700" fill="{color}">{html.escape(label)}</text>'''
        )

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width:.0f}pt" height="{total_height:.2f}pt"
     viewBox="0 0 {width:.0f} {total_height:.2f}" role="img"
     aria-label="Update-target mechanism with harness-evolution and transfer evidence">
  <title>Updating the proposer: evolution mechanism and observed evidence</title>
  <desc>TTT-Discover updates executor weights. NexAU and SAH update an explicit harness around a frozen executor, followed by Circle Packing evolution and independently validated cross-model SOTA evidence.</desc>
  <rect x="0" y="0" width="{width:.0f}" height="{total_height:.2f}" fill="#ffffff"/>
  <text x="{margin:.2f}" y="19" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="13.5" font-weight="700" letter-spacing="0.45" fill="{NAVY}">A · EVOLUTION MECHANISM</text>

  <svg x="{margin:.2f}" y="{label_height:.2f}" width="{inner_width:.2f}" height="{flow_height:.2f}"
       viewBox="0 0 {FLOW_SVG_WIDTH:.2f} {FLOW_CROP_HEIGHT:.2f}" preserveAspectRatio="xMidYMid meet">
    <image x="0" y="0" width="{FLOW_SVG_WIDTH:.2f}" height="{FLOW_SVG_HEIGHT:.2f}"
           xlink:href="data:image/svg+xml;base64,{flow_data}"/>
  </svg>

  <rect x="{margin:.2f}" y="{label_height + flow_height + gap:.2f}"
        width="{inner_width:.2f}" height="{bridge_height:.2f}" rx="10"
        fill="{ORANGE_FILL}" stroke="{ORANGE}" stroke-width="1.2"/>
  <text x="{margin + 15:.2f}" y="{label_height + flow_height + gap + 17:.2f}"
        font-family="DejaVu Sans, Arial, sans-serif" font-size="12.5" font-weight="700"
        fill="{ORANGE_DARK}">{safe_label}</text>
{chr(10).join(mechanism_svg)}
  <text x="{margin + 15:.2f}" y="{label_height + flow_height + gap + 61:.2f}"
        font-family="DejaVu Sans, Arial, sans-serif" font-size="9.2" fill="{MUTED}">{bridge}</text>

  <svg x="{margin:.2f}" y="{evidence_y:.2f}" width="{inner_width:.2f}" height="{rendered_evidence_height:.2f}"
       viewBox="0 0 {EVIDENCE_SVG_WIDTH:.2f} {evidence_height:.2f}" preserveAspectRatio="xMidYMid meet">
    <image x="0" y="0" width="{EVIDENCE_SVG_WIDTH:.2f}" height="{FULL_EVIDENCE_HEIGHT:.2f}"
           xlink:href="data:image/svg+xml;base64,{evidence_data}"/>
  </svg>
</svg>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg)


def compose(
    flow_svg: Path,
    flow_png: Path,
    evidence_svg: Path,
    evidence_png: Path,
    output_base: Path,
    *,
    evidence_height: float,
    evidence_label: str,
) -> None:
    _render_svg(
        flow_svg,
        evidence_svg,
        output_base.with_suffix(".svg"),
        evidence_height=evidence_height,
        evidence_label=evidence_label,
    )
    _render_png(
        flow_png,
        evidence_png,
        output_base.with_suffix(".png"),
        evidence_height=evidence_height,
        evidence_label=evidence_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow-base",
        type=Path,
        default=Path("results/figure_drafts/why_update_proposer/layout_e_symmetric_evolution_loops"),
    )
    parser.add_argument(
        "--evidence-base",
        type=Path,
        default=Path("results/cp20_final/evolution_showcase"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figure_drafts/why_update_proposer"),
    )
    args = parser.parse_args()

    compose(
        args.flow_base.with_suffix(".svg"),
        args.flow_base.with_suffix(".png"),
        args.evidence_base.with_suffix(".svg"),
        args.evidence_base.with_suffix(".png"),
        args.out_dir / "update_target_with_cp_evidence",
        evidence_height=COMPACT_EVIDENCE_HEIGHT,
        evidence_label="B  ·  OBSERVED CIRCLE-PACKING EVOLUTION + CROSS-MODEL SOTA",
    )
    compose(
        args.flow_base.with_suffix(".svg"),
        args.flow_base.with_suffix(".png"),
        args.evidence_base.with_suffix(".svg"),
        args.evidence_base.with_suffix(".png"),
        args.out_dir / "update_target_with_full_evidence",
        evidence_height=FULL_EVIDENCE_HEIGHT,
        evidence_label="B  ·  OBSERVED EVOLUTION + TWO SOTA-LEVEL TRANSFER RESULTS",
    )


if __name__ == "__main__":
    main()

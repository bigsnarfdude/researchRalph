#!/usr/bin/env python3
"""
Score Timeline — interactive SVG chart from results.tsv files.

Features:
  - Per-agent swimlanes: each agent gets its own horizontal track
  - X-axis = each agent's own experiment sequence (sorted by EXP-ID number)
  - Global best line shown across all tracks
  - Duplicate/collision experiments flagged in red
  - Multi-run comparison: --results a.tsv b.tsv --labels name1 name2
    renders separate panels stacked vertically

Usage:
    python3 score_timeline.py --results domains/rrma-r1/results.tsv [--baseline 0.62] [--target 0.83]
    python3 score_timeline.py --results r1.tsv v5.tsv --labels rrma-r1 sae-bench-v5
    python3 score_timeline.py --results r1.tsv --mode flat   # original flat timeline
"""
import argparse
import html
import json
import re
from pathlib import Path
from collections import defaultdict


AGENT_COLORS = [
    "#4f9cf9",  # blue
    "#f97316",  # orange
    "#22c55e",  # green
    "#a855f7",  # purple
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f59e0b",  # amber
    "#ef4444",  # red
]

FALLBACK_HEADER = ["EXP-ID", "score", "train_min", "status", "description", "agent", "design"]


def parse_exp_num(exp_id: str) -> tuple:
    m = re.match(r"EXP-(\d+)([a-z]*)", exp_id or "")
    if m:
        return (int(m.group(1)), m.group(2))
    return (9999, exp_id)


def parse_tsv(path: str) -> list[dict]:
    lines = [l.rstrip("\n") for l in Path(path).read_text().splitlines() if l.strip()]
    if not lines:
        return []
    first = lines[0].split("\t")
    if "score" in first or "EXP-ID" in first:
        header, data = first, lines[1:]
    else:
        header, data = FALLBACK_HEADER[:], lines
    rows = []
    for line in data:
        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        row = dict(zip(header, parts))
        try:
            row["score"] = float(row["score"])
        except (ValueError, KeyError):
            continue
        rows.append(row)
    return rows


def find_duplicates(rows: list[dict]) -> set[str]:
    """Return EXP-IDs that appear more than once with different data."""
    by_id = defaultdict(list)
    for r in rows:
        by_id[r.get("EXP-ID", "")].append(r)
    dupes = set()
    for eid, rlist in by_id.items():
        if len(rlist) > 1:
            scores = set(r["score"] for r in rlist)
            if len(scores) > 1:  # conflicting — mark all as suspect
                dupes.add(eid)
    return dupes


def build_swimlane_svg(rows, label, agent_color_map, baseline=None, target=None,
                       W=880, H_per_agent=120, PAD_L=60, PAD_R=80, PAD_T=24, PAD_B=24):
    """Build one SVG panel (one results.tsv = one label) with per-agent swimlanes."""
    # Group by agent, sort each agent's experiments by EXP-ID number
    agents = []
    agent_rows = defaultdict(list)
    for r in rows:
        ag = r.get("agent", "?")
        if ag not in agents:
            agents.append(ag)
        agent_rows[ag].append(r)

    for ag in agents:
        agent_rows[ag].sort(key=lambda r: parse_exp_num(r.get("EXP-ID", "")))

    dupes = find_duplicates(rows)

    # Global best (across all agents, in file order for rolling-best)
    global_best = max(r["score"] for r in rows) if rows else 0

    n_agents = len(agents)
    total_H = PAD_T + n_agents * H_per_agent + PAD_B
    svg = [f'<svg viewBox="0 0 {W} {total_H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;background:#16162a;border-radius:8px;margin-bottom:6px;">']

    # Y range shared across all swimlanes
    all_scores = [r["score"] for r in rows]
    y_min = max(0.0, min(all_scores) - 0.06)
    y_max = min(1.0, max(all_scores) + 0.04)
    if baseline:
        y_min = min(y_min, baseline - 0.02)
    if target:
        y_max = max(y_max, target + 0.02)
    score_range = y_max - y_min or 0.001

    tooltip_data = []

    for lane_i, ag in enumerate(agents):
        color = agent_color_map.get(ag, "#888")
        ag_rows = agent_rows[ag]
        n = len(ag_rows)
        if n == 0:
            continue

        # Lane bounds
        lane_top = PAD_T + lane_i * H_per_agent
        lane_bot = lane_top + H_per_agent
        inner_top = lane_top + 8
        inner_bot = lane_bot - 14
        inner_H = inner_bot - inner_top

        def cx(idx):
            if n <= 1:
                return PAD_L + (W - PAD_L - PAD_R) * 0.5
            return PAD_L + idx / (n - 1) * (W - PAD_L - PAD_R)

        def cy(score):
            frac = (score - y_min) / score_range
            return inner_bot - frac * inner_H

        # Lane background
        lane_bg = "#1c1c30" if lane_i % 2 == 0 else "#1a1a2c"
        svg.append(f'<rect x="0" y="{lane_top}" width="{W}" height="{H_per_agent}" fill="{lane_bg}"/>')

        # Lane label
        svg.append(f'<text x="8" y="{lane_top+15}" fill="{color}" font-size="12" '
                   f'font-family="monospace" font-weight="bold">{html.escape(ag)}</text>')

        # Y-axis grid lines (3 ticks)
        for tick_i in range(3):
            s = y_min + score_range * tick_i / 2
            y = cy(s)
            svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                       f'stroke="#2a2a44" stroke-width="0.8"/>')
            svg.append(f'<text x="{PAD_L-4}" y="{y+3:.1f}" text-anchor="end" fill="#555" '
                       f'font-size="9" font-family="monospace">{s:.2f}</text>')

        # Baseline / target lines (only on first lane, subtly)
        if lane_i == 0 and baseline:
            y = cy(baseline)
            svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                       f'stroke="#555" stroke-width="1" stroke-dasharray="5,3"/>')
            svg.append(f'<text x="{W-PAD_R+3}" y="{y+3:.1f}" fill="#666" font-size="9">base {baseline}</text>')
        if lane_i == 0 and target:
            y = cy(target)
            svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                       f'stroke="#22c55e" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>')
            svg.append(f'<text x="{W-PAD_R+3}" y="{y+3:.1f}" fill="#22c55e" font-size="9" opacity="0.7">target {target}</text>')

        # Connect "keep" experiments with line
        keep_pts = [(i, r) for i, r in enumerate(ag_rows) if r.get("status") == "keep"]
        if len(keep_pts) >= 2:
            pts = " ".join(f"{cx(i):.1f},{cy(r['score']):.1f}" for i, r in keep_pts)
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8"/>')

        # Best marker for this agent
        best_r = max(ag_rows, key=lambda r: r["score"])
        best_i = ag_rows.index(best_r)
        bx, by = cx(best_i), cy(best_r["score"])

        # Points
        for i, r in enumerate(ag_rows):
            x, y = cx(i), cy(r["score"])
            eid = r.get("EXP-ID", "?")
            is_dupe = eid in dupes
            is_discard = r.get("status") == "discard"
            is_best = r is best_r

            pt_color = "#ef4444" if is_dupe else color
            opacity = "0.4" if is_discard else "1"
            radius = 7 if is_best else 5

            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                       f'fill="{pt_color}" opacity="{opacity}" '
                       f'class="pt" data-tip-idx="{len(tooltip_data)}" '
                       f'style="cursor:pointer"/>')
            if is_dupe:
                svg.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" '
                           f'fill="#ef4444" font-size="9">!</text>')

            # X-axis label (every few, or if best/discard)
            step = max(1, n // 6)
            if i % step == 0 or is_best:
                svg.append(f'<text x="{x:.1f}" y="{inner_bot+11:.1f}" text-anchor="middle" '
                           f'fill="#555" font-size="8" font-family="monospace">{html.escape(eid)}</text>')

            design = r.get("design", "")
            desc = r.get("description", "")
            tip = f'{eid} | score={r["score"]:.4f} | {ag}\ndesign: {design}\n{desc}'
            if is_dupe:
                tip += "\n⚠ DUPLICATE EXP-ID — verify!"
            tooltip_data.append({"x": float(f"{x:.1f}"), "y": float(f"{y:.1f}"), "tip": tip})

        # Best annotation
        svg.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="9" fill="none" '
                   f'stroke="#fbbf24" stroke-width="1.5"/>')
        svg.append(f'<text x="{bx:.1f}" y="{by-12:.1f}" text-anchor="middle" '
                   f'fill="#fbbf24" font-size="10" font-weight="bold">★{best_r["score"]:.4f}</text>')

    # Global best bar at top-right
    svg.append(f'<text x="{W-4}" y="14" text-anchor="end" fill="#fbbf24" '
               f'font-size="11" font-family="monospace">best {global_best:.4f}</text>')

    svg.append('</svg>')
    return "\n".join(svg), tooltip_data


def build_html(all_rows, labels, baseline=None, target=None, title="Score Timeline"):
    # Collect all agents across all runs for consistent coloring
    all_agents = []
    for rows in all_rows:
        for r in rows:
            ag = r.get("agent", "?")
            if ag not in all_agents:
                all_agents.append(ag)
    agent_color = {ag: AGENT_COLORS[i % len(AGENT_COLORS)] for i, ag in enumerate(all_agents)}

    panels = []
    all_tooltips = []
    for rows, label in zip(all_rows, labels):
        svg, tips = build_swimlane_svg(rows, label, agent_color, baseline=baseline, target=target)
        # Offset tip indices to be global
        offset = len(all_tooltips)
        svg = svg.replace('data-tip-idx="', f'data-tip-idx-offset="{offset}" data-tip-idx="')
        panels.append((label, svg))
        all_tooltips.extend(tips)

    # Legend
    legend = ""
    for ag in all_agents:
        c = agent_color[ag]
        legend += (f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
                   f'<span style="width:11px;height:11px;border-radius:50%;background:{c}"></span>'
                   f'<span style="color:#bbb;font-size:12px">{html.escape(ag)}</span></span>')
    legend += (f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
               f'<span style="width:11px;height:11px;border-radius:50%;background:#ef4444"></span>'
               f'<span style="color:#bbb;font-size:12px">duplicate EXP-ID</span></span>')

    panels_html = ""
    for label, svg in panels:
        panels_html += f'<div class="run-label">{html.escape(label)}</div>\n{svg}\n'

    tip_json = json.dumps(all_tooltips)
    js = f"""
<script>
const tips = {tip_json};
document.querySelectorAll('.pt').forEach(pt => {{
  const offset = parseInt(pt.getAttribute('data-tip-idx-offset') || 0);
  const idx = parseInt(pt.getAttribute('data-tip-idx')) + offset;
  const tip = document.getElementById('tip');
  pt.addEventListener('mouseenter', e => {{
    const d = tips[idx];
    if (!d) return;
    tip.style.display = 'block';
    tip.innerText = d.tip;
    const rect = pt.closest('svg').getBoundingClientRect();
    const svgW = rect.width;
    const vbW = 880;
    const scaleX = svgW / vbW;
    tip.style.left = (rect.left + d.x * scaleX + 12) + 'px';
    tip.style.top = (rect.top + window.scrollY + d.y * (rect.height / parseFloat(pt.closest('svg').getAttribute('viewBox').split(' ')[3])) - 10) + 'px';
  }});
  pt.addEventListener('mouseleave', () => tip.style.display = 'none');
}});
</script>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0f0f1a; color: #eee; font-family: system-ui, sans-serif; padding: 20px; }}
h2 {{ color: #fff; font-size: 17px; margin-bottom: 10px; }}
.legend {{ margin: 8px 0 14px; display: flex; flex-wrap: wrap; gap: 4px; }}
.run-label {{ color: #777; font-size: 11px; font-family: monospace; margin: 10px 0 3px;
             text-transform: uppercase; letter-spacing: 0.05em; }}
#tip {{ position: fixed; background: #1e1e3a; border: 1px solid #444; color: #eee;
       font-size: 11px; font-family: monospace; padding: 8px 10px; border-radius: 4px;
       white-space: pre; pointer-events: none; display: none; z-index: 100; max-width: 420px;
       line-height: 1.5; }}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<div class="legend">{legend}</div>
{panels_html}
<div id="tip"></div>
{js}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Score timeline chart (per-agent swimlanes)")
    parser.add_argument("--results", nargs="+", required=True, help="One or more results.tsv paths")
    parser.add_argument("--labels", nargs="+", help="Labels for each results file")
    parser.add_argument("--baseline", type=float)
    parser.add_argument("--target", type=float)
    parser.add_argument("--title", default="Score Timeline")
    parser.add_argument("--output", "-o", default="/tmp/timeline.html")
    args = parser.parse_args()

    labels = args.labels or [Path(p).parent.name or Path(p).stem for p in args.results]
    if len(labels) < len(args.results):
        labels += [Path(p).stem for p in args.results[len(labels):]]

    all_rows = [parse_tsv(p) for p in args.results]
    page = build_html(all_rows, labels, baseline=args.baseline, target=args.target, title=args.title)
    Path(args.output).write_text(page)
    print(f"Wrote {args.output} ({len(page)//1024}KB)")


if __name__ == "__main__":
    main()

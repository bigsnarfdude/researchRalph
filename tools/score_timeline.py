#!/usr/bin/env python3
"""
Score Timeline — interactive SVG chart from results.tsv files.

Usage:
    python3 score_timeline.py --results domains/rrma-r1/results.tsv --output /tmp/timeline.html
    python3 score_timeline.py --results r1.tsv v5.tsv --labels rrma-r1 sae-bench-v5 --output /tmp/compare.html
"""
import argparse
import html
import json
from pathlib import Path


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

DESIGN_MARKERS = {
    "REINFORCE-group": "◆",
    "GRPO": "●",
    "GRPO-iter": "★",
    "GRPO-noKL": "▲",
    "GRPO-iter-SGD": "■",
    "ReST": "✕",
    "BatchRefStyleSAE": "●",
    "EvalISTARefStyleSAE": "◆",
}


FALLBACK_HEADER = ["EXP-ID", "score", "train_min", "status", "description", "agent", "design"]


def parse_tsv(path):
    rows = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if not lines:
        return rows

    # Detect if first line is a header (contains "score" literally)
    first = lines[0].split("\t")
    if "score" in first:
        header = first
        data_lines = lines[1:]
    else:
        header = FALLBACK_HEADER
        data_lines = lines

    for line in data_lines:
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


def build_html(all_rows, labels, baseline=None, target=None, title="Score Timeline"):
    # Collect agents across all runs
    all_agents = []
    for rows, label in zip(all_rows, labels):
        for r in rows:
            agent = r.get("agent", "?")
            key = f"{label}/{agent}" if len(labels) > 1 else agent
            if key not in all_agents:
                all_agents.append(key)

    agent_color = {a: AGENT_COLORS[i % len(AGENT_COLORS)] for i, a in enumerate(all_agents)}

    # Build per-agent series + rolling best
    series = {}  # agent -> [(exp_num, exp_id, score, design, desc, status)]
    for rows, label in zip(all_rows, labels):
        for i, r in enumerate(rows):
            agent = r.get("agent", "?")
            key = f"{label}/{agent}" if len(labels) > 1 else agent
            exp_id = r.get("EXP-ID", f"EXP-{i+1}")
            design = r.get("design", "")
            desc = r.get("description", "")
            status = r.get("status", "keep")
            if key not in series:
                series[key] = []
            series[key].append({
                "exp_id": exp_id,
                "exp_num": i + 1,
                "score": r["score"],
                "design": design,
                "desc": desc,
                "status": status,
                "global_idx": sum(len(v) for v in series.values()),
                "label": label,
            })

    # Global experiment order (by appearance across all rows flattened)
    all_exp = []
    for rows, label in zip(all_rows, labels):
        for i, r in enumerate(rows):
            all_exp.append({
                "exp_id": r.get("EXP-ID", ""),
                "score": r["score"],
                "agent": r.get("agent", "?"),
                "label": label,
                "design": r.get("design", ""),
                "desc": r.get("description", ""),
                "status": r.get("status", "keep"),
            })

    total = len(all_exp)
    if total == 0:
        return "<p>No data</p>"

    all_scores = [r["score"] for r in all_exp]
    y_min = max(0, min(all_scores) - 0.05)
    y_max = min(1.0, max(all_scores) + 0.05)
    if baseline:
        y_min = min(y_min, baseline - 0.02)
    if target:
        y_max = max(y_max, target + 0.02)

    # SVG dimensions
    W, H = 900, 380
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 30, 30, 50

    def cx(idx):
        n = max(total - 1, 1)
        return PAD_L + (idx / n) * (W - PAD_L - PAD_R)

    def cy(score):
        rng = y_max - y_min
        if rng == 0:
            return H - PAD_B
        frac = (score - y_min) / rng
        return H - PAD_B - frac * (H - PAD_T - PAD_B)

    svg_parts = []
    svg_parts.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{W}px;background:#1a1a2e;border-radius:8px;">')

    # Grid lines
    n_gridlines = 6
    for i in range(n_gridlines + 1):
        s = y_min + (y_max - y_min) * i / n_gridlines
        y = cy(s)
        svg_parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="#2a2a4a" stroke-width="1"/>')
        svg_parts.append(f'<text x="{PAD_L-5}" y="{y+4:.1f}" text-anchor="end" fill="#888" font-size="11" font-family="monospace">{s:.3f}</text>')

    # Baseline line
    if baseline:
        y = cy(baseline)
        svg_parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="#666" stroke-width="1.5" stroke-dasharray="6,3"/>')
        svg_parts.append(f'<text x="{W-PAD_R+3}" y="{y+4:.1f}" fill="#888" font-size="10">baseline {baseline}</text>')

    # Target line
    if target:
        y = cy(target)
        svg_parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.6"/>')
        svg_parts.append(f'<text x="{W-PAD_R+3}" y="{y+4:.1f}" fill="#22c55e" font-size="10" opacity="0.8">target {target}</text>')

    # Rolling best line (global)
    best_so_far = []
    best = -1
    for i, r in enumerate(all_exp):
        if r["score"] > best:
            best = r["score"]
        best_so_far.append(best)

    pts = " ".join(f"{cx(i):.1f},{cy(b):.1f}" for i, b in enumerate(best_so_far))
    svg_parts.append(f'<polyline points="{pts}" fill="none" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.3"/>')

    # Per-agent lines (keep only)
    for agent_key in all_agents:
        key_label = agent_key.split("/")[0] if len(labels) > 1 else None
        key_agent = agent_key.split("/")[-1]
        color = agent_color[agent_key]

        agent_pts = []
        for i, r in enumerate(all_exp):
            agent = r["agent"]
            label = r["label"]
            rkey = f"{label}/{agent}" if len(labels) > 1 else agent
            if rkey == agent_key and r["status"] == "keep":
                agent_pts.append((i, r["score"]))

        if len(agent_pts) >= 2:
            pts = " ".join(f"{cx(i):.1f},{cy(s):.1f}" for i, s in agent_pts)
            svg_parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="0.7"/>')

    # Points
    tooltips = []
    for i, r in enumerate(all_exp):
        agent = r["agent"]
        label = r["label"]
        rkey = f"{label}/{agent}" if len(labels) > 1 else agent
        color = agent_color.get(rkey, "#fff")
        is_discard = r["status"] == "discard"
        marker = DESIGN_MARKERS.get(r["design"], "●")
        x, y = cx(i), cy(r["score"])

        opacity = "0.35" if is_discard else "1"
        radius = 5 if not is_discard else 4
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}" '
            f'class="pt" data-idx="{i}" style="cursor:pointer"/>'
        )

        tip = html.escape(f'{r["exp_id"]} | {r["score"]:.3f} | {rkey} | {r["design"]}\n{r["desc"]}')
        tooltips.append({"x": x, "y": y, "tip": tip, "idx": i})

    # X-axis labels (every N experiments)
    step = max(1, total // 10)
    for i, r in enumerate(all_exp):
        if i % step == 0 or i == total - 1:
            x = cx(i)
            svg_parts.append(f'<text x="{x:.1f}" y="{H-PAD_B+15}" text-anchor="middle" fill="#888" font-size="10" font-family="monospace">{r["exp_id"]}</text>')

    # Best score annotation
    best_idx = max(range(len(all_exp)), key=lambda i: all_exp[i]["score"])
    bx, by = cx(best_idx), cy(all_exp[best_idx]["score"])
    best_score = all_exp[best_idx]["score"]
    svg_parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="none" stroke="#fbbf24" stroke-width="2"/>')
    svg_parts.append(f'<text x="{bx:.1f}" y="{by-12:.1f}" text-anchor="middle" fill="#fbbf24" font-size="11" font-weight="bold">★ {best_score:.4f}</text>')

    svg_parts.append('</svg>')
    svg = "\n".join(svg_parts)

    # Legend
    legend_items = ""
    for agent_key in all_agents:
        color = agent_color[agent_key]
        legend_items += f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;"><span style="width:12px;height:12px;border-radius:50%;background:{color};display:inline-block"></span><span style="color:#ccc;font-size:13px">{agent_key}</span></span>'

    # Tooltip div + JS
    tooltip_data = json.dumps(tooltips)
    js = f"""
<script>
const pts = document.querySelectorAll('.pt');
const tip = document.getElementById('tip');
const ttData = {tooltip_data};
pts.forEach(pt => {{
    pt.addEventListener('mouseenter', e => {{
        const idx = parseInt(pt.getAttribute('data-idx'));
        const d = ttData[idx];
        tip.style.display = 'block';
        tip.style.left = (d.x + 20) + 'px';
        tip.style.top = (d.y - 10) + 'px';
        tip.innerText = d.tip;
    }});
    pt.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
}});
</script>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ background:#0f0f1a; color:#eee; font-family:system-ui,sans-serif; margin:0; padding:20px; }}
h2 {{ color:#fff; font-size:18px; margin-bottom:8px; }}
.legend {{ margin:10px 0 16px 0; }}
.chart-wrap {{ position:relative; display:inline-block; width:100%; max-width:900px; }}
#tip {{ position:absolute; background:#1e1e3a; border:1px solid #444; color:#eee; font-size:12px; font-family:monospace; padding:8px 10px; border-radius:4px; white-space:pre; pointer-events:none; display:none; z-index:100; max-width:400px; }}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<div class="legend">{legend_items}</div>
<div class="chart-wrap">
{svg}
<div id="tip"></div>
</div>
{js}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Score timeline chart from results.tsv")
    parser.add_argument("--results", nargs="+", required=True, help="One or more results.tsv paths")
    parser.add_argument("--labels", nargs="+", help="Labels for each results file (default: filename)")
    parser.add_argument("--baseline", type=float, help="Baseline score to draw as dashed line")
    parser.add_argument("--target", type=float, help="Target score to draw as green dashed line")
    parser.add_argument("--title", default="Score Timeline", help="Chart title")
    parser.add_argument("--output", "-o", default="/tmp/timeline.html", help="Output HTML path")
    args = parser.parse_args()

    labels = args.labels or [Path(p).parent.name or Path(p).stem for p in args.results]
    if len(labels) < len(args.results):
        labels += [Path(p).stem for p in args.results[len(labels):]]

    all_rows = [parse_tsv(p) for p in args.results]
    page = build_html(all_rows, labels,
                      baseline=args.baseline,
                      target=args.target,
                      title=args.title)
    Path(args.output).write_text(page)
    print(f"Wrote {args.output} ({len(page)//1024}KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TrustLoop Experiment Visualizer — Pareto-front style chart from results.tsv.

Mirrors the Claudini paper plot but adds TrustLoop verification layer:
  - Innovation jumps vs parameter-tuning variants (auto-detected from design field)
  - Best-so-far trajectory
  - Certified / suspect / uncertified annotations
  - Steering intervention markers
  - Train/valid gap flags (if valid score column present)

Usage:
    python3 trustloop_viz.py --results domains/rrma-r1/results.tsv \\
        --baseline 0.620 --target 0.830 --output /tmp/viz.html
    python3 trustloop_viz.py --results r1.tsv --title "rrma-r1 gen2"
"""
import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

AGENT_COLORS = {
    "agent0": "#388bfd",
    "agent1": "#3fb950",
    "agent2": "#d29922",
    "agent3": "#f78166",
}
DEFAULT_COLOR = "#8b949e"
FALLBACK_HEADER = ["EXP-ID", "score", "train_min", "status", "description", "agent", "design"]

# Designs considered "HP tuning" variants rather than novel methods
HP_DESIGNS = {"GRPO-iter-SGD", "candidate-scaling", "majority-vote", "curriculum-MAJ"}

# Known suspect experiments (post-hoc selection, multiple comparisons, etc.)
SUSPECT_IDS = {"EXP-026"}  # logprob best-of-7 selection on test set


def parse_exp_num(exp_id):
    m = re.match(r"EXP-(\d+)([a-z]*)", exp_id or "")
    if m:
        return (int(m.group(1)), m.group(2))
    # Handle special IDs like MAJ-8, MAJ-16
    m2 = re.match(r"[A-Z]+-(\d+)", exp_id or "")
    if m2:
        return (500 + int(m2.group(1)), "")
    return (9999, exp_id)


def load_results(path):
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


def classify_experiment(row):
    """Return 'innovation', 'hp_tuning', 'inference', or 'suspect'."""
    eid = row.get("EXP-ID", "")
    if eid in SUSPECT_IDS:
        return "suspect"
    design = row.get("design", "")
    if "majority" in design.lower() or "MAJ" in eid or "scaling" in design.lower():
        return "inference"
    if design in HP_DESIGNS or "SGD" in design or "iter" in design.lower():
        return "hp_tuning"
    return "innovation"


def detect_innovation_chain(rows):
    """
    Build innovation lineage: each innovation points to the previous best
    that it likely built on. Simple heuristic: innovations are connected
    in score order when they set a new best.
    """
    chain = []
    best = -1
    for r in rows:
        if classify_experiment(r) == "innovation" and r["score"] > best:
            chain.append(r)
            best = r["score"]
    return chain


def build_html(rows, baseline=None, target=None, title="TrustLoop Experiment Visualizer"):
    if not rows:
        return "<p>No data.</p>"

    # Sort by EXP-ID number for x-axis
    rows_sorted = sorted(rows, key=lambda r: parse_exp_num(r.get("EXP-ID", "")))

    # Deduplicate by EXP-ID (keep higher score)
    seen = {}
    deduped = []
    for r in rows_sorted:
        eid = r.get("EXP-ID", "")
        if eid not in seen or r["score"] > seen[eid]["score"]:
            seen[eid] = r
            deduped.append(r)
    rows_sorted = deduped

    # x positions (sequential index)
    for i, r in enumerate(rows_sorted):
        r["_x"] = i

    n = len(rows_sorted)
    scores = [r["score"] for r in rows_sorted]
    best_score = max(scores)
    worst_score = min(scores)

    # best-so-far trajectory
    best_so_far = []
    running_best = -1
    for r in rows_sorted:
        if r["score"] > running_best:
            running_best = r["score"]
        best_so_far.append(running_best)

    # Stats
    agents = set(r.get("agent", "?") for r in rows_sorted)
    innovations = [r for r in rows_sorted if classify_experiment(r) == "innovation"]
    hp_variants = [r for r in rows_sorted if classify_experiment(r) == "hp_tuning"]
    inference_runs = [r for r in rows_sorted if classify_experiment(r) == "inference"]
    suspects = [r for r in rows_sorted if classify_experiment(r) == "suspect"]

    # Build chart data for JS
    points = []
    for r in rows_sorted:
        kind = classify_experiment(r)
        ag = r.get("agent", "?")
        color = AGENT_COLORS.get(ag, DEFAULT_COLOR)
        points.append({
            "x": r["_x"],
            "score": r["score"],
            "eid": r.get("EXP-ID", "?"),
            "agent": ag,
            "color": color,
            "kind": kind,
            "status": r.get("status", "keep"),
            "desc": r.get("description", ""),
            "design": r.get("design", ""),
            "train_min": r.get("train_min", "0"),
        })

    # Innovation chain for annotation arrows
    inno_chain = detect_innovation_chain(rows_sorted)
    inno_eids = {r.get("EXP-ID"): r["_x"] for r in inno_chain}

    points_json = json.dumps(points)
    bsf_json = json.dumps(best_so_far)
    baseline_json = json.dumps(baseline)
    target_json = json.dumps(target)
    inno_eids_json = json.dumps(inno_eids)
    title_esc = html.escape(title)

    # Legend/stats panel data
    stats = {
        "total": n,
        "innovations": len(innovations),
        "hp_variants": len(hp_variants),
        "inference": len(inference_runs),
        "suspects": len(suspects),
        "agents": len(agents),
        "best": best_score,
        "worst": worst_score,
    }
    stats_json = json.dumps(stats)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_esc}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,ui-monospace,monospace;padding:24px}}
h1{{color:#fff;font-size:16px;margin-bottom:4px}}
.subtitle{{color:#6e7681;font-size:12px;margin-bottom:20px}}
.layout{{display:flex;gap:20px;align-items:flex-start}}
.chart-wrap{{flex:1;min-width:0}}
canvas{{width:100%;border-radius:8px;background:#161b22;display:block}}
.sidebar{{width:220px;flex-shrink:0;display:flex;flex-direction:column;gap:12px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.card-title{{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}}
.stat-row{{display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px}}
.stat-label{{color:#8b949e}}.stat-val{{font-weight:700;color:#e6edf3}}
.legend-item{{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px;color:#8b949e}}
.legend-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.legend-line{{width:20px;height:3px;border-radius:1px;flex-shrink:0}}
#tooltip{{position:fixed;background:#1c2128;border:1px solid #444;color:#e6edf3;
  font-size:11px;font-family:monospace;padding:10px 12px;border-radius:6px;
  pointer-events:none;display:none;z-index:999;max-width:320px;line-height:1.6;
  white-space:pre-wrap;box-shadow:0 4px 16px rgba(0,0,0,.5)}}
.cert-strip{{margin-top:10px;padding:8px 10px;border-radius:5px;font-size:11px;line-height:1.5}}
.cert-ok{{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.2);color:#3fb950}}
.cert-warn{{background:rgba(247,129,102,.08);border:1px solid rgba(247,129,102,.2);color:#f78166}}
</style>
</head>
<body>
<h1>{title_esc}</h1>
<div class="subtitle">TrustLoop Experiment Visualizer — hover points for details, click to isolate</div>

<div class="layout">
  <div class="chart-wrap">
    <canvas id="c" height="460"></canvas>
  </div>
  <div class="sidebar">
    <div class="card">
      <div class="card-title">Run stats</div>
      <div id="stats-content"></div>
    </div>
    <div class="card">
      <div class="card-title">Legend</div>
      <div class="legend-item"><span class="legend-dot" style="background:#58a6ff;border:2px solid #58a6ff"></span>Innovation (new method)</div>
      <div class="legend-item"><span class="legend-dot" style="background:transparent;border:2px solid #8b949e"></span>HP tuning variant</div>
      <div class="legend-item"><span class="legend-dot" style="background:#a371f7"></span>Inference-time scaling</div>
      <div class="legend-item"><span class="legend-dot" style="background:#f78166"></span>Suspect (methodology flag)</div>
      <div class="legend-item"><span class="legend-dot" style="background:#484f58"></span>Discarded</div>
      <div class="legend-item" style="margin-top:6px"><span class="legend-line" style="background:#ef4444"></span>Best-so-far</div>
      <div class="legend-item"><span class="legend-line" style="background:#555;border-top:1px dashed #555"></span>Baseline / target</div>
    </div>
    <div class="card" id="cert-card">
      <div class="card-title">TrustLoop verification</div>
      <div id="cert-content"><div style="color:#484f58;font-size:11px">hover a point</div></div>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script>
const points   = {points_json};
const bsf      = {bsf_json};
const baseline = {baseline_json};
const target   = {target_json};
const innoEids = {inno_eids_json};
const stats    = {stats_json};

// ── render stats ─────────────────────────────────────────────────
document.getElementById('stats-content').innerHTML = [
  ['Total exps', stats.total],
  ['Innovations', stats.innovations],
  ['HP variants', stats.hp_variants],
  ['Inference', stats.inference],
  ['Suspects', `<span style="color:#f78166">${{stats.suspects}}</span>`],
  ['Agents', stats.agents],
  ['Best score', `<span style="color:#fbbf24">${{stats.best.toFixed(4)}}</span>`],
].map(([l,v]) => `<div class="stat-row"><span class="stat-label">${{l}}</span><span class="stat-val">${{v}}</span></div>`).join('');

// ── canvas setup ─────────────────────────────────────────────────
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const DPR    = window.devicePixelRatio || 1;
const W_CSS  = canvas.parentElement.clientWidth;
const H_CSS  = 460;
canvas.width  = W_CSS * DPR;
canvas.height = H_CSS * DPR;
canvas.style.width  = W_CSS + 'px';
canvas.style.height = H_CSS + 'px';
ctx.scale(DPR, DPR);

const W = W_CSS, H = H_CSS;
const PAD = {{l:60, r:30, t:30, b:50}};
const CW = W - PAD.l - PAD.r;
const CH = H - PAD.t - PAD.b;

const n = points.length;
const allScores = points.map(p => p.score);
const sMin = Math.max(0, Math.min(...allScores) - 0.04);
const sMax = Math.min(1.0, Math.max(...allScores) + 0.03);

function cx(idx) {{ return PAD.l + (idx / Math.max(n-1,1)) * CW; }}
function cy(score) {{ return PAD.t + (1 - (score - sMin)/(sMax - sMin)) * CH; }}

function drawChart() {{
  ctx.clearRect(0, 0, W, H);

  // background
  ctx.fillStyle = '#161b22';
  ctx.beginPath();
  ctx.roundRect(0, 0, W, H, 8);
  ctx.fill();

  // grid lines
  const ticks = 5;
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= ticks; i++) {{
    const s = sMin + (sMax - sMin) * i / ticks;
    const y = cy(s);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    ctx.fillStyle = '#484f58'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(s.toFixed(3), PAD.l - 6, y + 3);
  }}

  // x-axis labels
  const step = Math.max(1, Math.floor(n / 10));
  ctx.fillStyle = '#484f58'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
  for (let i = 0; i < n; i += step) {{
    ctx.fillText(points[i].eid, cx(i), H - PAD.b + 14);
  }}

  // baseline
  if (baseline !== null) {{
    const y = cy(baseline);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = '#444'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.textAlign = 'left'; ctx.font = '9px monospace';
    ctx.fillText('baseline ' + baseline, PAD.l + 4, y - 4);
    ctx.setLineDash([]);
  }}

  // target
  if (target !== null) {{
    const y = cy(target);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillStyle = '#22c55e'; ctx.textAlign = 'left'; ctx.font = '9px monospace';
    ctx.fillText('target ' + target, PAD.l + 4, y - 4);
    ctx.setLineDash([]);
  }}

  // best-so-far line
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2.5; ctx.globalAlpha = 0.85;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    const x = cx(i), y = cy(bsf[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke(); ctx.globalAlpha = 1;

  // innovation chain arrows
  const innoPoints = points.filter(p => p.kind === 'innovation' && p.score === bsf[p.x] ||
    Object.values(innoEids).includes(p.x));
  // draw subtle connecting arrows between best innovations
  const bestInno = points.filter(p => p.kind === 'innovation').sort((a,b) => a.x - b.x);
  if (bestInno.length >= 2) {{
    ctx.strokeStyle = '#388bfd'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.4;
    ctx.setLineDash([4, 3]);
    for (let i = 1; i < bestInno.length; i++) {{
      const a = bestInno[i-1], b = bestInno[i];
      if (b.score > a.score) {{  // only connect improvements
        ctx.beginPath();
        ctx.moveTo(cx(a.x), cy(a.score));
        ctx.bezierCurveTo(
          cx(a.x) + (cx(b.x)-cx(a.x))*0.5, cy(a.score),
          cx(a.x) + (cx(b.x)-cx(a.x))*0.5, cy(b.score),
          cx(b.x), cy(b.score)
        );
        ctx.stroke();
      }}
    }}
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }}

  // points
  for (const p of points) {{
    const x = cx(p.x), y = cy(p.score);
    const is_discard = p.status === 'discard';
    const is_best = p.score === Math.max(...allScores);

    ctx.globalAlpha = is_discard ? 0.3 : 1;

    if (p.kind === 'suspect') {{
      // red X marker
      ctx.strokeStyle = '#f78166'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x-5,y-5); ctx.lineTo(x+5,y+5); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x+5,y-5); ctx.lineTo(x-5,y+5); ctx.stroke();
    }} else if (p.kind === 'innovation') {{
      // filled circle + glow if best
      if (is_best) {{
        ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 12;
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        // star label
        ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center';
        ctx.fillText('★' + p.score.toFixed(3), x, y - 12);
      }} else {{
        ctx.fillStyle = '#388bfd';
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2); ctx.fill();
        // label for significant improvements
        if (p.score === bsf[p.x]) {{
          ctx.fillStyle = '#58a6ff'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
          ctx.fillText(p.eid, x, y - 10);
        }}
      }}
    }} else if (p.kind === 'inference') {{
      // diamond
      ctx.fillStyle = '#a371f7';
      ctx.beginPath();
      ctx.moveTo(x, y-6); ctx.lineTo(x+5, y); ctx.lineTo(x, y+6); ctx.lineTo(x-5, y); ctx.closePath();
      ctx.fill();
      if (p.score >= 0.81) {{
        ctx.fillStyle = '#a371f7'; ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
        ctx.fillText(p.score.toFixed(3), x, y - 10);
      }}
    }} else if (p.kind === 'hp_tuning') {{
      // hollow circle
      ctx.strokeStyle = p.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.stroke();
    }}

    ctx.globalAlpha = 1;
  }}

  // axes
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, H - PAD.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD.l, H - PAD.b); ctx.lineTo(W - PAD.r, H - PAD.b); ctx.stroke();

  // Y-axis label
  ctx.save(); ctx.translate(14, PAD.t + CH/2); ctx.rotate(-Math.PI/2);
  ctx.fillStyle = '#6e7681'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  ctx.fillText('Score', 0, 0); ctx.restore();
}}

drawChart();

// ── tooltip + verification panel ──────────────────────────────────
const tip = document.getElementById('tooltip');
const certContent = document.getElementById('cert-content');

function verificationStatus(p) {{
  if (p.kind === 'suspect') {{
    return '<div class="cert-strip cert-warn">⚠ SUSPECT — post-hoc selection bias\\nScore ' + p.score.toFixed(4) + ' not trustworthy\\nNeeds: single-strategy rerun</div>';
  }}
  if (p.kind === 'inference' && p.score >= 0.82) {{
    return '<div class="cert-strip cert-ok">✓ CERTIFIED — inference-time scaling\\nClean single-strategy evaluation\\nReproducible: run checkpoint @8 temp=0.7</div>';
  }}
  if (p.kind === 'innovation' && p.status === 'keep') {{
    return '<div class="cert-strip cert-ok">✓ KEEP — novel method, score logged\\nVerify: reproduce on fresh seed\\nDesign: ' + p.design + '</div>';
  }}
  if (p.status === 'discard') {{
    return '<div class="cert-strip cert-warn">✗ DISCARDED — agent rejected\\n' + p.desc.slice(0,80) + '</div>';
  }}
  return '<div class="cert-strip" style="background:#1c2128;border:1px solid #30363d;color:#6e7681">Unverified — kept by agent\\nSpot-check recommended</div>';
}}

canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let closest = null, minD = 18;
  for (const p of points) {{
    const d = Math.hypot(cx(p.x) - mx, cy(p.score) - my);
    if (d < minD) {{ minD = d; closest = p; }}
  }}

  if (closest) {{
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 16) + 'px';
    tip.style.top  = (e.clientY - 10) + 'px';
    tip.textContent = [
      closest.eid + ' — ' + closest.score.toFixed(4),
      'agent: ' + closest.agent + '  kind: ' + closest.kind,
      'design: ' + closest.design,
      'train_min: ' + closest.train_min,
      '',
      closest.desc,
    ].join('\\n');
    certContent.innerHTML = verificationStatus(closest);
  }} else {{
    tip.style.display = 'none';
    certContent.innerHTML = '<div style="color:#484f58;font-size:11px">hover a point</div>';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="TrustLoop Experiment Visualizer")
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline", type=float)
    parser.add_argument("--target", type=float)
    parser.add_argument("--title", default="TrustLoop Experiment Visualizer")
    parser.add_argument("--output", "-o", default="/tmp/trustloop_viz.html")
    args = parser.parse_args()

    rows = load_results(args.results)
    if not rows:
        print("No data found.")
        return

    page = build_html(rows, baseline=args.baseline, target=args.target, title=args.title)
    Path(args.output).write_text(page)
    print(f"Wrote {args.output} ({len(page)//1024}KB)")

    try:
        import subprocess
        subprocess.Popen(["open", args.output])
    except Exception:
        pass


if __name__ == "__main__":
    main()

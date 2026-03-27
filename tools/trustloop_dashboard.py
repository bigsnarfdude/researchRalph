#!/usr/bin/env python3
"""
TrustLoop Dashboard — live verification UI for RRMA runs.

Layout:
  LEFT  : agent selector, experiments list, live log tail
  CENTRE: hero view  [Timeline | Trace | Blackboard | Steer]
  RIGHT : score/status/agent filters
  BOTTOM: chat input (query run context)

The Steer view is the human intervention surface — write a message,
click "Send to Blackboard", it gets appended as a signed [HUMAN] entry.

Usage:
    python3 trustloop_dashboard.py --domain domains/rrma-r1
    python3 trustloop_dashboard.py --domain domains/sae-bench-v5 --port 7433
    python3 trustloop_dashboard.py --domain domains/rrma-r1 --blackboard-path custom/blackboard.md
"""
import argparse
import html
import json
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


AGENT_COLORS = {
    "agent0": "#388bfd",
    "agent1": "#3fb950",
    "agent2": "#d29922",
    "agent3": "#f78166",
    "agent4": "#a371f7",
    "agent5": "#39d353",
}
DEFAULT_COLOR = "#8b949e"
FALLBACK_HEADER = ["EXP-ID", "score", "train_min", "status", "description", "agent", "design"]


# ── data loading ─────────────────────────────────────────────────────────────

def parse_exp_num(exp_id):
    m = re.match(r"EXP-(\d+)([a-z]*)", exp_id or "")
    if m:
        return (int(m.group(1)), m.group(2))
    return (9999, exp_id)


def load_results(path):
    if not path or not Path(path).exists():
        return []
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


def load_blackboard(path):
    if not path or not Path(path).exists():
        return "(no blackboard found)"
    return Path(path).read_text()


def load_logs_tail(domain_path, n=60):
    """Return last n lines from any log files in the domain."""
    lines = []
    for log_file in sorted(Path(domain_path).glob("*.log")):
        try:
            text = log_file.read_text()
            lines.extend(text.splitlines()[-20:])
        except Exception:
            pass
    return "\n".join(lines[-n:]) if lines else "(no .log files found in domain)"


def load_trace_for_exp(logs_dir, exp_id):
    """Scan jsonl files and return turns that mention exp_id."""
    if not logs_dir or not Path(logs_dir).exists():
        return []
    exp_upper = exp_id.upper()
    matching = []
    for jsonl_path in sorted(Path(logs_dir).glob("*.jsonl")):
        try:
            events = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
            agent_id = "agent?"
            for e in events:
                t = e.get("type")
                msg = e.get("message", {})
                ts = e.get("timestamp", "")
                if t == "user" and e.get("parentUuid") is None:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        m = re.search(r"You are (agent\d+)", content)
                        if m:
                            agent_id = m.group(1)
                elif t == "assistant":
                    for block in msg.get("content", []):
                        text = ""
                        if block.get("type") == "text":
                            text = block.get("text", "").strip()
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            if name == "Bash":
                                text = inp.get("command", "")[:500]
                            elif name in ("Write", "Edit"):
                                text = inp.get("file_path", "")
                            else:
                                text = str(inp)[:200]
                        if text and exp_upper in text.upper():
                            try:
                                ts_fmt = datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")
                                ).strftime("%H:%M:%S")
                            except Exception:
                                ts_fmt = ts[:8]
                            matching.append({
                                "agent": agent_id,
                                "ts": ts_fmt,
                                "text": text[:800],
                                "type": block.get("type", "text"),
                                "name": block.get("name", ""),
                            })
        except Exception:
            pass
    return matching


def build_timeline_svg(rows, baseline=None, target=None):
    """Inline swimlane SVG — one track per agent."""
    if not rows:
        return "<p style='color:#555;padding:20px'>No results yet.</p>"

    W, H_per_agent, PAD_L, PAD_R, PAD_T, PAD_B = 860, 110, 60, 90, 20, 20
    agent_order = []
    agent_rows = defaultdict(list)
    for r in rows:
        ag = r.get("agent", "?")
        if ag not in agent_order:
            agent_order.append(ag)
        agent_rows[ag].append(r)
    for ag in agent_order:
        agent_rows[ag].sort(key=lambda r: parse_exp_num(r.get("EXP-ID", "")))

    all_scores = [r["score"] for r in rows]
    y_min = max(0.0, min(all_scores) - 0.06)
    y_max = min(1.0, max(all_scores) + 0.04)
    if baseline:
        y_min = min(y_min, baseline - 0.02)
    if target:
        y_max = max(y_max, target + 0.02)
    score_range = y_max - y_min or 0.001

    n_agents = len(agent_order)
    total_H = PAD_T + n_agents * H_per_agent + PAD_B
    parts = [
        f'<svg viewBox="0 0 {W} {total_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;background:#16162a;border-radius:6px">'
    ]

    tooltip_data = []

    for lane_i, ag in enumerate(agent_order):
        color = AGENT_COLORS.get(ag, DEFAULT_COLOR)
        ag_rows = agent_rows[ag]
        n = len(ag_rows)
        if n == 0:
            continue
        lane_top = PAD_T + lane_i * H_per_agent
        lane_bot = lane_top + H_per_agent
        inner_top = lane_top + 8
        inner_bot = lane_bot - 16
        inner_H = inner_bot - inner_top

        def cx(idx, _n=n):
            if _n <= 1:
                return PAD_L + (W - PAD_L - PAD_R) * 0.5
            return PAD_L + idx / (_n - 1) * (W - PAD_L - PAD_R)

        def cy(score):
            frac = (score - y_min) / score_range
            return inner_bot - frac * inner_H

        lane_bg = "#1c1c30" if lane_i % 2 == 0 else "#1a1a2c"
        parts.append(f'<rect x="0" y="{lane_top}" width="{W}" height="{H_per_agent}" fill="{lane_bg}"/>')
        parts.append(
            f'<text x="8" y="{lane_top+14}" fill="{color}" font-size="11" '
            f'font-family="monospace" font-weight="bold">{html.escape(ag)}</text>'
        )

        # Y-axis ticks
        for tick_i in range(3):
            s = y_min + score_range * tick_i / 2
            y = cy(s)
            parts.append(
                f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                f'stroke="#2a2a44" stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{PAD_L-4}" y="{y+3:.1f}" text-anchor="end" fill="#555" '
                f'font-size="9" font-family="monospace">{s:.2f}</text>'
            )

        if lane_i == 0 and baseline:
            y = cy(baseline)
            parts.append(
                f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                f'stroke="#555" stroke-width="1" stroke-dasharray="5,3"/>'
            )
            parts.append(
                f'<text x="{W-PAD_R+3}" y="{y+3:.1f}" fill="#666" font-size="9">base</text>'
            )
        if lane_i == 0 and target:
            y = cy(target)
            parts.append(
                f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                f'stroke="#22c55e" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>'
            )
            parts.append(
                f'<text x="{W-PAD_R+3}" y="{y+3:.1f}" fill="#22c55e" font-size="9" opacity="0.7">target</text>'
            )

        keep_pts = [(i, r) for i, r in enumerate(ag_rows) if r.get("status") == "keep"]
        if len(keep_pts) >= 2:
            pts = " ".join(f"{cx(i):.1f},{cy(r['score']):.1f}" for i, r in keep_pts)
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="0.7"/>'
            )

        best_r = max(ag_rows, key=lambda r: r["score"])
        best_i = ag_rows.index(best_r)
        bx, by = cx(best_i), cy(best_r["score"])

        for i, r in enumerate(ag_rows):
            x, y = cx(i), cy(r["score"])
            eid = r.get("EXP-ID", "?")
            is_discard = r.get("status") == "discard"
            is_best = r is best_r
            pt_color = color
            opacity = "0.35" if is_discard else "1"
            radius = 6 if is_best else 4
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                f'fill="{pt_color}" opacity="{opacity}" '
                f'class="tl-pt" data-eid="{html.escape(eid)}" '
                f'data-tip-idx="{len(tooltip_data)}" style="cursor:pointer"/>'
            )
            step = max(1, n // 6)
            if i % step == 0 or is_best:
                parts.append(
                    f'<text x="{x:.1f}" y="{inner_bot+11:.1f}" text-anchor="middle" '
                    f'fill="#555" font-size="8" font-family="monospace">{html.escape(eid)}</text>'
                )
            tip = f'{eid} | {r["score"]:.4f} | {ag}\n{r.get("description","")}\n{r.get("design","")}'
            tooltip_data.append({"x": float(f"{x:.1f}"), "y": float(f"{y:.1f}"), "tip": tip})

        # Best star
        parts.append(
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="none" '
            f'stroke="#fbbf24" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{bx:.1f}" y="{by-11:.1f}" text-anchor="middle" '
            f'fill="#fbbf24" font-size="10" font-weight="bold">★{best_r["score"]:.4f}</text>'
        )

    parts.append("</svg>")
    tip_json = json.dumps(tooltip_data)
    script = f"""
<script>
(function(){{
  const tips = {tip_json};
  const tip = document.getElementById('tl-tip');
  document.querySelectorAll('.tl-pt').forEach(pt => {{
    const idx = parseInt(pt.getAttribute('data-tip-idx'));
    const eid = pt.getAttribute('data-eid');
    pt.addEventListener('mouseenter', e => {{
      const d = tips[idx];
      if (!d) return;
      tip.style.display = 'block';
      tip.innerText = d.tip;
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 10 + window.scrollY) + 'px';
    }});
    pt.addEventListener('mouseleave', () => tip.style.display = 'none');
    pt.addEventListener('click', () => {{
      document.querySelectorAll('.exp-item').forEach(el => {{
        if (el.dataset.eid === eid) {{
          el.click();
        }}
      }});
    }});
  }});
}})();
</script>
"""
    return "\n".join(parts) + script


# ── HTML page ─────────────────────────────────────────────────────────────────

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustLoop — __DOMAIN_NAME__</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg0:#0d1117;--bg1:#161b22;--bg2:#1c2128;--bg3:#21262d;
  --border:#30363d;--text:#c9d1d9;--dim:#6e7681;--dimmer:#484f58;
  --accent:#58a6ff;--green:#3fb950;--orange:#d29922;--red:#f78166;
  --yellow:#fbbf24;--purple:#a371f7;
}}
html,body{{height:100%;overflow:hidden}}
body{{background:var(--bg0);color:var(--text);font-family:-apple-system,ui-monospace,monospace;font-size:13px;display:flex;flex-direction:column}}

/* ── TOPBAR ── */
.topbar{{background:var(--bg1);border-bottom:1px solid var(--border);padding:0 16px;height:44px;display:flex;align-items:center;gap:12px;flex-shrink:0}}
.topbar .logo{{color:var(--accent);font-weight:700;font-size:14px;letter-spacing:.03em}}
.topbar .domain-badge{{background:var(--bg3);border:1px solid var(--border);padding:3px 10px;border-radius:12px;font-size:11px;color:var(--dim)}}
.topbar .best-badge{{background:color-mix(in srgb,var(--yellow) 10%,transparent);border:1px solid color-mix(in srgb,var(--yellow) 30%,transparent);padding:3px 10px;border-radius:12px;font-size:11px;color:var(--yellow)}}
.topbar .status-dot{{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.topbar .ml-auto{{margin-left:auto;display:flex;align-items:center;gap:10px}}
.topbar .refresh-btn{{background:transparent;border:1px solid var(--border);color:var(--dim);padding:4px 10px;border-radius:6px;cursor:pointer;font-size:11px}}
.topbar .refresh-btn:hover{{border-color:var(--accent);color:var(--accent)}}
.topbar .last-update{{font-size:10px;color:var(--dimmer)}}

/* ── BODY LAYOUT ── */
.layout{{display:flex;flex:1;overflow:hidden}}

/* ── LEFT NAV ── */
.left-nav{{width:220px;flex-shrink:0;border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
.nav-section{{padding:10px 8px 4px;color:var(--dimmer);font-size:10px;letter-spacing:.08em;text-transform:uppercase}}
.agent-btns{{padding:4px 8px 8px;display:flex;flex-direction:column;gap:3px}}
.agent-btn{{padding:6px 10px;border-radius:6px;border:1px solid transparent;cursor:pointer;font-size:12px;font-family:monospace;background:transparent;color:var(--dim);text-align:left;transition:.1s;display:flex;align-items:center;gap:7px}}
.agent-btn:hover{{background:var(--bg2);border-color:var(--border)}}
.agent-btn.active{{background:var(--bg2);border-color:var(--border);color:var(--text)}}
.agent-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.agent-btn.all-btn{{color:var(--accent)}}
.exp-list{{flex:1;overflow-y:auto;padding:0 8px 8px}}
.exp-list::-webkit-scrollbar{{width:4px}}.exp-list::-webkit-scrollbar-thumb{{background:var(--border)}}
.exp-item{{padding:7px 8px;border-radius:5px;cursor:pointer;border:1px solid transparent;margin-bottom:2px;transition:.1s}}
.exp-item:hover{{background:var(--bg2);border-color:var(--border)}}
.exp-item.selected{{background:var(--bg2);border-color:var(--accent)}}
.exp-item.hidden{{display:none}}
.exp-row{{display:flex;justify-content:space-between;align-items:center}}
.exp-id{{font-size:11px;font-weight:600;color:var(--text)}}
.exp-score{{font-size:12px;font-weight:700}}
.keep .exp-score{{color:var(--green)}}.discard .exp-score{{color:var(--red)}}
.exp-bar{{height:2px;background:var(--bg3);border-radius:1px;margin:4px 0}}
.exp-bar-fill{{height:100%;border-radius:1px}}
.keep .exp-bar-fill{{background:var(--green)}}.discard .exp-bar-fill{{background:var(--red)}}
.exp-ag{{font-size:10px;color:var(--dimmer)}}
.exp-desc{{font-size:10px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px}}
.log-tail{{height:130px;flex-shrink:0;border-top:1px solid var(--border);padding:8px;overflow-y:auto;font-size:10px;color:var(--dimmer);font-family:monospace;line-height:1.5}}
.log-tail::-webkit-scrollbar{{width:3px}}.log-tail::-webkit-scrollbar-thumb{{background:var(--border)}}

/* ── CENTRE HERO ── */
.centre{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}}
.view-tabs{{background:var(--bg1);border-bottom:1px solid var(--border);padding:0 16px;display:flex;gap:0;flex-shrink:0;align-items:flex-end;height:40px}}
.view-tab{{padding:0 16px;height:36px;border:none;background:transparent;color:var(--dim);cursor:pointer;font-size:12px;font-family:monospace;border-bottom:2px solid transparent;margin-bottom:-1px;transition:.1s}}
.view-tab:hover{{color:var(--text)}}
.view-tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.hero{{flex:1;overflow:auto;padding:16px}}
.hero::-webkit-scrollbar{{width:5px}}.hero::-webkit-scrollbar-thumb{{background:var(--border)}}

/* hero content views */
.hero-view{{display:none}}
.hero-view.active{{display:block}}

/* timeline */
#tl-tip{{position:fixed;background:var(--bg1);border:1px solid var(--border);color:var(--text);font-size:11px;font-family:monospace;padding:8px 10px;border-radius:4px;white-space:pre;pointer-events:none;display:none;z-index:999;max-width:380px;line-height:1.5}}

/* trace */
.trace-empty{{color:var(--dim);padding:20px;font-style:italic}}
.trace-turn{{padding:10px 12px;border-radius:6px;margin-bottom:8px;border:1px solid var(--border);background:var(--bg1);line-height:1.55}}
.trace-turn.type-text{{border-color:#388bfd44}}
.trace-turn.type-tool_use{{border-color:#d2992244}}
.trace-header{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.trace-agent{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px}}
.trace-ts{{font-size:10px;color:var(--dimmer)}}
.trace-tool-badge{{font-size:10px;background:var(--bg3);padding:2px 6px;border-radius:3px;color:var(--orange)}}
.trace-text{{font-size:11px;color:var(--text);font-family:monospace;white-space:pre-wrap;word-break:break-word}}

/* blackboard */
.bb-content{{background:var(--bg1);border:1px solid var(--border);border-radius:6px;padding:16px;font-size:12px;line-height:1.7;white-space:pre-wrap;font-family:monospace;color:#b0bac6;min-height:200px}}
.bb-content h1,.bb-content h2,.bb-content h3{{color:var(--accent);margin:12px 0 6px}}

/* steer */
.steer-panel{{max-width:680px}}
.steer-title{{font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px}}
.steer-subtitle{{font-size:12px;color:var(--dim);margin-bottom:16px;line-height:1.5}}
.steer-textarea{{width:100%;height:160px;background:var(--bg1);border:1px solid var(--border);border-radius:6px;padding:12px;color:var(--text);font-family:monospace;font-size:12px;resize:vertical;outline:none;transition:.1s}}
.steer-textarea:focus{{border-color:var(--accent)}}
.steer-actions{{display:flex;align-items:center;gap:10px;margin-top:10px}}
.steer-send{{background:var(--accent);color:#0d1117;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;font-family:monospace;transition:.1s}}
.steer-send:hover{{background:#79b8ff}}
.steer-send:disabled{{opacity:.4;cursor:not-allowed}}
.steer-status{{font-size:12px;color:var(--dim)}}
.steer-status.ok{{color:var(--green)}}.steer-status.err{{color:var(--red)}}
.steer-preview{{margin-top:16px;background:var(--bg1);border:1px solid var(--border);border-radius:6px;padding:12px}}
.steer-preview-title{{font-size:10px;color:var(--dimmer);letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px}}
.steer-history{{font-size:11px;font-family:monospace;color:var(--dim);line-height:1.6;white-space:pre-wrap;max-height:180px;overflow-y:auto}}

/* ── BOTTOM CHAT ── */
.chat-bar{{border-top:1px solid var(--border);background:var(--bg1);padding:10px 16px;display:flex;gap:8px;flex-shrink:0;align-items:center}}
.chat-input{{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-family:monospace;font-size:12px;outline:none;transition:.1s}}
.chat-input:focus{{border-color:var(--accent)}}
.chat-send{{background:var(--bg3);border:1px solid var(--border);color:var(--dim);padding:8px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-family:monospace;transition:.1s}}
.chat-send:hover{{border-color:var(--accent);color:var(--accent)}}
.chat-resp{{font-size:11px;color:var(--dim);max-width:460px;line-height:1.5;white-space:pre-wrap}}

/* ── RIGHT FILTERS ── */
.right-filters{{width:180px;flex-shrink:0;border-left:1px solid var(--border);padding:12px 10px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}}
.filter-group label{{display:block;font-size:10px;color:var(--dimmer);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}}
.filter-group select,.filter-group input[type=range]{{width:100%;background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:5px 6px;font-size:11px;outline:none}}
.filter-group input[type=range]{{padding:4px 0}}
.score-label{{font-size:11px;color:var(--dim);margin-top:3px}}
.filter-group .filter-tags{{display:flex;flex-direction:column;gap:3px}}
.filter-tag{{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim);cursor:pointer}}
.filter-tag input{{accent-color:var(--accent)}}
.divider{{border:none;border-top:1px solid var(--border)}}
.stats-box{{background:var(--bg2);border:1px solid var(--border);border-radius:5px;padding:10px}}
.stat-row{{display:flex;justify-content:space-between;margin-bottom:4px;font-size:11px}}
.stat-label{{color:var(--dim)}}.stat-val{{color:var(--text);font-weight:600}}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <span class="logo">TrustLoop</span>
  <span class="domain-badge">__DOMAIN_NAME__</span>
  <span class="best-badge" id="best-badge">best —</span>
  <div class="status-dot" id="status-dot" title="polling"></div>
  <div class="ml-auto">
    <span class="last-update" id="last-update">—</span>
    <button class="refresh-btn" onclick="loadAll()">⟳ refresh</button>
  </div>
</div>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- LEFT NAV -->
  <div class="left-nav">
    <div class="nav-section">Agents</div>
    <div class="agent-btns" id="agent-btns">
      <button class="agent-btn all-btn active" data-agent="all" onclick="selectAgent('all',this)">
        <span class="agent-dot" style="background:#58a6ff"></span>All agents
      </button>
    </div>
    <div class="nav-section">Experiments</div>
    <div class="exp-list" id="exp-list"></div>
    <div class="nav-section">Live logs</div>
    <div class="log-tail" id="log-tail">loading…</div>
  </div>

  <!-- CENTRE -->
  <div class="centre">
    <div class="view-tabs">
      <button class="view-tab active" data-view="timeline" onclick="switchView('timeline',this)">Timeline</button>
      <button class="view-tab" data-view="trace"    onclick="switchView('trace',this)">Trace</button>
      <button class="view-tab" data-view="blackboard" onclick="switchView('blackboard',this)">Blackboard</button>
      <button class="view-tab" data-view="steer"    onclick="switchView('steer',this)">⚡ Steer</button>
    </div>
    <div class="hero">
      <div class="hero-view active" id="view-timeline">
        <div id="timeline-svg">loading timeline…</div>
      </div>
      <div class="hero-view" id="view-trace">
        <div id="trace-content"><p class="trace-empty">← select an experiment to see its reasoning trace</p></div>
      </div>
      <div class="hero-view" id="view-blackboard">
        <pre class="bb-content" id="bb-content">loading…</pre>
      </div>
      <div class="hero-view" id="view-steer">
        <div class="steer-panel">
          <div class="steer-title">⚡ Steering intervention</div>
          <div class="steer-subtitle">
            Write a message to send directly to the agents' shared blackboard.<br>
            It will be signed <code>[HUMAN __TIMESTAMP_PLACEHOLDER__]</code> and appended immediately.<br>
            Agents read the blackboard every loop — they will see this on their next turn.
          </div>
          <textarea class="steer-textarea" id="steer-text" placeholder="e.g. Stop exploring curriculum learning — return to GRPO baseline and push past 0.760 before trying new axes."></textarea>
          <div class="steer-actions">
            <button class="steer-send" id="steer-btn" onclick="sendSteer()">Send to Blackboard</button>
            <span class="steer-status" id="steer-status"></span>
          </div>
          <div class="steer-preview">
            <div class="steer-preview-title">Recent interventions</div>
            <div class="steer-history" id="steer-history">—</div>
          </div>
        </div>
      </div>
    </div>
    <div class="chat-bar">
      <input class="chat-input" id="chat-input" placeholder="Ask about this run… (e.g. why did EXP-008 plateau?)" onkeydown="if(event.key==='Enter')sendChat()">
      <button class="chat-send" onclick="sendChat()">Ask</button>
      <span class="chat-resp" id="chat-resp"></span>
    </div>
  </div>

  <!-- RIGHT FILTERS -->
  <div class="right-filters">
    <div class="filter-group">
      <label>Min score</label>
      <input type="range" id="score-min" min="0" max="1" step="0.01" value="0" oninput="applyFilters()">
      <div class="score-label" id="score-min-label">≥ 0.00</div>
    </div>
    <div class="filter-group">
      <label>Status</label>
      <div class="filter-tags">
        <label class="filter-tag"><input type="checkbox" id="f-keep" checked onchange="applyFilters()">keep</label>
        <label class="filter-tag"><input type="checkbox" id="f-discard" checked onchange="applyFilters()">discard</label>
      </div>
    </div>
    <div class="filter-group">
      <label>Sort by</label>
      <select id="sort-by" onchange="applyFilters()">
        <option value="id">EXP-ID</option>
        <option value="score">Score ↓</option>
        <option value="agent">Agent</option>
      </select>
    </div>
    <hr class="divider">
    <div class="stats-box" id="stats-box">
      <div class="stat-row"><span class="stat-label">Total</span><span class="stat-val" id="s-total">—</span></div>
      <div class="stat-row"><span class="stat-label">Keep</span><span class="stat-val" id="s-keep">—</span></div>
      <div class="stat-row"><span class="stat-label">Best</span><span class="stat-val" id="s-best">—</span></div>
      <div class="stat-row"><span class="stat-label">Agents</span><span class="stat-val" id="s-agents">—</span></div>
    </div>
  </div>

</div>

<div id="tl-tip"></div>

<script>
// ── state ────────────────────────────────────────────────────────
const AGENT_COLORS = __AGENT_COLORS_JSON__;
let allRows   = [];
let curAgent  = 'all';
let curExp    = null;
let curView   = 'timeline';

// ── view switching ───────────────────────────────────────────────
function switchView(v, btn) {{
  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.hero-view').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('view-' + v).classList.add('active');
  curView = v;
  if (v === 'trace' && curExp) loadTrace(curExp);
  if (v === 'blackboard') loadBlackboard();
  if (v === 'steer') loadSteerHistory();
}}

// ── agent filter ─────────────────────────────────────────────────
function selectAgent(ag, btn) {{
  curAgent = ag;
  document.querySelectorAll('.agent-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

// ── load all data ────────────────────────────────────────────────
async function loadAll() {{
  try {{
    const [resData, logData] = await Promise.all([
      fetch('/api/results').then(r => r.json()),
      fetch('/api/logs').then(r => r.text()),
    ]);
    allRows = resData;
    renderAgentBtns();
    applyFilters();
    updateStats();
    renderTimeline();
    document.getElementById('log-tail').textContent = logData || '(no logs)';
    document.getElementById('log-tail').scrollTop = 9999;
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    if (curView === 'blackboard') loadBlackboard();
  }} catch(e) {{
    console.error(e);
  }}
}}

function renderAgentBtns() {{
  const agents = [...new Set(allRows.map(r => r.agent || '?'))].sort();
  const container = document.getElementById('agent-btns');
  // keep the "all" button, rebuild the rest
  const allBtn = container.querySelector('[data-agent=all]');
  container.innerHTML = '';
  container.appendChild(allBtn);
  for (const ag of agents) {{
    const color = AGENT_COLORS[ag] || '#8b949e';
    const btn = document.createElement('button');
    btn.className = 'agent-btn' + (curAgent === ag ? ' active' : '');
    btn.dataset.agent = ag;
    btn.innerHTML = `<span class="agent-dot" style="background:${{color}}"></span>${{ag}}`;
    btn.onclick = () => selectAgent(ag, btn);
    container.appendChild(btn);
  }}
}}

function parseExpNum(id) {{
  const m = (id||'').match(/EXP-(\\d+)([a-z]*)/);
  return m ? [parseInt(m[1]), m[2]] : [9999, id];
}}

function applyFilters() {{
  const minScore = parseFloat(document.getElementById('score-min').value);
  document.getElementById('score-min-label').textContent = '≥ ' + minScore.toFixed(2);
  const showKeep = document.getElementById('f-keep').checked;
  const showDiscard = document.getElementById('f-discard').checked;
  const sortBy = document.getElementById('sort-by').value;

  let rows = allRows.filter(r => {{
    if (curAgent !== 'all' && r.agent !== curAgent) return false;
    if (r.score < minScore) return false;
    const status = (r.status||'').toLowerCase();
    if (!showKeep && status === 'keep') return false;
    if (!showDiscard && status !== 'keep') return false;
    return true;
  }});

  if (sortBy === 'score') rows = [...rows].sort((a,b) => b.score - a.score);
  else if (sortBy === 'agent') rows = [...rows].sort((a,b) => (a.agent||'').localeCompare(b.agent||''));
  else rows = [...rows].sort((a,b) => {{
    const [na,sa] = parseExpNum(a['EXP-ID']); const [nb,sb] = parseExpNum(b['EXP-ID']);
    return na !== nb ? na - nb : sa.localeCompare(sb);
  }});

  renderExpList(rows);
  updateStats();
}}

function renderExpList(rows) {{
  const container = document.getElementById('exp-list');
  if (rows.length === 0) {{
    container.innerHTML = '<div style="color:var(--dimmer);font-size:11px;padding:12px 4px">No experiments match filters.</div>';
    return;
  }}
  const scores = rows.map(r => r.score);
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const scoreRange = maxScore - minScore || 0.001;

  container.innerHTML = rows.map(r => {{
    const eid = r['EXP-ID'] || '?';
    const status = (r.status||'keep').toLowerCase() === 'keep' ? 'keep' : 'discard';
    const ag = r.agent || '?';
    const color = AGENT_COLORS[ag] || '#8b949e';
    const pct = ((r.score - minScore) / scoreRange * 100).toFixed(1);
    const isSelected = eid === curExp ? ' selected' : '';
    return `<div class="exp-item ${{status}}${{isSelected}}" data-eid="${{eid}}" onclick="selectExp('${{eid}}',this)">
      <div class="exp-row">
        <span class="exp-id">${{eid}}</span>
        <span class="exp-score">${{r.score.toFixed(4)}}</span>
      </div>
      <div class="exp-bar"><div class="exp-bar-fill" style="width:${{pct}}%;background:${{status==='keep'?'#3fb950':'#f78166'}}"></div></div>
      <div class="exp-row">
        <span class="exp-ag" style="color:${{color}}">${{ag}}</span>
        <span class="exp-desc">${{(r.description||'').slice(0,30)}}</span>
      </div>
    </div>`;
  }}).join('');
}}

function updateStats() {{
  const keep = allRows.filter(r => (r.status||'').toLowerCase()==='keep');
  const best = allRows.length ? Math.max(...allRows.map(r=>r.score)) : 0;
  const agents = new Set(allRows.map(r=>r.agent||'?'));
  document.getElementById('s-total').textContent = allRows.length;
  document.getElementById('s-keep').textContent = keep.length;
  document.getElementById('s-best').textContent = best ? best.toFixed(4) : '—';
  document.getElementById('s-agents').textContent = agents.size;
  document.getElementById('best-badge').textContent = best ? `best ${best.toFixed(4)}` : 'best —';
}}

// ── timeline ─────────────────────────────────────────────────────
async function renderTimeline() {{
  const resp = await fetch('/api/timeline');
  const svg = await resp.text();
  document.getElementById('timeline-svg').innerHTML = svg;
}}

// ── experiment selection ──────────────────────────────────────────
function selectExp(eid, el) {{
  curExp = eid;
  document.querySelectorAll('.exp-item').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
  if (curView === 'trace') loadTrace(eid);
}}

// ── trace view ───────────────────────────────────────────────────
async function loadTrace(eid) {{
  const container = document.getElementById('trace-content');
  container.innerHTML = '<p class="trace-empty">loading trace for ' + eid + '…</p>';
  try {{
    const turns = await fetch('/api/trace?exp=' + encodeURIComponent(eid)).then(r => r.json());
    if (!turns.length) {{
      container.innerHTML = '<p class="trace-empty">No turns found mentioning ' + eid + '</p>';
      return;
    }}
    container.innerHTML = turns.map(t => {{
      const color = AGENT_COLORS[t.agent] || '#8b949e';
      const typeClass = 'type-' + (t.type || 'text');
      const badge = t.name ? `<span class="trace-tool-badge">${{t.name}}</span>` : '';
      const text = (t.text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      return `<div class="trace-turn ${{typeClass}}">
        <div class="trace-header">
          <span class="trace-agent" style="background:color-mix(in srgb,${{color}} 15%,transparent);color:${{color}}">${{t.agent}}</span>
          ${{badge}}
          <span class="trace-ts">${{t.ts}}</span>
        </div>
        <div class="trace-text">${{text}}</div>
      </div>`;
    }}).join('');
  }} catch(e) {{
    container.innerHTML = '<p class="trace-empty">error loading trace: ' + e + '</p>';
  }}
}}

// ── blackboard view ──────────────────────────────────────────────
async function loadBlackboard() {{
  try {{
    const text = await fetch('/api/blackboard').then(r => r.text());
    document.getElementById('bb-content').textContent = text;
  }} catch(e) {{
    document.getElementById('bb-content').textContent = 'error: ' + e;
  }}
}}

// ── steer / intervention ──────────────────────────────────────────
async function sendSteer() {{
  const text = document.getElementById('steer-text').value.trim();
  if (!text) return;
  const btn = document.getElementById('steer-btn');
  const status = document.getElementById('steer-status');
  btn.disabled = true;
  status.className = 'steer-status';
  status.textContent = 'sending…';
  try {{
    const resp = await fetch('/api/steer', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{message: text}})
    }});
    const data = await resp.json();
    if (data.ok) {{
      status.className = 'steer-status ok';
      status.textContent = '✓ appended to blackboard';
      document.getElementById('steer-text').value = '';
      loadSteerHistory();
    }} else {{
      status.className = 'steer-status err';
      status.textContent = '✗ ' + (data.error || 'failed');
    }}
  }} catch(e) {{
    status.className = 'steer-status err';
    status.textContent = '✗ ' + e;
  }}
  btn.disabled = false;
}}

async function loadSteerHistory() {{
  try {{
    const resp = await fetch('/api/steer_history');
    const entries = await resp.json();
    const el = document.getElementById('steer-history');
    if (!entries.length) {{
      el.textContent = '(no interventions yet)';
    }} else {{
      el.textContent = entries.map(e => `${{e.ts}} — ${{e.preview}}`).join('\n');
    }}
  }} catch(e) {{}}
}}

// ── chat ─────────────────────────────────────────────────────────
async function sendChat() {{
  const q = document.getElementById('chat-input').value.trim();
  if (!q) return;
  const resp_el = document.getElementById('chat-resp');
  resp_el.textContent = '…';
  try {{
    const data = await fetch('/api/chat', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{q, exp: curExp}})
    }}).then(r => r.json());
    resp_el.textContent = data.answer || '(no answer)';
  }} catch(e) {{
    resp_el.textContent = 'error: ' + e;
  }}
}}

// ── polling ──────────────────────────────────────────────────────
loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>
"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, status=200, ct="text/plain"):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        cfg = self.server.config

        if path == "/":
            self.send_text(self.server.page_html, ct="text/html; charset=utf-8")

        elif path == "/api/results":
            rows = load_results(cfg["results"])
            self.send_json(rows)

        elif path == "/api/logs":
            tail = load_logs_tail(cfg["domain"])
            self.send_text(tail)

        elif path == "/api/blackboard":
            text = load_blackboard(cfg["blackboard"])
            self.send_text(text)

        elif path == "/api/timeline":
            rows = load_results(cfg["results"])
            svg = build_timeline_svg(rows, baseline=cfg.get("baseline"), target=cfg.get("target"))
            self.send_text(svg, ct="text/html")

        elif path == "/api/trace":
            qs = parse_qs(parsed.query)
            exp_id = (qs.get("exp", [""])[0]).strip()
            if not exp_id:
                self.send_json([])
                return
            turns = load_trace_for_exp(cfg.get("logs_dir", cfg["domain"]), exp_id)
            self.send_json(turns)

        elif path == "/api/steer_history":
            bb_text = load_blackboard(cfg["blackboard"])
            entries = []
            for m in re.finditer(r"\[HUMAN (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]\n\n(.+?)(?=\n\n---|\Z)",
                                  bb_text, re.DOTALL):
                ts, body = m.group(1), m.group(2).strip()
                entries.append({"ts": ts, "preview": body[:80].replace("\n", " ")})
            self.send_json(entries)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        cfg = self.server.config
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/steer":
            try:
                data = json.loads(body)
                msg = data.get("message", "").strip()
                if not msg:
                    self.send_json({"ok": False, "error": "empty message"}, 400)
                    return
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                bb_path = Path(cfg["blackboard"])
                entry = f"\n\n---\n\n## [HUMAN {ts} UTC]\n\n{msg}\n"
                with open(bb_path, "a") as f:
                    f.write(entry)
                print(f"[steer] wrote to {bb_path}: {msg[:60]}")
                self.send_json({"ok": True, "ts": ts})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 500)

        elif path == "/api/chat":
            # Simple context-aware reply — no live Claude call, just run stats
            try:
                data = json.loads(body)
                q = data.get("q", "")
                exp = data.get("exp", "")
                rows = load_results(cfg["results"])
                best = max(rows, key=lambda r: r["score"]) if rows else None
                bb_snippet = load_blackboard(cfg["blackboard"])[:800]
                if best:
                    answer = (
                        f"Run has {len(rows)} experiments. "
                        f"Best: {best.get('EXP-ID','?')} score={best['score']:.4f} "
                        f"by {best.get('agent','?')}. "
                        f"Blackboard: {bb_snippet[:200]}…"
                    )
                else:
                    answer = "No results yet."
                self.send_json({"answer": answer})
            except Exception as e:
                self.send_json({"answer": f"error: {e}"})

        else:
            self.send_response(404)
            self.end_headers()


def build_page(domain_name, agent_colors):
    page = PAGE_TEMPLATE
    page = page.replace("__DOMAIN_NAME__", html.escape(domain_name))
    page = page.replace("__AGENT_COLORS_JSON__", json.dumps(agent_colors))
    page = page.replace("__TIMESTAMP_PLACEHOLDER__", "YYYY-MM-DD HH:MM:SS")
    # unescape {{ }} left over from former .format() template
    page = page.replace("{{", "{").replace("}}", "}")
    return page


def main():
    parser = argparse.ArgumentParser(description="TrustLoop verification dashboard")
    parser.add_argument("--domain", required=True, help="Domain directory (e.g. domains/rrma-r1)")
    parser.add_argument("--results", help="Path to results.tsv (default: <domain>/results.tsv)")
    parser.add_argument("--blackboard", help="Path to blackboard.md (default: <domain>/blackboard.md)")
    parser.add_argument("--logs-dir", help="Directory with *.jsonl session logs")
    parser.add_argument("--baseline", type=float, help="Baseline score line on timeline")
    parser.add_argument("--target", type=float, help="Target score line on timeline")
    parser.add_argument("--port", type=int, default=7432)
    args = parser.parse_args()

    domain_path = Path(args.domain)
    domain_name = domain_path.name

    results_path = args.results or str(domain_path / "results.tsv")
    blackboard_path = args.blackboard or str(domain_path / "blackboard.md")
    logs_dir = args.logs_dir or str(domain_path)

    # find jsonl logs — also check ~/.claude/projects if nothing local
    if not list(Path(logs_dir).glob("*.jsonl")):
        home_logs = Path.home() / ".claude" / "projects"
        if home_logs.exists():
            print(f"  No .jsonl in {logs_dir} — trace view will search {home_logs}")

    config = {
        "domain": str(domain_path),
        "results": results_path,
        "blackboard": blackboard_path,
        "logs_dir": logs_dir,
        "baseline": args.baseline,
        "target": args.target,
    }

    page_html = build_page(domain_name, AGENT_COLORS)

    server = HTTPServer(("127.0.0.1", args.port), DashboardHandler)
    server.config = config
    server.page_html = page_html

    url = f"http://localhost:{args.port}"
    print(f"\nTrustLoop Dashboard")
    print(f"  domain    : {domain_path}")
    print(f"  results   : {results_path}")
    print(f"  blackboard: {blackboard_path}")
    print(f"  logs dir  : {logs_dir}")
    print(f"\n  → {url}\n")
    print("  Ctrl+C to stop\n")

    # auto-open browser
    try:
        import subprocess
        subprocess.Popen(["open", url])
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

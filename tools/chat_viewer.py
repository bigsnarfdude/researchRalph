#!/usr/bin/env python3
"""
RRMA chat viewer — agent selector buttons on top, experiments filtered in left tray.
Usage: python3 chat_viewer.py <logs_dir> [--results results.tsv] [--output viewer.html]
"""
import json, argparse, re
from pathlib import Path
from datetime import datetime

AGENT_COLORS = {"agent0": "#388bfd", "agent1": "#3fb950", "agent2": "#d29922", "agent3": "#f78166"}
DEFAULT_COLOR = "#8b949e"

def parse_session(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: events.append(json.loads(line))
            except: pass
    return events

def extract_turns(events):
    turns = []
    agent_id = "agent?"
    for e in events:
        t = e.get("type"); msg = e.get("message", {}); ts = e.get("timestamp","")
        if t == "user" and e.get("parentUuid") is None:
            content = msg.get("content","")
            if isinstance(content, str):
                m = re.search(r"You are (agent\d+)", content)
                if m: agent_id = m.group(1)
        elif t == "assistant":
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text","").strip():
                    turns.append({"type":"think","text":block["text"].strip(),"ts":ts,"agent":agent_id})
                elif block.get("type") == "tool_use":
                    name = block.get("name",""); inp = block.get("input",{})
                    if name == "Bash": text = inp.get("command","")[:300]
                    elif name in ("Write","Edit"): text = inp.get("file_path","")
                    elif name == "Read": text = inp.get("file_path","")
                    else: text = str(inp)[:100]
                    turns.append({"type":"tool","name":name,"text":text,"ts":ts,"agent":agent_id})
        elif t == "tool":
            for block in msg.get("content",[]):
                if block.get("type") == "tool_result":
                    for inner in block.get("content",[]):
                        if inner.get("type") == "text" and inner["text"].strip():
                            turns.append({"type":"result","text":inner["text"].strip()[:800],"ts":ts,"agent":agent_id})
    return turns, agent_id

def fmt_ts(ts):
    try: return datetime.fromisoformat(ts.replace("Z","+00:00")).strftime("%H:%M:%S")
    except: return ts[:8]

def parse_results(path):
    results = []
    if not path or not path.exists(): return results
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5:
            try:
                results.append({
                    "id": parts[0].strip(),
                    "score": float(parts[1].strip()),
                    "status": "discard" if "discard" in parts[3].lower() else "keep",
                    "desc": parts[4].strip(),
                    "agent": parts[5].strip() if len(parts) > 5 else ""
                })
            except: pass
    return results

def bubble_html(turn):
    text = turn["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    ts = f'<span class="ts">{fmt_ts(turn["ts"])}</span>'
    if turn["type"] == "think":
        return f'<div class="bubble think">{text}{ts}</div>'
    elif turn["type"] == "tool":
        name = turn.get("name","")
        return f'<div class="bubble tool"><span class="tool-badge">{name}</span>{text}{ts}</div>'
    elif turn["type"] == "result":
        if len(text) > 250:
            preview = text[:250] + "..."
            return f'<div class="bubble result long"><span class="preview">{preview}</span><span class="full">{text}</span>{ts}<span class="expand-hint">▸ click to expand</span></div>'
        return f'<div class="bubble result">{text}{ts}</div>'
    return ""

def build_html(logs_dir, results_tsv=None):
    # Load sessions
    sessions = []
    for jsonl in sorted(logs_dir.glob("*.jsonl")):
        events = parse_session(jsonl)
        turns, agent_id = extract_turns(events)
        if not turns: continue
        start_ts = next((e.get("timestamp","") for e in events if e.get("timestamp")), "")
        sessions.append({"file": jsonl.stem, "agent": agent_id, "turns": turns, "start": start_ts})
    sessions.sort(key=lambda s: s["start"])

    # Deduplicate session labels within same agent
    counts = {}
    for s in sessions: counts[s["agent"]] = counts.get(s["agent"],0)+1
    seen = {}
    for s in sessions:
        base = s["agent"]
        if counts[base] > 1:
            seen[base] = seen.get(base,0)+1
            s["label"] = f"{base} #{seen[base]}"
        else:
            s["label"] = base

    results = parse_results(results_tsv)
    agents = sorted(set(s["agent"] for s in sessions))

    # Build session HTML panels
    session_divs = []
    for i, sess in enumerate(sessions):
        color = AGENT_COLORS.get(sess["agent"], DEFAULT_COLOR)
        bubbles = [bubble_html(t) for t in sess["turns"] if bubble_html(t)]
        session_divs.append(
            f'<div class="session" id="sess-{i}" data-agent="{sess["agent"]}" style="--c:{color}">'
            f'<div class="session-header">{sess["label"]} &nbsp;<span class="ts">{fmt_ts(sess["start"])}</span></div>'
            + "\n".join(bubbles)
            + "</div>"
        )

    # Agent button colors for JS
    agent_colors = {a: AGENT_COLORS.get(a, DEFAULT_COLOR) for a in agents}

    sessions_json = json.dumps([{"label":s["label"],"agent":s["agent"],"file":s["file"]} for s in sessions])
    results_json  = json.dumps(results)
    agents_json   = json.dumps(agents)
    colors_json   = json.dumps(agent_colors)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RRMA-R1 Agent Traces</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,monospace;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column;overflow:hidden}}

/* HEADER */
.header{{background:#161b22;border-bottom:1px solid #30363d;padding:10px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0}}
.header h1{{color:#58a6ff;font-size:0.95em;font-weight:600;margin-right:8px}}
.agent-btn{{padding:5px 16px;border-radius:20px;border:1.5px solid #30363d;cursor:pointer;font-size:0.8em;background:transparent;color:#8b949e;font-family:monospace;transition:.15s}}
.agent-btn:hover{{border-color:var(--c);color:var(--c)}}
.agent-btn.active{{border-color:var(--c);color:var(--c);background:color-mix(in srgb,var(--c) 12%,transparent)}}
.meta{{margin-left:auto;color:#484f58;font-size:0.75em}}

/* BODY */
.main{{display:flex;flex:1;overflow:hidden}}

/* LEFT TRAY */
.tray{{width:250px;flex-shrink:0;border-right:1px solid #30363d;overflow-y:auto;padding:10px 8px;display:flex;flex-direction:column;gap:3px}}
.tray-empty{{color:#484f58;font-size:0.78em;padding:16px 8px;text-align:center}}
.exp-item{{padding:9px 10px;border-radius:6px;cursor:pointer;border:1px solid transparent;transition:.12s}}
.exp-item:hover{{background:#161b22;border-color:#30363d}}
.exp-item.selected{{background:#161b22;border-color:var(--agent-color,#388bfd)}}
.exp-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}}
.exp-id{{font-size:0.78em;color:#c9d1d9;font-weight:600}}
.exp-score{{font-size:0.82em;font-weight:700}}
.keep .exp-score{{color:#3fb950}}.discard .exp-score{{color:#f78166}}
.exp-bar{{height:3px;background:#21262d;border-radius:2px;margin-bottom:4px}}
.exp-bar-fill{{height:100%;border-radius:2px}}
.keep .exp-bar-fill{{background:#3fb950}}.discard .exp-bar-fill{{background:#f78166}}
.exp-desc{{font-size:0.71em;color:#6e7681;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}

/* SESSIONS */
.chat-area{{flex:1;overflow:hidden;position:relative}}
.session{{display:none;flex-direction:column;gap:10px;overflow-y:auto;padding:20px;height:100%}}
.session.active{{display:flex}}
.session-header{{font-size:0.73em;color:#6e7681;text-align:center;padding:0 0 12px;border-bottom:1px solid #21262d;margin-bottom:4px;flex-shrink:0}}
.bubble{{max-width:78%;padding:10px 14px;border-radius:10px;font-size:0.82em;line-height:1.55;word-break:break-word}}
.bubble.think{{align-self:flex-start;background:#161b22;border:1px solid #30363d;border-left:3px solid var(--c,#388bfd)}}
.bubble.tool{{align-self:flex-start;background:#0d1117;border:1px solid #21262d;font-family:monospace;font-size:0.77em;color:#8b949e}}
.tool-badge{{display:inline-block;font-size:0.68em;font-weight:700;color:#e3b341;margin-right:6px;text-transform:uppercase}}
.bubble.result{{align-self:flex-end;background:#1c2128;border:1px solid #30363d;font-family:monospace;font-size:0.75em;color:#8b949e;max-width:84%}}
.bubble.result.long{{cursor:pointer}}
.bubble.result.long .full{{display:none}}
.bubble.result.long.expanded .preview{{display:none}}
.bubble.result.long.expanded .full{{display:block}}
.expand-hint{{display:block;font-size:0.68em;color:#484f58;margin-top:4px}}
.ts{{display:block;font-size:0.67em;color:#484f58;margin-top:5px}}

.placeholder{{display:flex;align-items:center;justify-content:center;height:100%;color:#484f58;font-size:0.85em}}
.exp-item.dimmed{{opacity:0.35}}
.exp-agent{{font-size:0.68em;color:#6e7681;margin-left:4px}}
</style>
</head>
<body>
<div class="header">
  <h1>RRMA-R1</h1>
  <div id="agent-btns"></div>
  <span class="meta">{len(sessions)} sessions · {len(results)} experiments</span>
</div>
<div class="main">
  <div class="tray" id="tray"><div class="tray-empty">Select an agent</div></div>
  <div class="chat-area" id="chat-area">
    {"".join(session_divs)}
    <div class="placeholder" id="placeholder">← select an agent to begin</div>
  </div>
</div>

<script>
const sessions = {sessions_json};
const results  = {results_json};
const agents   = {agents_json};
const colors   = {colors_json};

let activeAgent = null;
let activeSession = null;

// Build agent buttons
const btnContainer = document.getElementById('agent-btns');
agents.forEach(agent => {{
  const c = colors[agent] || '#8b949e';
  const btn = document.createElement('button');
  btn.className = 'agent-btn';
  btn.textContent = agent;
  btn.style.setProperty('--c', c);
  btn.dataset.agent = agent;
  btn.onclick = () => selectAgent(agent);
  btnContainer.appendChild(btn);
}});

function buildTray(highlightAgent) {{
  const tray = document.getElementById('tray');
  tray.innerHTML = results.map(r => {{
    const pct = Math.max(4, ((r.score - 0.5) / 0.5 * 100)).toFixed(0);
    const agentColor = colors[r.agent] || '#8b949e';
    const dimmed = highlightAgent && r.agent !== highlightAgent ? 'dimmed' : '';
    return `<div class="exp-item ${{r.status}} ${{dimmed}}" onclick="selectExp('${{r.id}}','${{r.agent}}',this)" style="--agent-color:${{agentColor}}">
      <div class="exp-row"><span class="exp-id">${{r.id}}</span><span class="exp-score">${{r.score.toFixed(3)}}</span><span class="exp-agent" style="color:${{agentColor}}">${{r.agent}}</span></div>
      <div class="exp-bar"><div class="exp-bar-fill" style="width:${{pct}}%"></div></div>
      <div class="exp-desc">${{r.desc.slice(0,52)}}</div>
    </div>`;
  }}).join('');
}}

function selectAgent(agent) {{
  activeAgent = agent;
  document.querySelectorAll('.agent-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.agent === agent);
  }});
  buildTray(agent);
  const first = sessions.findIndex(s => s.agent === agent);
  if (first >= 0) showSession(first);
}}

function selectExp(expId, agentId, el) {{
  document.querySelectorAll('.exp-item').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
  const idx = sessions.findIndex(s => s.agent === agentId);
  if (idx >= 0) showSession(idx);
}}

function showSession(i) {{
  document.querySelectorAll('.session').forEach(s => s.classList.remove('active'));
  document.getElementById('placeholder').style.display = 'none';
  const sess = document.getElementById('sess-' + i);
  if (sess) {{ sess.classList.add('active'); activeSession = i; }}
}}

// Init — show all experiments on load
buildTray(null);

// Expand long results
document.addEventListener('click', e => {{
  const b = e.target.closest('.bubble.result.long');
  if (b) b.classList.toggle('expanded');
}});
</script>
</body>
</html>"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir")
    parser.add_argument("--results")
    parser.add_argument("--output", "-o", default="viewer.html")
    args = parser.parse_args()
    logs_dir = Path(args.logs_dir)
    results_tsv = Path(args.results) if args.results else logs_dir.parent / "results.tsv"
    html = build_html(logs_dir, results_tsv)
    out = Path(args.output)
    out.write_text(html)
    print(f"Wrote {out} ({len(html)//1024}KB)")

if __name__ == "__main__":
    main()

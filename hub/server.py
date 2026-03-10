"""
researchRalph Hub — Internal Research API

The blackboard protocol over HTTP. One file, SQLite, no dependencies beyond FastAPI.

Inspired by Karpathy's AgentHub, extended with:
  - Structured memory (fact/failure/hunch)
  - Operator controls (ban, directive, strategy)
  - Typed blackboard (CLAIM/RESPONSE/REQUEST) with threading

Usage:
    uv run server.py
    uv run server.py --port 9000 --host 0.0.0.0
"""

import argparse
import html
import json
import math
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ─── Database ────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "hub.db"


def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT DEFAULT '',
            platform TEXT DEFAULT 'unknown',
            api_key TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            last_seen TEXT
        );

        CREATE TABLE IF NOT EXISTS commits (
            hash TEXT PRIMARY KEY,
            parent TEXT DEFAULT '',
            agent_id TEXT NOT NULL,
            message TEXT NOT NULL,
            score REAL,
            status TEXT NOT NULL DEFAULT 'keep',
            memory_gb REAL DEFAULT 0,
            platform TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL REFERENCES agents(id),
            score REAL,
            status TEXT NOT NULL DEFAULT 'keep',
            description TEXT NOT NULL,
            commit_hash TEXT DEFAULT '',
            memory_gb REAL DEFAULT 0,
            platform TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT 'results',
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS blackboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('CLAIM', 'RESPONSE', 'REQUEST', 'REFUTE', 'OPERATOR')),
            message TEXT NOT NULL,
            target TEXT DEFAULT '',
            in_reply_to INTEGER REFERENCES blackboard(id),
            priority TEXT DEFAULT 'medium',
            evidence TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('fact', 'failure', 'hunch')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_results_agent ON results(agent_id);
        CREATE INDEX IF NOT EXISTS idx_results_score ON results(score);
        CREATE INDEX IF NOT EXISTS idx_commits_agent ON commits(agent_id);
        CREATE INDEX IF NOT EXISTS idx_posts_channel ON posts(channel);
        CREATE INDEX IF NOT EXISTS idx_blackboard_type ON blackboard(type);
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
    """)
    db.close()


# ─── Auth ────────────────────────────────────────────────────


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def time_ago(iso_str):
    """Convert ISO timestamp to human-readable '5m ago' format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        elif secs < 3600:
            return f"{secs // 60}m ago"
        elif secs < 86400:
            return f"{secs // 3600}h ago"
        else:
            return f"{secs // 86400}d ago"
    except Exception:
        return iso_str[:16]


def auth_agent(authorization: str = Header(None), db: sqlite3.Connection = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <api_key>")
    key = authorization.split(" ", 1)[1]
    row = db.execute("SELECT * FROM agents WHERE api_key = ?", (key,)).fetchone()
    if not row:
        raise HTTPException(401, "Invalid API key")
    db.execute("UPDATE agents SET last_seen = ? WHERE id = ?", (now_iso(), row["id"]))
    db.commit()
    return dict(row)


# ─── Models ──────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    name: str
    team: str = ""
    platform: str = "unknown"


class CommitRequest(BaseModel):
    hash: str
    parent: str = ""
    message: str
    score: Optional[float] = None
    status: str = "keep"
    memory_gb: float = 0


class ResultRequest(BaseModel):
    score: Optional[float] = None
    status: str = "keep"
    description: str
    commit_hash: str = ""
    memory_gb: float = 0


class PostRequest(BaseModel):
    channel: str = "results"
    content: str


class BlackboardRequest(BaseModel):
    type: str  # CLAIM, RESPONSE, REQUEST, REFUTE
    message: str
    target: str = ""
    in_reply_to: Optional[int] = None
    priority: str = "medium"
    evidence: dict = {}


class MemoryRequest(BaseModel):
    type: str  # fact, failure, hunch
    content: str


class OperatorRequest(BaseModel):
    message: str = ""
    content: str = ""
    target: str = ""


# ─── App ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="researchRalph Hub",
    description="Internal research API — the blackboard protocol over HTTP",
    version="0.2.0",
    lifespan=lifespan,
)


# ─── Registration ────────────────────────────────────────────


@app.post("/api/register")
def register(req: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    agent_id = f"{req.name}-{secrets.token_hex(4)}"
    api_key = f"rr_{secrets.token_hex(24)}"
    try:
        db.execute(
            "INSERT INTO agents (id, name, team, platform, api_key, created_at) VALUES (?,?,?,?,?,?)",
            (agent_id, req.name, req.team, req.platform, api_key, now_iso()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Agent name collision, try again")
    return {"agent_id": agent_id, "api_key": api_key}


# ─── Commits (git lineage) ──────────────────────────────────


@app.post("/api/commits")
def post_commit(
    req: CommitRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    try:
        db.execute(
            "INSERT INTO commits (hash, parent, agent_id, message, score, status, memory_gb, platform, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (req.hash, req.parent, agent["id"], req.message, req.score, req.status, req.memory_gb, agent["platform"], now_iso()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Commit hash already exists")
    return {"ok": True, "hash": req.hash}


@app.get("/api/commits")
def get_commits(
    limit: int = Query(50, le=500),
    agent: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM commits WHERE 1=1"
    params = []
    if agent:
        query += " AND agent_id = ?"
        params.append(agent)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── Results ─────────────────────────────────────────────────


@app.post("/api/results")
def post_result(
    req: ResultRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    db.execute(
        "INSERT INTO results (agent_id, score, status, description, commit_hash, memory_gb, platform, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (agent["id"], req.score, req.status, req.description, req.commit_hash, req.memory_gb, agent["platform"], now_iso()),
    )
    db.commit()
    return {"ok": True, "agent": agent["id"]}


@app.get("/api/results")
def get_results(
    limit: int = Query(50, le=500),
    agent: Optional[str] = None,
    status: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM results WHERE 1=1"
    params = []
    if agent:
        query += " AND agent_id = ?"
        params.append(agent)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/results/leaderboard")
def leaderboard(top: int = Query(10, le=100), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT agent_id, MIN(score) as best_score, COUNT(*) as experiments, "
        "SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) as keeps, platform "
        "FROM results WHERE score IS NOT NULL AND status='keep' "
        "GROUP BY agent_id ORDER BY best_score ASC LIMIT ?",
        (top,),
    ).fetchall()
    return [
        {
            **dict(r),
            "hit_rate": f"{r['keeps']*100//max(r['experiments'],1)}%",
        }
        for r in rows
    ]


# ─── Posts (channels: #results, #discussion) ────────────────


@app.post("/api/posts")
def create_post(
    req: PostRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    db.execute(
        "INSERT INTO posts (agent_id, channel, content, created_at) VALUES (?,?,?,?)",
        (agent["id"], req.channel, req.content, now_iso()),
    )
    db.commit()
    return {"ok": True}


@app.get("/api/posts")
def get_posts(
    channel: Optional[str] = None,
    limit: int = Query(50, le=500),
    since_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM posts WHERE 1=1"
    params = []
    if channel:
        query += " AND channel = ?"
        params.append(channel)
    if since_id:
        query += " AND id > ?"
        params.append(since_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── Blackboard ──────────────────────────────────────────────


@app.post("/api/blackboard")
def post_blackboard(
    req: BlackboardRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    if req.type not in ("CLAIM", "RESPONSE", "REQUEST", "REFUTE"):
        raise HTTPException(400, "type must be CLAIM, RESPONSE, REQUEST, or REFUTE")
    db.execute(
        "INSERT INTO blackboard (agent_id, type, message, target, in_reply_to, priority, evidence, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (agent["id"], req.type, req.message, req.target, req.in_reply_to, req.priority, json.dumps(req.evidence), now_iso()),
    )
    db.commit()
    return {"ok": True}


@app.get("/api/blackboard")
def get_blackboard(
    limit: int = Query(50, le=500),
    type: Optional[str] = None,
    since_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM blackboard WHERE 1=1"
    params = []
    if type:
        query += " AND type = ?"
        params.append(type)
    if since_id:
        query += " AND id > ?"
        params.append(since_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── Memory ──────────────────────────────────────────────────


@app.post("/api/memory")
def post_memory(
    req: MemoryRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    if req.type not in ("fact", "failure", "hunch"):
        raise HTTPException(400, "type must be fact, failure, or hunch")
    db.execute(
        "INSERT INTO memory (agent_id, type, content, created_at) VALUES (?,?,?,?)",
        (agent["id"], req.type, req.content, now_iso()),
    )
    db.commit()
    return {"ok": True}


@app.get("/api/memory")
def get_memory(
    type: Optional[str] = None,
    limit: int = Query(100, le=1000),
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM memory WHERE 1=1"
    params = []
    if type:
        query += " AND type = ?"
        params.append(type)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── Operator ────────────────────────────────────────────────


@app.post("/api/operator/claim")
def operator_claim(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    msg = req.message or req.content
    db.execute(
        "INSERT INTO blackboard (agent_id, type, message, created_at) VALUES (?,?,?,?)",
        ("OPERATOR", "OPERATOR", msg, now_iso()),
    )
    db.commit()
    return {"ok": True, "message": msg}


@app.post("/api/operator/ban")
def operator_ban(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    content = req.content or req.message
    db.execute(
        "INSERT INTO memory (agent_id, type, content, created_at) VALUES (?,?,?,?)",
        ("OPERATOR", "failure", f"[OPERATOR BAN] {content}", now_iso()),
    )
    db.commit()
    return {"ok": True}


@app.post("/api/operator/directive")
def operator_directive(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    db.execute(
        "INSERT INTO blackboard (agent_id, type, message, target, priority, created_at) VALUES (?,?,?,?,?,?)",
        ("OPERATOR", "OPERATOR", req.message or req.content, req.target, "high", now_iso()),
    )
    db.commit()
    return {"ok": True}


@app.post("/api/operator/strategy")
def operator_strategy(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    content = req.content or req.message
    db.execute(
        "INSERT INTO blackboard (agent_id, type, message, priority, created_at) VALUES (?,?,?,?,?)",
        ("OPERATOR", "OPERATOR", f"[STRATEGY] {content}", "high", now_iso()),
    )
    db.commit()
    return {"ok": True}


# ─── Agents ──────────────────────────────────────────────────


@app.get("/api/agents")
def list_agents(db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT a.id, a.name, a.team, a.platform, a.created_at, a.last_seen, "
        "COUNT(r.id) as experiments, "
        "MIN(r.score) as best_score, "
        "SUM(CASE WHEN r.status='keep' THEN 1 ELSE 0 END) as keeps "
        "FROM agents a LEFT JOIN results r ON a.id = r.agent_id "
        "GROUP BY a.id ORDER BY best_score ASC NULLS LAST"
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Agent not found")
    results = db.execute(
        "SELECT COUNT(*) as total, MIN(score) as best, "
        "SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) as keeps "
        "FROM results WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    return {**dict(row), "stats": dict(results)}


# ─── Dashboard ───────────────────────────────────────────────


def esc(s):
    """HTML-escape a string."""
    return html.escape(str(s)) if s else ""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: sqlite3.Connection = Depends(get_db)):
    total_agents = db.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    total_commits = db.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    total_posts = db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    total_exp = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    best = db.execute("SELECT MIN(score) FROM results WHERE status='keep'").fetchone()[0]
    keeps = db.execute("SELECT COUNT(*) FROM results WHERE status='keep'").fetchone()[0]

    # Commits table
    commits = db.execute("SELECT * FROM commits ORDER BY created_at DESC LIMIT 30").fetchall()
    commits_html = ""
    for c in commits:
        h = esc(c["hash"][:8])
        p = esc(c["parent"][:8]) if c["parent"] else "—"
        agent = esc(c["agent_id"])
        msg = esc(c["message"])
        when = time_ago(c["created_at"])
        score = f"{c['score']:.6f}" if c["score"] else "—"
        status_cls = "keep" if c["status"] == "keep" else "discard" if c["status"] == "discard" else "crash" if c["status"] == "crash" else ""
        commits_html += f'<tr><td class="mono">{h}</td><td class="mono dim">{p}</td><td>{agent}</td><td class="{status_cls}">{score}</td><td>{msg}</td><td class="dim">{when}</td></tr>\n'
    if not commits_html:
        commits_html = '<tr><td colspan="6" class="dim">No commits yet</td></tr>'

    # Board posts by channel
    posts = db.execute("SELECT * FROM posts ORDER BY id DESC LIMIT 40").fetchall()
    posts_html = ""
    for p in posts:
        agent = esc(p["agent_id"])
        channel = esc(p["channel"])
        content = esc(p["content"]).replace("\n", "<br>")
        when = time_ago(p["created_at"])
        posts_html += f'<div class="post"><span class="channel">#{channel}</span> <span class="agent-tag">{agent}</span> <span class="dim">{when}</span><div class="post-content">{content}</div></div>\n'

    # Blackboard (operator + typed messages)
    bb_msgs = db.execute("SELECT * FROM blackboard ORDER BY id DESC LIMIT 20").fetchall()
    bb_html = ""
    for m in bb_msgs:
        cls = m["type"].lower()
        agent = esc(m["agent_id"])
        msg = esc(m["message"]).replace("\n", "<br>")
        when = time_ago(m["created_at"])
        target = f' → {esc(m["target"])}' if m["target"] else ""
        bb_html += f'<div class="bb-msg {cls}"><span class="dim">{agent}{target} · {when}</span><br><strong>{m["type"]}</strong>: {msg}</div>\n'
    if not bb_html:
        bb_html = '<p class="dim">No blackboard messages yet</p>'

    # Memory
    mems = db.execute("SELECT * FROM memory ORDER BY id DESC LIMIT 20").fetchall()
    mem_html = ""
    for m in mems:
        cls = m["type"]
        icon = {"fact": "✓", "failure": "✗", "hunch": "?"}.get(cls, "·")
        content = esc(m["content"])
        agent = esc(m["agent_id"])
        mem_html += f'<div class="mem {cls}"><span class="mem-icon">{icon}</span> <span class="dim">[{agent}]</span> {content}</div>\n'
    if not mem_html:
        mem_html = '<p class="dim">No shared memory yet</p>'

    # Agents table
    agents = db.execute(
        "SELECT a.*, COUNT(r.id) as exp, MIN(r.score) as best "
        "FROM agents a LEFT JOIN results r ON a.id = r.agent_id "
        "GROUP BY a.id ORDER BY a.last_seen DESC NULLS LAST"
    ).fetchall()
    agents_html = ""
    for a in agents:
        best_str = f"{a['best']:.6f}" if a["best"] else "—"
        last = time_ago(a["last_seen"]) if a["last_seen"] else "never"
        agents_html += f'<tr><td>{esc(a["name"])}</td><td>{esc(a["platform"])}</td><td>{a["exp"]}</td><td class="score">{best_str}</td><td class="dim">{last}</td></tr>\n'

    return DASHBOARD_TEMPLATE.format(
        total_agents=total_agents,
        total_commits=total_commits,
        total_posts=total_posts + total_exp,
        best_score=f"{best:.6f}" if best else "—",
        commits_html=commits_html,
        posts_html=posts_html or '<p class="dim">No posts yet</p>',
        bb_html=bb_html,
        mem_html=mem_html,
        agents_html=agents_html or '<tr><td colspan="5" class="dim">No agents</td></tr>',
    )


DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>researchRalph Hub</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 1.5rem; max-width: 1400px; margin: 0 auto; font-size: 13px; }}
  h1 {{ color: #f0f6fc; margin-bottom: 0.25rem; font-size: 1.4rem; }}
  h2 {{ color: #8b949e; margin: 1.5rem 0 0.75rem; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #484f58; margin-bottom: 1.5rem; font-size: 0.85rem; }}
  .dim {{ color: #484f58; }}
  .mono {{ font-family: 'SF Mono', 'Menlo', monospace; }}

  /* Stats */
  .stats {{ display: flex; gap: 0.75rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.75rem 1rem; flex: 1; }}
  .stat-num {{ font-size: 1.5rem; color: #58a6ff; font-weight: bold; }}
  .stat-label {{ color: #484f58; font-size: 0.75rem; text-transform: uppercase; }}

  /* Tables */
  table {{ width: 100%%; border-collapse: collapse; }}
  th {{ text-align: left; color: #8b949e; font-weight: 500; padding: 0.4rem 0.6rem; border-bottom: 1px solid #21262d; font-size: 0.8rem; }}
  td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #161b22; }}
  tr:hover {{ background: #161b22; }}
  .score {{ color: #58a6ff; font-weight: bold; }}
  .keep {{ color: #3fb950; }}
  .discard {{ color: #f85149; }}
  .crash {{ color: #d29922; }}

  /* Layout */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  /* Posts */
  .post {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem; }}
  .post-content {{ margin-top: 0.4rem; line-height: 1.5; white-space: pre-wrap; }}
  .channel {{ color: #58a6ff; font-weight: bold; }}
  .agent-tag {{ color: #d2a8ff; }}

  /* Blackboard */
  .bb-msg {{ padding: 0.6rem 0.75rem; margin-bottom: 0.4rem; border-radius: 4px; border-left: 3px solid #21262d; background: #161b22; }}
  .bb-msg.claim {{ border-left-color: #58a6ff; }}
  .bb-msg.response {{ border-left-color: #3fb950; }}
  .bb-msg.request {{ border-left-color: #d29922; }}
  .bb-msg.refute {{ border-left-color: #f85149; }}
  .bb-msg.operator {{ border-left-color: #bc8cff; background: #1c1230; }}

  /* Memory */
  .mem {{ padding: 0.3rem 0; line-height: 1.4; }}
  .mem.fact {{ color: #3fb950; }}
  .mem.failure {{ color: #f85149; }}
  .mem.hunch {{ color: #d29922; }}
  .mem-icon {{ font-weight: bold; }}

  .footer {{ margin-top: 2rem; color: #484f58; font-size: 0.8rem; }}
  .footer a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<h1>researchRalph Hub</h1>
<p class="subtitle">auto-refreshes every 30s</p>

<div class="stats">
  <div class="stat"><div class="stat-num">{total_agents}</div><div class="stat-label">Agents</div></div>
  <div class="stat"><div class="stat-num">{total_commits}</div><div class="stat-label">Commits</div></div>
  <div class="stat"><div class="stat-num">{total_posts}</div><div class="stat-label">Posts</div></div>
  <div class="stat"><div class="stat-num">{best_score}</div><div class="stat-label">Best Score</div></div>
</div>

<h2>Commits</h2>
<table>
<tr><th>Hash</th><th>Parent</th><th>Agent</th><th>Score</th><th>Message</th><th>When</th></tr>
{commits_html}
</table>

<div class="two-col">
<div>
<h2>Board</h2>
{posts_html}
</div>
<div>
<h2>Blackboard</h2>
{bb_html}

<h2>Shared Memory</h2>
{mem_html}
</div>
</div>

<h2>Agents</h2>
<table>
<tr><th>Name</th><th>Platform</th><th>Experiments</th><th>Best</th><th>Last Seen</th></tr>
{agents_html}
</table>

<p class="footer">Powered by <a href="https://github.com/bigsnarfdude/researchRalph">researchRalph v2</a> — operator controls: POST /api/operator/{{claim,ban,directive,strategy}}</p>
</body>
</html>"""


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="researchRalph Hub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  researchRalph Hub v0.2")
    print(f"  API:       http://{args.host}:{args.port}/api")
    print(f"  Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"  Database:  {DB_PATH}\n")

    uvicorn.run(app, host=args.host, port=args.port)

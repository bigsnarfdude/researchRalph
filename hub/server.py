"""
researchRalph Hub — Internal Research API

The blackboard protocol over HTTP. One file, SQLite, no dependencies beyond FastAPI.

Usage:
    uv run server.py
    uv run server.py --port 9000 --host 0.0.0.0
"""

import argparse
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
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
        CREATE INDEX IF NOT EXISTS idx_blackboard_type ON blackboard(type);
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
    """)
    db.close()


# ─── Auth ────────────────────────────────────────────────────


def now_iso():
    return datetime.now(timezone.utc).isoformat()


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


class ResultRequest(BaseModel):
    score: Optional[float] = None
    status: str = "keep"
    description: str
    commit_hash: str = ""
    memory_gb: float = 0


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
    version="0.1.0",
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
        ("OPERATOR", "failure", f"[OPERATOR] {content}", now_iso()),
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


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>researchRalph Hub</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0a0a0a; color: #e0e0e0; padding: 2rem; max-width: 1200px; margin: 0 auto; }
  h1 { color: #f0f0f0; margin-bottom: 0.5rem; font-size: 1.5rem; }
  h2 { color: #888; margin: 2rem 0 1rem; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; }
  .subtitle { color: #666; margin-bottom: 2rem; }
  table { width: 100%%; border-collapse: collapse; margin-bottom: 1rem; }
  th { text-align: left; color: #888; font-weight: normal; padding: 0.5rem; border-bottom: 1px solid #222; font-size: 0.85rem; }
  td { padding: 0.5rem; border-bottom: 1px solid #1a1a1a; font-size: 0.9rem; }
  tr:hover { background: #111; }
  .score { color: #4fc3f7; font-weight: bold; }
  .keep { color: #66bb6a; }
  .discard { color: #ef5350; }
  .crash { color: #ff7043; }
  .claim { border-left: 3px solid #4fc3f7; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #0d1117; }
  .response { border-left: 3px solid #66bb6a; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #0d1117; }
  .request { border-left: 3px solid #ffa726; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #0d1117; }
  .operator { border-left: 3px solid #ce93d8; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #1a0d1f; }
  .meta { color: #666; font-size: 0.8rem; }
  .fact { color: #66bb6a; }
  .failure { color: #ef5350; }
  .hunch { color: #ffa726; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .stat-box { background: #111; padding: 1rem; border-radius: 4px; }
  .stat-number { font-size: 2rem; color: #4fc3f7; }
  .stat-label { color: #666; font-size: 0.8rem; }
  .stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
</style>
</head>
<body>
<h1>researchRalph Hub</h1>
<p class="subtitle">Auto-refreshes every 30s</p>

<div class="stats-row">
  <div class="stat-box"><div class="stat-number">{total_agents}</div><div class="stat-label">agents</div></div>
  <div class="stat-box"><div class="stat-number">{total_experiments}</div><div class="stat-label">experiments</div></div>
  <div class="stat-box"><div class="stat-number">{best_score}</div><div class="stat-label">best score</div></div>
  <div class="stat-box"><div class="stat-number">{hit_rate}</div><div class="stat-label">overall hit rate</div></div>
</div>

<div class="grid">
<div>
<h2>Leaderboard</h2>
<table>
<tr><th>#</th><th>Agent</th><th>Platform</th><th>Best</th><th>Experiments</th><th>Hit Rate</th></tr>
{leaderboard_rows}
</table>

<h2>Recent Results</h2>
<table>
<tr><th>Score</th><th>Status</th><th>Agent</th><th>Description</th></tr>
{results_rows}
</table>
</div>

<div>
<h2>Blackboard</h2>
{blackboard_items}

<h2>Shared Memory</h2>
{memory_items}
</div>
</div>

<h2>Agents</h2>
<table>
<tr><th>Name</th><th>Team</th><th>Platform</th><th>Experiments</th><th>Best</th><th>Last Seen</th></tr>
{agent_rows}
</table>

<p class="meta" style="margin-top: 2rem;">Powered by <a href="https://github.com/bigsnarfdude/researchRalph" style="color: #4fc3f7;">researchRalph v2</a></p>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: sqlite3.Connection = Depends(get_db)):
    # Stats
    total_agents = db.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    total_exp = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    best = db.execute("SELECT MIN(score) FROM results WHERE status='keep'").fetchone()[0]
    keeps = db.execute("SELECT COUNT(*) FROM results WHERE status='keep'").fetchone()[0]
    hit_rate = f"{keeps * 100 // max(total_exp, 1)}%" if total_exp else "—"

    # Leaderboard
    leaders = db.execute(
        "SELECT agent_id, MIN(score) as best, COUNT(*) as total, "
        "SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) as keeps, platform "
        "FROM results WHERE score IS NOT NULL GROUP BY agent_id ORDER BY best ASC LIMIT 10"
    ).fetchall()
    lb_rows = ""
    for i, r in enumerate(leaders, 1):
        hr = f"{r['keeps'] * 100 // max(r['total'], 1)}%"
        lb_rows += f"<tr><td>{i}</td><td>{r['agent_id']}</td><td>{r['platform']}</td><td class='score'>{r['best']:.4f}</td><td>{r['total']}</td><td>{hr}</td></tr>\n"
    if not lb_rows:
        lb_rows = "<tr><td colspan='6' style='color:#666'>No results yet</td></tr>"

    # Recent results
    recent = db.execute("SELECT * FROM results ORDER BY id DESC LIMIT 15").fetchall()
    res_rows = ""
    for r in recent:
        cls = r["status"]
        score_str = f"{r['score']:.4f}" if r["score"] else "—"
        res_rows += f"<tr><td class='score'>{score_str}</td><td class='{cls}'>{r['status']}</td><td>{r['agent_id']}</td><td>{r['description'][:60]}</td></tr>\n"
    if not res_rows:
        res_rows = "<tr><td colspan='4' style='color:#666'>No results yet</td></tr>"

    # Blackboard
    msgs = db.execute("SELECT * FROM blackboard ORDER BY id DESC LIMIT 15").fetchall()
    bb_items = ""
    for m in msgs:
        cls = m["type"].lower()
        prefix = f"<span class='meta'>{m['agent_id']} · {m['created_at'][:16]}</span><br>"
        bb_items += f"<div class='{cls}'>{prefix}<strong>{m['type']}</strong>: {m['message']}</div>\n"
    if not bb_items:
        bb_items = "<p style='color:#666'>No blackboard messages yet</p>"

    # Memory
    mems = db.execute("SELECT * FROM memory ORDER BY id DESC LIMIT 15").fetchall()
    mem_items = ""
    for m in mems:
        cls = m["type"]
        mem_items += f"<p class='{cls}'>{'✓' if cls == 'fact' else '✗' if cls == 'failure' else '?'} [{m['type']}] {m['content']}</p>\n"
    if not mem_items:
        mem_items = "<p style='color:#666'>No shared memory yet</p>"

    # Agents
    agents = db.execute(
        "SELECT a.*, COUNT(r.id) as exp, MIN(r.score) as best "
        "FROM agents a LEFT JOIN results r ON a.id = r.agent_id "
        "GROUP BY a.id ORDER BY a.last_seen DESC NULLS LAST"
    ).fetchall()
    a_rows = ""
    for a in agents:
        best_str = f"{a['best']:.4f}" if a["best"] else "—"
        last = a["last_seen"][:16] if a["last_seen"] else "never"
        a_rows += f"<tr><td>{a['name']}</td><td>{a['team']}</td><td>{a['platform']}</td><td>{a['exp']}</td><td class='score'>{best_str}</td><td class='meta'>{last}</td></tr>\n"
    if not a_rows:
        a_rows = "<tr><td colspan='6' style='color:#666'>No agents registered</td></tr>"

    return DASHBOARD_HTML.format(
        total_agents=total_agents,
        total_experiments=total_exp,
        best_score=f"{best:.4f}" if best else "—",
        hit_rate=hit_rate,
        leaderboard_rows=lb_rows,
        results_rows=res_rows,
        blackboard_items=bb_items,
        memory_items=mem_items,
        agent_rows=a_rows,
    )


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="researchRalph Hub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  researchRalph Hub")
    print(f"  API:       http://{args.host}:{args.port}/api")
    print(f"  Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"  Database:  {DB_PATH}\n")

    uvicorn.run(app, host=args.host, port=args.port)

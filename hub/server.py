"""
researchRalph Hub v0.3 — Unified Event Stream

Everything is an event. Old endpoints are views. New endpoints are the stream.

Architecture:
    events table     ← single source of truth
    /api/stream      ← SSE real-time
    /api/events      ← raw CRUD
    /api/results     ← backward-compat view (type=RESULT)
    /api/blackboard  ← backward-compat view (type=CLAIM/RESPONSE/REQUEST/REFUTE/OPERATOR)
    /api/memory      ← backward-compat view (type=FACT/FAILURE/HUNCH)
    /api/commits     ← backward-compat view (type=COMMIT)
    /api/posts       ← backward-compat view (type=POST)
    playbooks        ← reactive rules on the stream

Usage:
    uv run server.py
    uv run server.py --port 9000 --host 0.0.0.0
"""

import argparse
import asyncio
import html
import json
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# ─── Database ────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "hub.db"

# All event types in the system
EVENT_TYPES = {
    # Core experiment events
    "RESULT", "COMMIT",
    # Communication
    "CLAIM", "RESPONSE", "REQUEST", "REFUTE", "POST",
    # Memory
    "FACT", "FAILURE", "HUNCH",
    # Control
    "OPERATOR",
    # Reactions (lightweight signals)
    "CONFIRM", "CONTRADICT", "ADOPT",
    # Aletheia-inspired: Generator → Verifier → Reviser
    "VERIFY",
    # System
    "HEARTBEAT", "PLAYBOOK",
}

# Groupings for backward-compat endpoints
BLACKBOARD_TYPES = {"CLAIM", "RESPONSE", "REQUEST", "REFUTE", "OPERATOR"}
MEMORY_TYPES = {"FACT", "FAILURE", "HUNCH"}
REACTION_TYPES = {"CONFIRM", "CONTRADICT", "ADOPT"}


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

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            tags TEXT NOT NULL DEFAULT '[]',
            reply_to INTEGER REFERENCES events(id),
            platform TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
        CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
        CREATE INDEX IF NOT EXISTS idx_events_platform ON events(platform);
        CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_events_reply ON events(reply_to);
    """)
    db.close()


# ─── Helpers ─────────────────────────────────────────────────


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def time_ago(iso_str):
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


def esc(s):
    return html.escape(str(s)) if s else ""


def insert_event(db, etype, agent_id, payload, tags=None, reply_to=None, platform=""):
    """Insert an event and run playbooks. Returns the event dict."""
    tags_json = json.dumps(tags or [])
    payload_json = json.dumps(payload) if isinstance(payload, dict) else payload
    ts = now_iso()
    cursor = db.execute(
        "INSERT INTO events (type, agent_id, payload, tags, reply_to, platform, created_at) VALUES (?,?,?,?,?,?,?)",
        (etype, agent_id, payload_json, tags_json, reply_to, platform, ts),
    )
    db.commit()
    event = {
        "id": cursor.lastrowid,
        "type": etype,
        "agent_id": agent_id,
        "payload": payload if isinstance(payload, dict) else json.loads(payload_json),
        "tags": tags or [],
        "reply_to": reply_to,
        "platform": platform,
        "created_at": ts,
    }
    # Run playbooks (non-recursive)
    if agent_id != "PLAYBOOK":
        run_playbooks(event, db)
    return event


def format_event(row):
    """Convert a DB row to a clean event dict."""
    d = dict(row)
    try:
        d["payload"] = json.loads(d["payload"]) if isinstance(d["payload"], str) else d["payload"]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
    except (json.JSONDecodeError, TypeError):
        d["tags"] = []
    return d


# ─── Auth ────────────────────────────────────────────────────


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


class CommitRequest(BaseModel):
    hash: str
    parent: str = ""
    message: str
    score: Optional[float] = None
    status: str = "keep"
    memory_gb: float = 0


class PostRequest(BaseModel):
    channel: str = "results"
    content: str


class BlackboardRequest(BaseModel):
    type: str
    message: str
    target: str = ""
    in_reply_to: Optional[int] = None
    priority: str = "medium"
    evidence: dict = {}


class MemoryRequest(BaseModel):
    type: str
    content: str


class OperatorRequest(BaseModel):
    message: str = ""
    content: str = ""
    target: str = ""


class EventRequest(BaseModel):
    type: str
    payload: dict = {}
    tags: list = []
    reply_to: Optional[int] = None


class ReactionRequest(BaseModel):
    reason: str = ""


# ─── App ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="researchRalph Hub",
    description="Unified event stream — the blackboard protocol over HTTP",
    version="0.3.0",
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


# ═══════════════════════════════════════════════════════════════
# NEW: Unified Event API
# ═══════════════════════════════════════════════════════════════


@app.post("/api/events")
def post_event(
    req: EventRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    if req.type not in EVENT_TYPES:
        raise HTTPException(400, f"Unknown event type: {req.type}. Valid: {sorted(EVENT_TYPES)}")
    event = insert_event(db, req.type, agent["id"], req.payload, req.tags, req.reply_to, agent["platform"])
    return {"ok": True, "event": event}


@app.get("/api/events")
def get_events(
    types: Optional[str] = None,
    agent: Optional[str] = None,
    platform: Optional[str] = None,
    tags: Optional[str] = None,
    since_id: int = 0,
    limit: int = Query(50, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM events WHERE id > ?"
    params: list = [since_id]
    if types:
        type_list = [t.strip().upper() for t in types.split(",")]
        placeholders = ",".join("?" * len(type_list))
        query += f" AND type IN ({placeholders})"
        params.extend(type_list)
    if agent:
        query += " AND agent_id = ?"
        params.append(agent)
    if platform:
        query += " AND platform = ?"
        params.append(platform)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    events = [format_event(r) for r in rows]
    # Filter by tags in Python (JSON array in SQLite)
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        events = [e for e in events if any(t in e.get("tags", []) for t in tag_list)]
    return events


# ─── Reactions ───────────────────────────────────────────────


@app.post("/api/events/{event_id}/confirm")
def confirm_event(
    event_id: int,
    req: ReactionRequest = ReactionRequest(),
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    target = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not target:
        raise HTTPException(404, "Event not found")
    event = insert_event(db, "CONFIRM", agent["id"], {"reason": req.reason}, reply_to=event_id, platform=agent["platform"])
    return {"ok": True, "event": event}


@app.post("/api/events/{event_id}/contradict")
def contradict_event(
    event_id: int,
    req: ReactionRequest = ReactionRequest(),
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    target = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not target:
        raise HTTPException(404, "Event not found")
    event = insert_event(db, "CONTRADICT", agent["id"], {"reason": req.reason}, reply_to=event_id, platform=agent["platform"])
    return {"ok": True, "event": event}


@app.post("/api/events/{event_id}/adopt")
def adopt_event(
    event_id: int,
    req: ReactionRequest = ReactionRequest(),
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    target = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not target:
        raise HTTPException(404, "Event not found")
    event = insert_event(db, "ADOPT", agent["id"], {"reason": req.reason}, reply_to=event_id, platform=agent["platform"])
    return {"ok": True, "event": event}


# ─── SSE Stream ──────────────────────────────────────────────


@app.get("/api/stream")
async def stream_events(
    types: Optional[str] = None,
    agent: Optional[str] = None,
    platform: Optional[str] = None,
    tags: Optional[str] = None,
    since_id: int = 0,
):
    type_list = [t.strip().upper() for t in types.split(",")] if types else None
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    async def generate():
        last_id = since_id
        while True:
            db = sqlite3.connect(str(DB_PATH))
            db.row_factory = sqlite3.Row
            query = "SELECT * FROM events WHERE id > ?"
            params: list = [last_id]
            if type_list:
                placeholders = ",".join("?" * len(type_list))
                query += f" AND type IN ({placeholders})"
                params.extend(type_list)
            if agent:
                query += " AND agent_id = ?"
                params.append(agent)
            if platform:
                query += " AND platform = ?"
                params.append(platform)
            query += " ORDER BY id ASC LIMIT 50"
            rows = db.execute(query, params).fetchall()
            for row in rows:
                event = format_event(row)
                # Tag filter
                if tag_list and not any(t in event.get("tags", []) for t in tag_list):
                    last_id = row["id"]
                    continue
                yield f"id: {row['id']}\nevent: {row['type'].lower()}\ndata: {json.dumps(event)}\n\n"
                last_id = row["id"]
            db.close()
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════
# BACKWARD-COMPAT: Old endpoints as views on the event stream
# ═══════════════════════════════════════════════════════════════


# ─── Results ─────────────────────────────────────────────────


@app.post("/api/results")
def post_result(
    req: ResultRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    payload = {
        "score": req.score,
        "status": req.status,
        "description": req.description,
        "commit_hash": req.commit_hash,
        "memory_gb": req.memory_gb,
    }
    event = insert_event(db, "RESULT", agent["id"], payload, platform=agent["platform"])
    return {"ok": True, "agent": agent["id"], "event_id": event["id"]}


@app.get("/api/results")
def get_results(
    limit: int = Query(50, le=500),
    agent: Optional[str] = None,
    status: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM events WHERE type = 'RESULT'"
    params: list = []
    if agent:
        query += " AND agent_id = ?"
        params.append(agent)
    if status:
        query += " AND json_extract(payload, '$.status') = ?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    # Return in old format for backward compat
    results = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        results.append({
            "id": e["id"],
            "agent_id": e["agent_id"],
            "score": p.get("score"),
            "status": p.get("status", "keep"),
            "description": p.get("description", ""),
            "commit_hash": p.get("commit_hash", ""),
            "memory_gb": p.get("memory_gb", 0),
            "platform": e["platform"],
            "created_at": e["created_at"],
        })
    return results


@app.get("/api/results/leaderboard")
def leaderboard(
    top: int = Query(10, le=100),
    platform: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = """
        SELECT agent_id, platform,
            MIN(CASE WHEN json_extract(payload, '$.status') = 'keep'
                THEN CAST(json_extract(payload, '$.score') AS REAL) END) as best_score,
            COUNT(*) as experiments,
            SUM(CASE WHEN json_extract(payload, '$.status') = 'keep' THEN 1 ELSE 0 END) as keeps
        FROM events
        WHERE type = 'RESULT' AND json_extract(payload, '$.score') IS NOT NULL
    """
    params: list = []
    if platform:
        query += " AND platform = ?"
        params.append(platform)
    query += " GROUP BY agent_id ORDER BY best_score ASC LIMIT ?"
    params.append(top)
    rows = db.execute(query, params).fetchall()
    return [
        {
            **dict(r),
            "hit_rate": f"{r['keeps'] * 100 // max(r['experiments'], 1)}%",
        }
        for r in rows
    ]


# ─── Commits ─────────────────────────────────────────────────


@app.post("/api/commits")
def post_commit(
    req: CommitRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    payload = {
        "hash": req.hash,
        "parent": req.parent,
        "message": req.message,
        "score": req.score,
        "status": req.status,
        "memory_gb": req.memory_gb,
    }
    event = insert_event(db, "COMMIT", agent["id"], payload, platform=agent["platform"])
    return {"ok": True, "hash": req.hash, "event_id": event["id"]}


@app.get("/api/commits")
def get_commits(
    limit: int = Query(50, le=500),
    agent: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM events WHERE type = 'COMMIT'"
    params: list = []
    if agent:
        query += " AND agent_id = ?"
        params.append(agent)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    results = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        results.append({
            "hash": p.get("hash", ""),
            "parent": p.get("parent", ""),
            "agent_id": e["agent_id"],
            "message": p.get("message", ""),
            "score": p.get("score"),
            "status": p.get("status", "keep"),
            "memory_gb": p.get("memory_gb", 0),
            "platform": e["platform"],
            "created_at": e["created_at"],
        })
    return results


# ─── Posts ───────────────────────────────────────────────────


@app.post("/api/posts")
def create_post(
    req: PostRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    payload = {"channel": req.channel, "content": req.content}
    event = insert_event(db, "POST", agent["id"], payload, platform=agent["platform"])
    return {"ok": True, "event_id": event["id"]}


@app.get("/api/posts")
def get_posts(
    channel: Optional[str] = None,
    limit: int = Query(50, le=500),
    since_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    query = "SELECT * FROM events WHERE type = 'POST'"
    params: list = []
    if channel:
        query += " AND json_extract(payload, '$.channel') = ?"
        params.append(channel)
    if since_id:
        query += " AND id > ?"
        params.append(since_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    results = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        results.append({
            "id": e["id"],
            "agent_id": e["agent_id"],
            "channel": p.get("channel", "results"),
            "content": p.get("content", ""),
            "created_at": e["created_at"],
        })
    return results


# ─── Blackboard ──────────────────────────────────────────────


@app.post("/api/blackboard")
def post_blackboard(
    req: BlackboardRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    etype = req.type.upper()
    if etype not in ("CLAIM", "RESPONSE", "REQUEST", "REFUTE"):
        raise HTTPException(400, "type must be CLAIM, RESPONSE, REQUEST, or REFUTE")
    payload = {
        "message": req.message,
        "target": req.target,
        "priority": req.priority,
        "evidence": req.evidence,
    }
    event = insert_event(db, etype, agent["id"], payload, reply_to=req.in_reply_to, platform=agent["platform"])
    return {"ok": True, "event_id": event["id"]}


@app.get("/api/blackboard")
def get_blackboard(
    limit: int = Query(50, le=500),
    type: Optional[str] = None,
    since_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    if type:
        type_list = [type.upper()]
    else:
        type_list = list(BLACKBOARD_TYPES)
    placeholders = ",".join("?" * len(type_list))
    query = f"SELECT * FROM events WHERE type IN ({placeholders})"
    params: list = list(type_list)
    if since_id:
        query += " AND id > ?"
        params.append(since_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    results = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        results.append({
            "id": e["id"],
            "agent_id": e["agent_id"],
            "type": e["type"],
            "message": p.get("message", p.get("content", "")),
            "target": p.get("target", ""),
            "in_reply_to": e["reply_to"],
            "priority": p.get("priority", "medium"),
            "evidence": json.dumps(p.get("evidence", {})),
            "created_at": e["created_at"],
        })
    return results


# ─── Memory ──────────────────────────────────────────────────


@app.post("/api/memory")
def post_memory(
    req: MemoryRequest,
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
):
    etype = req.type.upper()
    if etype not in MEMORY_TYPES:
        raise HTTPException(400, "type must be fact, failure, or hunch")
    payload = {"content": req.content}
    event = insert_event(db, etype, agent["id"], payload, platform=agent["platform"])
    return {"ok": True, "event_id": event["id"]}


@app.get("/api/memory")
def get_memory(
    type: Optional[str] = None,
    limit: int = Query(100, le=1000),
    db: sqlite3.Connection = Depends(get_db),
):
    if type:
        type_list = [type.upper()]
    else:
        type_list = list(MEMORY_TYPES)
    placeholders = ",".join("?" * len(type_list))
    query = f"SELECT * FROM events WHERE type IN ({placeholders})"
    params: list = list(type_list)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    results = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        results.append({
            "id": e["id"],
            "agent_id": e["agent_id"],
            "type": e["type"].lower(),
            "content": p.get("content", ""),
            "created_at": e["created_at"],
        })
    return results


# ─── Operator ────────────────────────────────────────────────


@app.post("/api/operator/claim")
def operator_claim(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    msg = req.message or req.content
    insert_event(db, "OPERATOR", "OPERATOR", {"message": msg, "subtype": "claim"}, tags=["operator"])
    return {"ok": True, "message": msg}


@app.post("/api/operator/ban")
def operator_ban(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    content = req.content or req.message
    insert_event(db, "FAILURE", "OPERATOR", {"content": f"[OPERATOR BAN] {content}"}, tags=["operator", "ban"])
    return {"ok": True}


@app.post("/api/operator/directive")
def operator_directive(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    msg = req.message or req.content
    insert_event(db, "OPERATOR", "OPERATOR", {"message": msg, "target": req.target, "subtype": "directive", "priority": "high"}, tags=["operator", "directive"])
    return {"ok": True}


@app.post("/api/operator/strategy")
def operator_strategy(req: OperatorRequest, db: sqlite3.Connection = Depends(get_db)):
    content = req.content or req.message
    insert_event(db, "OPERATOR", "OPERATOR", {"message": f"[STRATEGY] {content}", "subtype": "strategy", "priority": "high"}, tags=["operator", "strategy"])
    return {"ok": True}


# ─── Agents ──────────────────────────────────────────────────


@app.get("/api/agents")
def list_agents(db: sqlite3.Connection = Depends(get_db)):
    agents = db.execute("SELECT * FROM agents ORDER BY last_seen DESC NULLS LAST").fetchall()
    result = []
    for a in agents:
        stats = db.execute(
            """SELECT COUNT(*) as experiments,
                MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best_score,
                SUM(CASE WHEN json_extract(payload, '$.status') = 'keep' THEN 1 ELSE 0 END) as keeps
            FROM events WHERE type = 'RESULT' AND agent_id = ?""",
            (a["id"],),
        ).fetchone()
        result.append({
            **dict(a),
            "experiments": stats["experiments"] if stats else 0,
            "best_score": stats["best_score"] if stats else None,
            "keeps": stats["keeps"] if stats else 0,
        })
    return result


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Agent not found")
    stats = db.execute(
        """SELECT COUNT(*) as total,
            MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best,
            SUM(CASE WHEN json_extract(payload, '$.status') = 'keep' THEN 1 ELSE 0 END) as keeps
        FROM events WHERE type = 'RESULT' AND agent_id = ?""",
        (agent_id,),
    ).fetchone()
    return {**dict(row), "stats": dict(stats)}


# ═══════════════════════════════════════════════════════════════
# PLAYBOOKS — Reactive rules on the event stream
# ═══════════════════════════════════════════════════════════════

PLAYBOOKS = []


def playbook(name, event_types=None):
    """Register a playbook function. Receives (event, db), returns list of dicts to insert."""
    def decorator(fn):
        fn._playbook_name = name
        fn._event_types = set(event_types) if event_types else None
        PLAYBOOKS.append(fn)
        return fn
    return decorator


def run_playbooks(event, db):
    """Run all playbooks against a new event."""
    for pb in PLAYBOOKS:
        if pb._event_types and event["type"] not in pb._event_types:
            continue
        try:
            new_events = pb(event, db)
            for ne in (new_events or []):
                insert_event(
                    db,
                    ne["type"],
                    "PLAYBOOK",
                    ne.get("payload", {}),
                    tags=ne.get("tags", ["auto", pb._playbook_name]),
                    reply_to=ne.get("reply_to"),
                    platform="",
                )
        except Exception as exc:
            print(f"[playbook:{pb._playbook_name}] error: {exc}")


# ─── Built-in Playbook: Dead End Detector ────────────────────


@playbook("dead-end-detector", event_types=["RESULT"])
def dead_end_detector(event, db):
    """When 2+ agents independently discard similar configs, auto-promote to FAILURE."""
    p = event["payload"]
    if p.get("status") != "discard":
        return []
    desc = p.get("description", "").strip()
    if len(desc) < 5:
        return []

    # Find other agents who also discarded something with similar description
    # Use first 30 chars as a rough match key
    match_key = desc[:30]
    similar = db.execute(
        """SELECT DISTINCT agent_id FROM events
        WHERE type = 'RESULT' AND id != ?
        AND agent_id != ?
        AND json_extract(payload, '$.status') = 'discard'
        AND SUBSTR(json_extract(payload, '$.description'), 1, 30) = ?""",
        (event["id"], event["agent_id"], match_key),
    ).fetchall()

    if len(similar) >= 1:  # this agent + 1 other = 2 total
        agents_str = ", ".join([event["agent_id"]] + [r["agent_id"] for r in similar])
        return [{
            "type": "FAILURE",
            "payload": {"content": f"[AUTO] Dead end detected: '{desc}' — failed on {len(similar) + 1} agents ({agents_str})"},
            "tags": ["auto", "dead-end-detector"],
        }]
    return []


# ─── Built-in Playbook: Convergence Signal ───────────────────


@playbook("convergence-signal", event_types=["RESULT"])
def convergence_signal(event, db):
    """Alert when 3+ agents' best results are within 1% of each other."""
    p = event["payload"]
    if p.get("status") != "keep" or p.get("score") is None:
        return []

    # Get best score per agent (keeps only)
    bests = db.execute(
        """SELECT agent_id, MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best
        FROM events WHERE type = 'RESULT'
        AND json_extract(payload, '$.status') = 'keep'
        AND json_extract(payload, '$.score') IS NOT NULL
        GROUP BY agent_id"""
    ).fetchall()

    if len(bests) < 3:
        return []

    scores = sorted([r["best"] for r in bests])
    # Check if top 3 are within 1% of each other
    top3 = scores[:3]
    if top3[0] > 0 and (top3[2] - top3[0]) / top3[0] < 0.01:
        # Don't spam — check if we already fired this
        existing = db.execute(
            """SELECT id FROM events WHERE agent_id = 'PLAYBOOK'
            AND json_extract(payload, '$.subtype') = 'convergence'
            AND created_at > datetime('now', '-1 hour')"""
        ).fetchone()
        if existing:
            return []

        return [{
            "type": "OPERATOR",
            "payload": {
                "message": f"[CONVERGENCE] Top {len(bests)} agents within 1%: {', '.join(f'{s:.4f}' for s in top3)}. Consider switching to exploit phase.",
                "subtype": "convergence",
            },
            "tags": ["auto", "convergence-signal"],
        }]
    return []


# ─── Built-in Playbook: Platform Mismatch Warning ────────────


@playbook("platform-mismatch", event_types=["RESULT"])
def platform_mismatch_warning(event, db):
    """Warn when results from different platforms might be compared incorrectly."""
    # Check if there are now results from 2+ platforms
    platforms = db.execute(
        """SELECT DISTINCT platform FROM events
        WHERE type = 'RESULT' AND platform != ''"""
    ).fetchall()

    if len(platforms) < 2:
        return []

    # Don't spam — only fire once per hour
    existing = db.execute(
        """SELECT id FROM events
        WHERE agent_id = 'PLAYBOOK'
        AND json_extract(payload, '$.subtype') = 'platform-mismatch'
        AND created_at > datetime('now', '-1 hour')"""
    ).fetchone()
    if existing:
        return []

    # Get best score per platform
    platform_bests = db.execute(
        """SELECT platform,
            MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best,
            COUNT(*) as count
        FROM events WHERE type = 'RESULT'
        AND json_extract(payload, '$.status') = 'keep'
        AND json_extract(payload, '$.score') IS NOT NULL
        AND platform != ''
        GROUP BY platform"""
    ).fetchall()

    if len(platform_bests) < 2:
        return []

    lines = [f"{r['platform']}: best={r['best']:.4f} ({r['count']} experiments)" for r in platform_bests]
    return [{
        "type": "OPERATOR",
        "payload": {
            "message": f"[PLATFORM WARNING] Results from {len(platform_bests)} platforms detected. Scores may not be comparable across different hardware.\n" + "\n".join(lines),
            "subtype": "platform-mismatch",
        },
        "tags": ["auto", "platform-mismatch"],
    }]


# ─── Aletheia-inspired Playbook: Verification Request ────────


@playbook("verification-request", event_types=["RESULT"])
def verification_request(event, db):
    """When a new best score arrives, auto-request another agent to verify it.

    Inspired by Aletheia's Generator → Verifier → Reviser loop:
    decoupling generation from verification catches errors the generator misses.
    """
    p = event["payload"]
    if p.get("status") != "keep" or p.get("score") is None:
        return []

    score = p["score"]
    agent_id = event["agent_id"]
    platform = event.get("platform", "")

    # Get current best for this platform (or global if no platform)
    if platform:
        prev_best = db.execute(
            """SELECT MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best
            FROM events WHERE type = 'RESULT'
            AND json_extract(payload, '$.status') = 'keep'
            AND json_extract(payload, '$.score') IS NOT NULL
            AND platform = ? AND id < ?""",
            (platform, event["id"]),
        ).fetchone()
    else:
        prev_best = db.execute(
            """SELECT MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best
            FROM events WHERE type = 'RESULT'
            AND json_extract(payload, '$.status') = 'keep'
            AND json_extract(payload, '$.score') IS NOT NULL
            AND id < ?""",
            (event["id"],),
        ).fetchone()

    if not prev_best or prev_best["best"] is None:
        return []  # First result, nothing to beat

    # Only request verification for results that beat the previous best
    if score >= prev_best["best"]:
        return []

    # Don't spam — check if we already requested verification for this result
    existing = db.execute(
        """SELECT id FROM events WHERE type = 'VERIFY'
        AND json_extract(payload, '$.subtype') = 'request'
        AND reply_to = ?""",
        (event["id"],),
    ).fetchone()
    if existing:
        return []

    desc = p.get("description", "unknown config")
    improvement = ((prev_best["best"] - score) / prev_best["best"]) * 100
    return [{
        "type": "VERIFY",
        "payload": {
            "subtype": "request",
            "message": f"[VERIFY] New best {score:.6f} by {agent_id} ({improvement:.1f}% improvement). Config: {desc}. Another agent should reproduce this.",
            "original_result_id": event["id"],
            "original_agent": agent_id,
            "original_score": score,
            "original_description": desc,
            "platform": platform,
        },
        "tags": ["auto", "verification-request"],
        "reply_to": event["id"],
    }]


# ─── Aletheia-inspired Playbook: Revision Prompt ─────────────


@playbook("revision-prompt", event_types=["RESULT"])
def revision_prompt(event, db):
    """When an experiment fails, suggest a revision instead of starting from scratch.

    Inspired by Aletheia's Reviser subagent: take verifier feedback and improve
    the solution, rather than generating a completely new attempt.
    """
    p = event["payload"]
    if p.get("status") not in ("discard", "crash"):
        return []

    desc = p.get("description", "").strip()
    if len(desc) < 5:
        return []

    agent_id = event["agent_id"]
    score = p.get("score")
    score_str = f"{score:.6f}" if score else "crashed"

    # Check what this agent's best score is (for context)
    agent_best = db.execute(
        """SELECT MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best
        FROM events WHERE type = 'RESULT'
        AND json_extract(payload, '$.status') = 'keep'
        AND json_extract(payload, '$.score') IS NOT NULL
        AND agent_id = ?""",
        (agent_id,),
    ).fetchone()

    best_str = f"{agent_best['best']:.6f}" if agent_best and agent_best["best"] else "none"

    # Don't spam — max 1 revision prompt per agent per 10 min
    existing = db.execute(
        """SELECT id FROM events WHERE agent_id = 'PLAYBOOK'
        AND json_extract(payload, '$.subtype') = 'revision'
        AND json_extract(payload, '$.target_agent') = ?
        AND created_at > datetime('now', '-10 minutes')""",
        (agent_id,),
    ).fetchone()
    if existing:
        return []

    return [{
        "type": "HUNCH",
        "payload": {
            "subtype": "revision",
            "content": f"[REVISE] {agent_id}'s experiment failed ({score_str}): '{desc}'. Best so far: {best_str}. Instead of discarding entirely, consider: what if you revised this approach with a smaller change? (Aletheia pattern: Reviser takes failure feedback and adjusts, rather than starting over.)",
            "target_agent": agent_id,
            "failed_description": desc,
            "failed_score": score,
        },
        "tags": ["auto", "revision-prompt"],
        "reply_to": event["id"],
    }]


# ═══════════════════════════════════════════════════════════════
# VERIFICATION API (Aletheia-inspired)
# ═══════════════════════════════════════════════════════════════


@app.get("/api/verify/queue")
def get_verify_queue(
    platform: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    """Get pending verification requests (results that need independent reproduction)."""
    query = """SELECT * FROM events WHERE type = 'VERIFY'
        AND json_extract(payload, '$.subtype') = 'request'
        ORDER BY id DESC LIMIT 20"""
    rows = db.execute(query).fetchall()
    queue = []
    for r in rows:
        e = format_event(r)
        p = e["payload"]
        # Check if already verified
        verified = db.execute(
            """SELECT id, agent_id, json_extract(payload, '$.reproduced_score') as score,
                json_extract(payload, '$.verdict') as verdict
            FROM events WHERE type = 'VERIFY'
            AND json_extract(payload, '$.subtype') = 'result'
            AND reply_to = ?""",
            (e["id"],),
        ).fetchall()
        verifications = [{"agent": v["agent_id"], "score": v["score"], "verdict": v["verdict"]} for v in verified]
        if platform and p.get("platform") and p["platform"] != platform:
            continue  # Skip cross-platform verifications
        queue.append({
            "id": e["id"],
            "original_result_id": p.get("original_result_id"),
            "original_agent": p.get("original_agent"),
            "original_score": p.get("original_score"),
            "description": p.get("original_description", ""),
            "platform": p.get("platform", ""),
            "verifications": verifications,
            "verified": len(verifications) > 0,
            "created_at": e["created_at"],
        })
    return queue


@app.post("/api/verify")
def post_verification(
    agent: dict = Depends(auth_agent),
    db: sqlite3.Connection = Depends(get_db),
    verify_request_id: int = Query(..., description="ID of the VERIFY request event"),
    reproduced_score: float = Query(..., description="Score you got when reproducing"),
    verdict: str = Query("confirmed", description="confirmed or contradicted"),
    notes: str = Query("", description="Additional notes"),
):
    """Post a verification result (confirming or contradicting a claimed result)."""
    # Find the verification request
    vr = db.execute("SELECT * FROM events WHERE id = ? AND type = 'VERIFY'", (verify_request_id,)).fetchone()
    if not vr:
        raise HTTPException(404, "Verification request not found")
    vr_data = format_event(vr)
    orig_score = vr_data["payload"].get("original_score", 0)

    event = insert_event(db, "VERIFY", agent["id"], {
        "subtype": "result",
        "reproduced_score": reproduced_score,
        "original_score": orig_score,
        "verdict": verdict,
        "notes": notes,
        "original_result_id": vr_data["payload"].get("original_result_id"),
    }, tags=["verification-result"], reply_to=verify_request_id, platform=agent["platform"])

    # Also post a CONFIRM or CONTRADICT on the original result
    orig_result_id = vr_data["payload"].get("original_result_id")
    if orig_result_id:
        if verdict == "confirmed":
            insert_event(db, "CONFIRM", agent["id"], {
                "reason": f"Independently reproduced: got {reproduced_score:.6f} vs claimed {orig_score:.6f}. {notes}",
            }, reply_to=orig_result_id, platform=agent["platform"])
        else:
            insert_event(db, "CONTRADICT", agent["id"], {
                "reason": f"Could not reproduce: got {reproduced_score:.6f} vs claimed {orig_score:.6f}. {notes}",
            }, reply_to=orig_result_id, platform=agent["platform"])

    return {"ok": True, "event": event}


# ═══════════════════════════════════════════════════════════════
# HAI CARDS — Human-AI Interaction Cards (Aletheia-inspired)
# ═══════════════════════════════════════════════════════════════


@app.get("/api/hai-card")
def hai_card(
    agent_id: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    """Generate a Human-AI Interaction Card showing the contribution breakdown.

    Inspired by Aletheia's documentation framework (Section 6.2):
    transparently document what the human vs AI contributed.
    """
    # Get all operator events (human contributions)
    operator_events = db.execute(
        "SELECT * FROM events WHERE type = 'OPERATOR' ORDER BY id ASC"
    ).fetchall()

    # Get all agent events
    agent_query = "SELECT * FROM events WHERE agent_id != 'OPERATOR' AND agent_id != 'PLAYBOOK'"
    agent_params = []
    if agent_id:
        agent_query += " AND agent_id = ?"
        agent_params.append(agent_id)
    agent_query += " ORDER BY id ASC"
    agent_events = db.execute(agent_query, agent_params).fetchall()

    # Count by category
    human_directives = [format_event(e) for e in operator_events]
    agent_results = []
    agent_claims = []
    agent_failures = []
    verifications = []

    for row in agent_events:
        e = format_event(row)
        if e["type"] == "RESULT":
            agent_results.append(e)
        elif e["type"] == "CLAIM":
            agent_claims.append(e)
        elif e["type"] == "FAILURE":
            agent_failures.append(e)
        elif e["type"] == "VERIFY":
            verifications.append(e)

    # Get unique agents
    agents = db.execute("SELECT id, name, platform FROM agents WHERE id != 'OPERATOR'").fetchall()
    agent_list = [dict(a) for a in agents]

    # Best result
    best = db.execute(
        """SELECT agent_id, platform, json_extract(payload, '$.score') as score,
            json_extract(payload, '$.description') as description
        FROM events WHERE type = 'RESULT'
        AND json_extract(payload, '$.status') = 'keep'
        AND json_extract(payload, '$.score') IS NOT NULL
        ORDER BY CAST(json_extract(payload, '$.score') AS REAL) ASC LIMIT 1"""
    ).fetchone()

    # Build the interaction timeline
    timeline = []
    all_events = db.execute(
        "SELECT * FROM events WHERE type IN ('OPERATOR','RESULT','CLAIM','VERIFY','FAILURE','FACT') ORDER BY id ASC"
    ).fetchall()
    for row in all_events:
        e = format_event(row)
        p = e["payload"]
        if e["agent_id"] == "OPERATOR":
            role = "Human"
            action = p.get("message", p.get("content", ""))
        elif e["agent_id"] == "PLAYBOOK":
            role = "System"
            action = p.get("message", p.get("content", ""))
        else:
            role = e["agent_id"]
            if e["type"] == "RESULT":
                score = p.get("score")
                score_str = f"{score:.6f}" if score else "crash"
                action = f"Experiment: {p.get('description', '')} → {score_str} ({p.get('status', '')})"
            elif e["type"] == "CLAIM":
                action = f"Claim: {p.get('message', '')}"
            elif e["type"] == "VERIFY":
                subtype = p.get("subtype", "")
                if subtype == "request":
                    action = f"Verification requested for score {p.get('original_score', '')}"
                else:
                    action = f"Verified: got {p.get('reproduced_score', '')} ({p.get('verdict', '')})"
            elif e["type"] == "FAILURE":
                action = f"Dead end: {p.get('content', '')}"
            elif e["type"] == "FACT":
                action = f"Confirmed: {p.get('content', '')}"
            else:
                action = str(p)[:100]

        timeline.append({
            "id": e["id"],
            "role": role,
            "type": e["type"],
            "action": action,
            "created_at": e["created_at"],
        })

    # Autonomy level assessment (Aletheia Table 8)
    total_operator = len(human_directives)
    total_agent = len(agent_results) + len(agent_claims)
    if total_operator == 0 and total_agent > 0:
        autonomy = {"level": "A", "label": "Essentially Autonomous", "description": "Core optimization fully AI-driven without essential human intervention."}
    elif total_operator > 0 and total_agent > total_operator * 3:
        autonomy = {"level": "C", "label": "Human-AI Collaboration", "description": "Both human directives and AI exploration contributed substantively."}
    else:
        autonomy = {"level": "H", "label": "Primarily Human", "description": "Human directives drove the optimization; AI executed."}

    return {
        "title": "Human-AI Interaction Card",
        "version": "1.0",
        "inspired_by": "Aletheia (Google DeepMind, arxiv:2602.10177v3)",
        "summary": {
            "agents": len(agent_list),
            "total_experiments": len(agent_results),
            "human_directives": total_operator,
            "claims": len(agent_claims),
            "dead_ends": len(agent_failures),
            "verifications": len(verifications),
            "best_result": dict(best) if best else None,
        },
        "autonomy_level": autonomy,
        "agents": agent_list,
        "timeline": timeline[-50:],  # Last 50 interactions
        "human_contributions": [
            {"type": e["type"], "message": e["payload"].get("message", e["payload"].get("content", "")), "created_at": e["created_at"]}
            for e in human_directives
        ],
    }


@app.get("/api/hai-card/markdown")
def hai_card_markdown(
    agent_id: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
):
    """Render a Human-AI Interaction Card as Markdown (for GitHub notebooks)."""
    card = hai_card(agent_id=agent_id, db=db)
    s = card["summary"]
    a = card["autonomy_level"]

    md = f"""# Human-AI Interaction Card

> Inspired by [Aletheia](https://arxiv.org/abs/2602.10177v3) (Google DeepMind)

## Summary

| Metric | Value |
|--------|-------|
| Agents | {s['agents']} |
| Experiments | {s['total_experiments']} |
| Human Directives | {s['human_directives']} |
| Claims | {s['claims']} |
| Dead Ends | {s['dead_ends']} |
| Verifications | {s['verifications']} |
| Best Score | {s['best_result']['score'] if s['best_result'] else 'N/A'} |

## Autonomy Level: {a['level']} — {a['label']}

{a['description']}

## Human Contributions

"""
    for h in card["human_contributions"]:
        md += f"- **{h['type']}** ({h['created_at'][:16]}): {h['message']}\n"

    if not card["human_contributions"]:
        md += "*No human directives — fully autonomous run.*\n"

    md += "\n## Interaction Timeline (last 50)\n\n"
    md += "| # | Role | Type | Action |\n|---|------|------|--------|\n"
    for t in card["timeline"]:
        action = t["action"][:80].replace("|", "\\|")
        md += f"| {t['id']} | {t['role']} | {t['type']} | {action} |\n"

    md += f"\n---\n*Generated by researchRalph Hub v0.3 — {card['inspired_by']}*\n"
    return {"markdown": md}


# ═══════════════════════════════════════════════════════════════
# DASHBOARD — TweetDeck-style multi-column view with live SSE
# ═══════════════════════════════════════════════════════════════


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: sqlite3.Connection = Depends(get_db)):
    total_agents = db.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    total_events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    best_row = db.execute(
        "SELECT MIN(CAST(json_extract(payload, '$.score') AS REAL)) FROM events WHERE type='RESULT' AND json_extract(payload, '$.status')='keep'"
    ).fetchone()
    best = best_row[0] if best_row else None
    platform_count = db.execute("SELECT COUNT(DISTINCT platform) FROM events WHERE platform != ''").fetchone()[0]

    # ── Column 1: All Events (firehose) ──
    all_events = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 40").fetchall()
    firehose_html = ""
    for r in all_events:
        e = format_event(r)
        p = e["payload"]
        agent = esc(e["agent_id"])
        when = time_ago(e["created_at"])
        etype = e["type"]
        platform_tag = f' <span class="platform-tag">{esc(e["platform"])}</span>' if e["platform"] else ""

        # Format content based on type
        if etype == "RESULT":
            score = f"{p['score']:.6f}" if p.get("score") else "—"
            status_cls = p.get("status", "keep")
            content = f'<span class="{status_cls}">{score}</span> {esc(p.get("description", ""))}'
        elif etype in BLACKBOARD_TYPES:
            content = esc(p.get("message", ""))
        elif etype in MEMORY_TYPES:
            content = esc(p.get("content", ""))
        elif etype == "COMMIT":
            h = esc(str(p.get("hash", ""))[:8])
            content = f'<span class="mono">{h}</span> {esc(p.get("message", ""))}'
        elif etype == "POST":
            content = f'<span class="channel">#{esc(p.get("channel", ""))}</span> {esc(p.get("content", ""))}'
        elif etype in REACTION_TYPES:
            content = f'on #{e["reply_to"]} — {esc(p.get("reason", ""))}'
        else:
            content = esc(str(p)[:100])

        reply_badge = f' <span class="reply-badge">reply to #{e["reply_to"]}</span>' if e["reply_to"] else ""
        tags_html = "".join(f' <span class="tag">#{t}</span>' for t in e.get("tags", []))
        firehose_html += f'<div class="event" data-type="{etype.lower()}" id="event-{e["id"]}"><span class="type-badge {etype.lower()}">{etype}</span> <span class="agent-tag">{agent}</span>{platform_tag} <span class="dim">{when}</span>{reply_badge}{tags_html}<div class="event-content">{content}</div></div>\n'

    # ── Column 2: Results (by score) ──
    results = db.execute(
        """SELECT * FROM events WHERE type = 'RESULT'
        AND json_extract(payload, '$.status') = 'keep'
        AND json_extract(payload, '$.score') IS NOT NULL
        ORDER BY CAST(json_extract(payload, '$.score') AS REAL) ASC LIMIT 20"""
    ).fetchall()
    results_html = ""
    for r in results:
        e = format_event(r)
        p = e["payload"]
        score = f"{p['score']:.6f}" if p.get("score") else "—"
        agent = esc(e["agent_id"])
        desc = esc(p.get("description", ""))
        plat = esc(e["platform"])
        when = time_ago(e["created_at"])
        # Count reactions
        confirms = db.execute("SELECT COUNT(*) FROM events WHERE type='CONFIRM' AND reply_to=?", (e["id"],)).fetchone()[0]
        contradicts = db.execute("SELECT COUNT(*) FROM events WHERE type='CONTRADICT' AND reply_to=?", (e["id"],)).fetchone()[0]
        reaction_html = ""
        if confirms:
            reaction_html += f' <span class="reaction confirm-count">{confirms} confirmed</span>'
        if contradicts:
            reaction_html += f' <span class="reaction contradict-count">{contradicts} contradicted</span>'
        results_html += f'<div class="result-row"><span class="score">{score}</span> <span class="agent-tag">{agent}</span> <span class="platform-tag">{plat}</span>{reaction_html}<div class="dim">{desc} · {when}</div></div>\n'
    if not results_html:
        results_html = '<p class="dim">No results yet</p>'

    # ── Column 3: Claims + Threads ──
    claims = db.execute(
        """SELECT * FROM events WHERE type IN ('CLAIM','RESPONSE','REQUEST','REFUTE')
        ORDER BY id DESC LIMIT 20"""
    ).fetchall()
    claims_html = ""
    for r in claims:
        e = format_event(r)
        p = e["payload"]
        cls = e["type"].lower()
        agent = esc(e["agent_id"])
        msg = esc(p.get("message", "")).replace("\n", "<br>")
        when = time_ago(e["created_at"])
        target = f' &rarr; {esc(p.get("target", ""))}' if p.get("target") else ""
        reply = f' <span class="reply-badge">reply to #{e["reply_to"]}</span>' if e["reply_to"] else ""
        claims_html += f'<div class="bb-msg {cls}"><span class="agent-tag">{agent}</span>{target} <span class="dim">{when}</span>{reply}<br><strong>{e["type"]}</strong>: {msg}</div>\n'
    if not claims_html:
        claims_html = '<p class="dim">No claims yet</p>'

    # ── Column 4: Operator + Memory ──
    ops = db.execute(
        "SELECT * FROM events WHERE type = 'OPERATOR' ORDER BY id DESC LIMIT 10"
    ).fetchall()
    ops_html = ""
    for r in ops:
        e = format_event(r)
        p = e["payload"]
        msg = esc(p.get("message", "")).replace("\n", "<br>")
        when = time_ago(e["created_at"])
        tags_html = "".join(f' <span class="tag">#{t}</span>' for t in e.get("tags", []))
        ops_html += f'<div class="bb-msg operator"><span class="dim">{when}</span>{tags_html}<div>{msg}</div></div>\n'
    if not ops_html:
        ops_html = '<p class="dim">No operator messages</p>'

    mems = db.execute(
        "SELECT * FROM events WHERE type IN ('FACT','FAILURE','HUNCH') ORDER BY id DESC LIMIT 15"
    ).fetchall()
    mem_html = ""
    for r in mems:
        e = format_event(r)
        p = e["payload"]
        cls = e["type"].lower()
        icon = {"fact": "&#10003;", "failure": "&#10007;", "hunch": "?"}.get(cls, "&middot;")
        content = esc(p.get("content", ""))
        agent = esc(e["agent_id"])
        mem_html += f'<div class="mem {cls}"><span class="mem-icon">{icon}</span> <span class="dim">[{agent}]</span> {content}</div>\n'
    if not mem_html:
        mem_html = '<p class="dim">No shared memory</p>'

    # ── Agents ──
    agents = db.execute("SELECT * FROM agents ORDER BY last_seen DESC NULLS LAST").fetchall()
    agents_html = ""
    for a in agents:
        stats = db.execute(
            """SELECT COUNT(*) as exp,
                MIN(CAST(json_extract(payload, '$.score') AS REAL)) as best
            FROM events WHERE type='RESULT' AND agent_id=?""",
            (a["id"],),
        ).fetchone()
        best_str = f"{stats['best']:.6f}" if stats["best"] else "—"
        last = time_ago(a["last_seen"]) if a["last_seen"] else "never"
        agents_html += f'<tr><td>{esc(a["name"])}</td><td>{esc(a["platform"])}</td><td>{stats["exp"]}</td><td class="score">{best_str}</td><td class="dim">{last}</td></tr>\n'

    return DASHBOARD_HTML.format(
        total_agents=total_agents,
        total_events=total_events,
        platform_count=platform_count,
        best_score=f"{best:.6f}" if best else "—",
        firehose_html=firehose_html or '<p class="dim">No events yet</p>',
        results_html=results_html,
        claims_html=claims_html,
        ops_html=ops_html,
        mem_html=mem_html,
        agents_html=agents_html or '<tr><td colspan="5" class="dim">No agents</td></tr>',
    )


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>researchRalph Hub</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 1rem; font-size: 12px; }}
  h1 {{ color: #f0f6fc; margin-bottom: 0.25rem; font-size: 1.3rem; }}
  h2 {{ color: #8b949e; margin: 0 0 0.5rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }}
  .subtitle {{ color: #484f58; margin-bottom: 1rem; font-size: 0.8rem; }}
  .dim {{ color: #484f58; }}
  .mono {{ font-family: 'SF Mono', 'Menlo', monospace; }}

  /* Stats bar */
  .stats {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; }}
  .stat {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.5rem 0.75rem; flex: 1; text-align: center; }}
  .stat-num {{ font-size: 1.3rem; color: #58a6ff; font-weight: bold; }}
  .stat-label {{ color: #484f58; font-size: 0.7rem; text-transform: uppercase; }}

  /* TweetDeck columns */
  .columns {{ display: grid; grid-template-columns: 2fr 1.2fr 1.5fr 1.3fr; gap: 0.75rem; margin-bottom: 1rem; }}
  .column {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.75rem; max-height: 70vh; overflow-y: auto; }}
  .column h2 {{ position: sticky; top: 0; background: #161b22; padding-bottom: 0.5rem; z-index: 1; }}
  @media (max-width: 1200px) {{ .columns {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 700px) {{ .columns {{ grid-template-columns: 1fr; }} }}

  /* Events */
  .event {{ padding: 0.5rem; margin-bottom: 0.4rem; border-radius: 4px; border-left: 3px solid #21262d; background: #0d1117; }}
  .event-content {{ margin-top: 0.3rem; line-height: 1.4; }}
  .type-badge {{ display: inline-block; font-size: 0.65rem; padding: 1px 5px; border-radius: 3px; font-weight: 600; color: #0d1117; }}
  .type-badge.result {{ background: #58a6ff; }}
  .type-badge.claim {{ background: #58a6ff; }}
  .type-badge.response {{ background: #3fb950; }}
  .type-badge.request {{ background: #d29922; }}
  .type-badge.refute {{ background: #f85149; }}
  .type-badge.operator {{ background: #bc8cff; }}
  .type-badge.fact {{ background: #3fb950; }}
  .type-badge.failure {{ background: #f85149; }}
  .type-badge.hunch {{ background: #d29922; }}
  .type-badge.commit {{ background: #8b949e; }}
  .type-badge.post {{ background: #484f58; color: #c9d1d9; }}
  .type-badge.confirm {{ background: #238636; }}
  .type-badge.contradict {{ background: #da3633; }}
  .type-badge.adopt {{ background: #1f6feb; }}
  .type-badge.playbook {{ background: #6e40c9; }}
  .type-badge.verify {{ background: #39d353; }}
  .type-badge.heartbeat {{ background: #21262d; color: #484f58; }}
  .agent-tag {{ color: #d2a8ff; }}
  .platform-tag {{ color: #8b949e; font-size: 0.75rem; background: #21262d; padding: 1px 4px; border-radius: 3px; }}
  .channel {{ color: #58a6ff; font-weight: bold; }}
  .reply-badge {{ font-size: 0.7rem; color: #8b949e; background: #21262d; padding: 1px 4px; border-radius: 3px; }}
  .tag {{ font-size: 0.65rem; color: #7ee787; background: #0d1117; border: 1px solid #238636; padding: 0px 4px; border-radius: 3px; }}

  /* Results */
  .result-row {{ padding: 0.4rem; margin-bottom: 0.3rem; border-left: 2px solid #58a6ff; background: #0d1117; border-radius: 0 4px 4px 0; }}
  .score {{ color: #58a6ff; font-weight: bold; font-size: 1.1em; }}
  .keep {{ color: #3fb950; }}
  .discard {{ color: #f85149; }}
  .crash {{ color: #d29922; }}
  .reaction {{ font-size: 0.7rem; padding: 1px 4px; border-radius: 3px; }}
  .confirm-count {{ background: #0d2818; color: #3fb950; }}
  .contradict-count {{ background: #2d0b0b; color: #f85149; }}

  /* Blackboard */
  .bb-msg {{ padding: 0.5rem; margin-bottom: 0.4rem; border-radius: 4px; border-left: 3px solid #21262d; background: #0d1117; }}
  .bb-msg.claim {{ border-left-color: #58a6ff; }}
  .bb-msg.response {{ border-left-color: #3fb950; }}
  .bb-msg.request {{ border-left-color: #d29922; }}
  .bb-msg.refute {{ border-left-color: #f85149; }}
  .bb-msg.operator {{ border-left-color: #bc8cff; background: #1c1230; }}

  /* Memory */
  .mem {{ padding: 0.25rem 0; line-height: 1.4; }}
  .mem.fact {{ color: #3fb950; }}
  .mem.failure {{ color: #f85149; }}
  .mem.hunch {{ color: #d29922; }}
  .mem-icon {{ font-weight: bold; }}

  /* Agents table */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; color: #8b949e; font-weight: 500; padding: 0.3rem 0.5rem; border-bottom: 1px solid #21262d; font-size: 0.75rem; }}
  td {{ padding: 0.3rem 0.5rem; border-bottom: 1px solid #161b22; }}
  tr:hover {{ background: #161b22; }}

  .footer {{ margin-top: 1rem; color: #484f58; font-size: 0.75rem; }}
  .footer a {{ color: #58a6ff; text-decoration: none; }}

  /* Live indicator */
  .live-dot {{ display: inline-block; width: 8px; height: 8px; background: #3fb950; border-radius: 50%; margin-right: 4px; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
  #connection-status {{ font-size: 0.75rem; color: #3fb950; }}
</style>
</head>
<body>
<h1>researchRalph Hub <span class="live-dot"></span><span id="connection-status">connecting...</span></h1>
<p class="subtitle">v0.3 — unified event stream + playbooks</p>

<div class="stats">
  <div class="stat"><div class="stat-num">{total_agents}</div><div class="stat-label">Agents</div></div>
  <div class="stat"><div class="stat-num">{total_events}</div><div class="stat-label">Events</div></div>
  <div class="stat"><div class="stat-num">{platform_count}</div><div class="stat-label">Platforms</div></div>
  <div class="stat"><div class="stat-num">{best_score}</div><div class="stat-label">Best Score</div></div>
</div>

<div class="columns">
  <div class="column" id="col-firehose">
    <h2>All Events</h2>
    <div id="firehose">{firehose_html}</div>
  </div>
  <div class="column" id="col-results">
    <h2>Results (by score)</h2>
    <div id="results">{results_html}</div>
  </div>
  <div class="column" id="col-claims">
    <h2>Claims + Discussion</h2>
    <div id="claims">{claims_html}</div>
  </div>
  <div class="column" id="col-ops">
    <h2>Operator</h2>
    <div id="ops">{ops_html}</div>
    <h2 style="margin-top:1rem;">Shared Memory</h2>
    <div id="memory">{mem_html}</div>
  </div>
</div>

<h2>Agents</h2>
<table>
<tr><th>Name</th><th>Platform</th><th>Experiments</th><th>Best</th><th>Last Seen</th></tr>
{agents_html}
</table>

<p class="footer">
  <a href="/docs">API Docs</a> &middot;
  <a href="/api/stream">Raw Stream (SSE)</a> &middot;
  Operator: POST /api/operator/{{claim,ban,directive,strategy}} &middot;
  Powered by <a href="https://github.com/bigsnarfdude/researchRalph">researchRalph v2</a>
</p>

<script>
// Live SSE connection — append new events to columns without page refresh
(function() {{
  const status = document.getElementById('connection-status');
  const firehose = document.getElementById('firehose');

  // Find the highest event ID on the page
  let lastId = 0;
  document.querySelectorAll('[id^="event-"]').forEach(el => {{
    const id = parseInt(el.id.replace('event-', ''));
    if (id > lastId) lastId = id;
  }});

  function connect() {{
    const es = new EventSource('/api/stream?since_id=' + lastId);

    es.onopen = function() {{
      status.textContent = 'live';
      status.style.color = '#3fb950';
    }};

    es.onmessage = function(e) {{
      try {{
        const event = JSON.parse(e.data);
        lastId = Math.max(lastId, event.id);
        addToFirehose(event);
      }} catch(err) {{
        console.error('SSE parse error:', err);
      }}
    }};

    es.onerror = function() {{
      status.textContent = 'reconnecting...';
      status.style.color = '#d29922';
      es.close();
      setTimeout(connect, 3000);
    }};
  }}

  function addToFirehose(event) {{
    const div = document.createElement('div');
    div.className = 'event';
    div.id = 'event-' + event.id;
    div.dataset.type = event.type.toLowerCase();

    const p = event.payload || {{}};
    let content = '';
    if (event.type === 'RESULT') {{
      const score = p.score ? p.score.toFixed(6) : '—';
      content = '<span class="' + (p.status || 'keep') + '">' + score + '</span> ' + escHtml(p.description || '');
    }} else if (['CLAIM','RESPONSE','REQUEST','REFUTE','OPERATOR'].includes(event.type)) {{
      content = escHtml(p.message || '');
    }} else if (['FACT','FAILURE','HUNCH'].includes(event.type)) {{
      content = escHtml(p.content || '');
    }} else if (event.type === 'COMMIT') {{
      content = '<span class="mono">' + escHtml((p.hash || '').slice(0,8)) + '</span> ' + escHtml(p.message || '');
    }} else if (event.type === 'POST') {{
      content = '<span class="channel">#' + escHtml(p.channel || '') + '</span> ' + escHtml(p.content || '');
    }} else {{
      content = escHtml(JSON.stringify(p).slice(0,100));
    }}

    const platform = event.platform ? ' <span class="platform-tag">' + escHtml(event.platform) + '</span>' : '';
    const tags = (event.tags || []).map(t => ' <span class="tag">#' + t + '</span>').join('');
    const replyBadge = event.reply_to ? ' <span class="reply-badge">reply to #' + event.reply_to + '</span>' : '';

    div.innerHTML = '<span class="type-badge ' + event.type.toLowerCase() + '">' + event.type + '</span> '
      + '<span class="agent-tag">' + escHtml(event.agent_id) + '</span>'
      + platform + ' <span class="dim">just now</span>' + replyBadge + tags
      + '<div class="event-content">' + content + '</div>';

    // Flash effect
    div.style.borderLeftColor = '#58a6ff';
    setTimeout(() => {{ div.style.borderLeftColor = '#21262d'; }}, 2000);

    firehose.insertBefore(div, firehose.firstChild);

    // Keep firehose from growing unbounded
    while (firehose.children.length > 60) {{
      firehose.removeChild(firehose.lastChild);
    }}
  }}

  function escHtml(s) {{
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  }}

  connect();
}})();
</script>
</body>
</html>"""


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="researchRalph Hub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  researchRalph Hub v0.3 — Unified Event Stream")
    print(f"  API:       http://{args.host}:{args.port}/api")
    print(f"  Stream:    http://{args.host}:{args.port}/api/stream")
    print(f"  Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"  Database:  {DB_PATH}")
    print(f"  Playbooks: {', '.join(pb._playbook_name for pb in PLAYBOOKS)}\n")

    uvicorn.run(app, host=args.host, port=args.port)

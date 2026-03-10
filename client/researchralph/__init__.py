"""
researchRalph — Python client for the Hub API.

Usage:
    from researchralph import Hub

    hub = Hub("http://localhost:8000", key="rr_...")

    # Read
    for event in hub.since(0, types=["CLAIM", "OPERATOR"]):
        print(event)

    leaderboard = hub.leaderboard(platform="GH200")
    failures = hub.memory(type="failure")

    # Write
    hub.result(score=1.037, status="keep", description="AR96+batch2^17")
    hub.claim("WD cosine > linear", evidence={"runs": 3})
    hub.failure("depth 12 = OOM at 62GB")
    hub.fact("LR 0.08 better than 0.04")
    hub.hunch("weight decay interacts with batch size")

    # React
    hub.confirm(event_id=42, reason="reproduced on my GPU")
    hub.contradict(event_id=42, reason="didn't hold on 4070Ti")
    hub.adopt(event_id=42, reason="using as new baseline")

    # Stream (blocking — for daemon-style agents)
    for event in hub.stream(types=["OPERATOR", "CLAIM"]):
        handle(event)

    # Post raw event
    hub.event("REQUEST", payload={"message": "test HEAD_DIM=64"}, tags=["optimizer"])
"""

__version__ = "0.3.0"

import json
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class HubError(Exception):
    """Error from the hub API."""
    def __init__(self, status, message):
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


class Hub:
    """Client for the researchRalph Hub API.

    Args:
        url: Hub base URL (e.g. "http://localhost:8000")
        key: API key from /api/register (e.g. "rr_...")
        agent_id: Optional agent ID (for logging)
    """

    def __init__(self, url: str, key: str = "", agent_id: str = ""):
        self.url = url.rstrip("/")
        self.key = key
        self.agent_id = agent_id
        self._last_seen_id = 0

    @property
    def last_seen(self) -> int:
        """Last event ID seen by this client."""
        return self._last_seen_id

    # ─── Low-level ───────────────────────────────────────────

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        url = f"{self.url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            raise HubError(e.code, e.read().decode()[:200])

    def _get(self, path: str, params: dict = None) -> dict:
        if params:
            qs = "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v is not None)
            path = f"{path}?{qs}" if qs else path
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    # ─── Registration ────────────────────────────────────────

    @classmethod
    def register(cls, url: str, name: str, team: str = "", platform: str = "unknown") -> "Hub":
        """Register a new agent and return a connected Hub client."""
        hub = cls(url)
        resp = hub._post("/api/register", {"name": name, "team": team, "platform": platform})
        hub.key = resp["api_key"]
        hub.agent_id = resp["agent_id"]
        return hub

    # ─── Events (unified API) ────────────────────────────────

    def event(self, etype: str, payload: dict = None, tags: list = None, reply_to: int = None) -> dict:
        """Post a raw event to the stream."""
        body = {"type": etype, "payload": payload or {}, "tags": tags or []}
        if reply_to:
            body["reply_to"] = reply_to
        return self._post("/api/events", body)

    def events(self, types: str = None, agent: str = None, platform: str = None,
               tags: str = None, since_id: int = 0, limit: int = 50) -> list:
        """Get events with filters."""
        params = {"types": types, "agent": agent, "platform": platform,
                  "tags": tags, "since_id": since_id, "limit": limit}
        result = self._get("/api/events", params)
        for e in result:
            self._last_seen_id = max(self._last_seen_id, e.get("id", 0))
        return result

    def since(self, since_id: int = None, types: list = None) -> list:
        """Get events since a given ID. Convenience for polling."""
        sid = since_id if since_id is not None else self._last_seen_id
        type_str = ",".join(types) if types else None
        return self.events(types=type_str, since_id=sid)

    # ─── Results ─────────────────────────────────────────────

    def result(self, score: float = None, status: str = "keep", description: str = "",
               commit_hash: str = "", memory_gb: float = 0) -> dict:
        """Post an experiment result."""
        return self._post("/api/results", {
            "score": score, "status": status, "description": description,
            "commit_hash": commit_hash, "memory_gb": memory_gb,
        })

    def results(self, agent: str = None, status: str = None, limit: int = 50) -> list:
        """Get results."""
        return self._get("/api/results", {"agent": agent, "status": status, "limit": limit})

    def leaderboard(self, top: int = 10, platform: str = None) -> list:
        """Get leaderboard."""
        return self._get("/api/results/leaderboard", {"top": top, "platform": platform})

    # ─── Blackboard ──────────────────────────────────────────

    def claim(self, message: str, evidence: dict = None, tags: list = None) -> dict:
        """Post a CLAIM to the blackboard."""
        return self._post("/api/blackboard", {
            "type": "CLAIM", "message": message, "evidence": evidence or {},
        })

    def request(self, message: str, target: str = "any") -> dict:
        """Post a REQUEST to the blackboard."""
        return self._post("/api/blackboard", {
            "type": "REQUEST", "message": message, "target": target,
        })

    def respond(self, in_reply_to: int, message: str) -> dict:
        """Post a RESPONSE to the blackboard."""
        return self._post("/api/blackboard", {
            "type": "RESPONSE", "message": message, "in_reply_to": in_reply_to,
        })

    def refute(self, in_reply_to: int, message: str) -> dict:
        """Post a REFUTE to the blackboard."""
        return self._post("/api/blackboard", {
            "type": "REFUTE", "message": message, "in_reply_to": in_reply_to,
        })

    def blackboard(self, type: str = None, limit: int = 50, since_id: int = None) -> list:
        """Get blackboard messages."""
        return self._get("/api/blackboard", {"type": type, "limit": limit, "since_id": since_id})

    # ─── Memory ──────────────────────────────────────────────

    def fact(self, content: str) -> dict:
        """Record a confirmed fact."""
        return self._post("/api/memory", {"type": "fact", "content": content})

    def failure(self, content: str) -> dict:
        """Record a dead end."""
        return self._post("/api/memory", {"type": "failure", "content": content})

    def hunch(self, content: str) -> dict:
        """Record a hypothesis."""
        return self._post("/api/memory", {"type": "hunch", "content": content})

    def memory(self, type: str = None, limit: int = 100) -> list:
        """Get shared memory entries."""
        return self._get("/api/memory", {"type": type, "limit": limit})

    # ─── Commits ─────────────────────────────────────────────

    def commit(self, hash: str, message: str, parent: str = "", score: float = None,
               status: str = "keep", memory_gb: float = 0) -> dict:
        """Post a git commit."""
        return self._post("/api/commits", {
            "hash": hash, "parent": parent, "message": message,
            "score": score, "status": status, "memory_gb": memory_gb,
        })

    def commits(self, agent: str = None, limit: int = 50) -> list:
        """Get commits."""
        return self._get("/api/commits", {"agent": agent, "limit": limit})

    # ─── Posts (channels) ────────────────────────────────────

    def post(self, content: str, channel: str = "results") -> dict:
        """Post to a channel."""
        return self._post("/api/posts", {"channel": channel, "content": content})

    def posts(self, channel: str = None, limit: int = 50, since_id: int = None) -> list:
        """Get channel posts."""
        return self._get("/api/posts", {"channel": channel, "limit": limit, "since_id": since_id})

    # ─── Reactions ───────────────────────────────────────────

    def confirm(self, event_id: int, reason: str = "") -> dict:
        """Confirm/reproduce an event's finding."""
        return self._post(f"/api/events/{event_id}/confirm", {"reason": reason})

    def contradict(self, event_id: int, reason: str = "") -> dict:
        """Contradict an event's finding."""
        return self._post(f"/api/events/{event_id}/contradict", {"reason": reason})

    def adopt(self, event_id: int, reason: str = "") -> dict:
        """Adopt an event's config as baseline."""
        return self._post(f"/api/events/{event_id}/adopt", {"reason": reason})

    # ─── Verification (Aletheia-inspired) ───────────────────

    def verify_queue(self, platform: str = None) -> list:
        """Get pending verification requests."""
        return self._get("/api/verify/queue", {"platform": platform})

    def verify(self, verify_request_id: int, reproduced_score: float,
               verdict: str = "confirmed", notes: str = "") -> dict:
        """Post a verification result (reproduce another agent's claimed result)."""
        params = {
            "verify_request_id": verify_request_id,
            "reproduced_score": reproduced_score,
            "verdict": verdict,
            "notes": notes,
        }
        qs = "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v is not None)
        return self._post(f"/api/verify?{qs}", {})

    # ─── HAI Cards (Aletheia-inspired) ────────────────────

    def hai_card(self, agent_id: str = None) -> dict:
        """Get a Human-AI Interaction Card (contribution breakdown)."""
        return self._get("/api/hai-card", {"agent_id": agent_id})

    def hai_card_markdown(self, agent_id: str = None) -> str:
        """Get a Human-AI Interaction Card as Markdown."""
        result = self._get("/api/hai-card/markdown", {"agent_id": agent_id})
        return result.get("markdown", "")

    # ─── Agents ──────────────────────────────────────────────

    def agents(self) -> list:
        """List all agents."""
        return self._get("/api/agents")

    def agent(self, agent_id: str) -> dict:
        """Get agent details."""
        return self._get(f"/api/agents/{agent_id}")

    # ─── Streaming (SSE) ─────────────────────────────────────

    def stream(self, types: list = None, agent: str = None, platform: str = None,
               tags: list = None, since_id: int = None):
        """Subscribe to the event stream (blocking generator).

        Yields event dicts as they arrive. Reconnects on failure.

        Usage:
            for event in hub.stream(types=["OPERATOR", "CLAIM"]):
                if event["type"] == "OPERATOR":
                    follow_directive(event)
        """
        sid = since_id if since_id is not None else self._last_seen_id
        params = {}
        if types:
            params["types"] = ",".join(types)
        if agent:
            params["agent"] = agent
        if platform:
            params["platform"] = platform
        if tags:
            params["tags"] = ",".join(tags)
        params["since_id"] = sid

        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.url}/api/stream?{qs}"

        while True:
            try:
                req = Request(url)
                with urlopen(req, timeout=300) as resp:
                    buffer = ""
                    while True:
                        chunk = resp.read(1).decode()
                        if not chunk:
                            break
                        buffer += chunk
                        if buffer.endswith("\n\n"):
                            for line in buffer.strip().split("\n"):
                                if line.startswith("data: "):
                                    try:
                                        event = json.loads(line[6:])
                                        self._last_seen_id = max(self._last_seen_id, event.get("id", 0))
                                        yield event
                                    except json.JSONDecodeError:
                                        pass
                                elif line.startswith("id: "):
                                    # Update since_id for reconnection
                                    try:
                                        params["since_id"] = int(line[4:])
                                        qs = "&".join(f"{k}={v}" for k, v in params.items())
                                        url = f"{self.url}/api/stream?{qs}"
                                    except ValueError:
                                        pass
                            buffer = ""
            except (URLError, OSError, TimeoutError):
                time.sleep(3)
                continue

    # ─── Convenience ─────────────────────────────────────────

    def check_operator(self) -> list:
        """Check for operator directives (no auth required)."""
        return self.blackboard(type="OPERATOR")

    def check_failures(self) -> list:
        """Get all recorded failures."""
        return self.memory(type="failure")

    def check_facts(self) -> list:
        """Get all confirmed facts."""
        return self.memory(type="fact")

    def __repr__(self):
        return f"Hub({self.url!r}, agent={self.agent_id!r})"

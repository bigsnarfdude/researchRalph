"""
researchRalph Hub API test skeleton.

All test method signatures defined; bodies are `pass` (to be implemented).
The test server infrastructure (setUpClass/tearDownClass) is fully functional.

Run:
    python -m unittest tests.test_hub -v
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


# ─── Test Server Base ─────────────────────────────────────────

TEST_PORT = 8765
TEST_URL = f"http://127.0.0.1:{TEST_PORT}"


class TestServerBase(unittest.TestCase):
    """Base class that starts a real Hub server for testing."""

    server_thread = None
    _original_db_path = None
    _tmp_db = None

    @classmethod
    def setUpClass(cls):
        """Start a real uvicorn server on TEST_PORT with a temp database."""
        import importlib

        # Create temp database
        cls._tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp_db.close()

        # Patch DB_PATH before importing/reloading server
        import hub.server as server_mod

        cls._original_db_path = server_mod.DB_PATH
        server_mod.DB_PATH = Path(cls._tmp_db.name)

        # Initialize the database
        server_mod.init_db()

        # Reimport to pick up patched path
        importlib.reload(server_mod)
        server_mod.DB_PATH = Path(cls._tmp_db.name)
        server_mod.init_db()

        # Start uvicorn in a daemon thread
        import uvicorn

        config = uvicorn.Config(
            server_mod.app,
            host="127.0.0.1",
            port=TEST_PORT,
            log_level="error",
        )
        cls._server = uvicorn.Server(config)
        cls.server_thread = threading.Thread(target=cls._server.run, daemon=True)
        cls.server_thread.start()

        # Wait for server to be ready
        for _ in range(30):
            try:
                urlopen(f"{TEST_URL}/api/agents", timeout=1)
                break
            except (URLError, OSError):
                time.sleep(0.1)
        else:
            raise RuntimeError("Test server did not start within 3 seconds")

    @classmethod
    def tearDownClass(cls):
        """Shut down the server and clean up the temp database."""
        if hasattr(cls, "_server"):
            cls._server.should_exit = True
        if cls.server_thread:
            cls.server_thread.join(timeout=5)

        # Restore original DB_PATH
        import hub.server as server_mod

        if cls._original_db_path:
            server_mod.DB_PATH = cls._original_db_path

        # Delete temp database
        if cls._tmp_db:
            try:
                os.unlink(cls._tmp_db.name)
            except OSError:
                pass

    def setUp(self):
        """Truncate all tables for a clean slate per test."""
        import hub.server as server_mod

        db = sqlite3.connect(str(server_mod.DB_PATH))
        db.execute("DELETE FROM events")
        db.execute("DELETE FROM agents")
        db.commit()
        db.close()

    # ─── Helpers ──────────────────────────────────────────────

    def _request(self, method, path, body=None, key=None):
        """Make an HTTP request to the test server. Returns parsed JSON."""
        url = f"{TEST_URL}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode()), resp.status

    def _get(self, path, key=None):
        return self._request("GET", path, key=key)

    def _post(self, path, body, key=None):
        return self._request("POST", path, body=body, key=key)

    def _register(self, name="test-agent", team="", platform="unknown"):
        """Register an agent, return (agent_id, api_key)."""
        resp, status = self._post("/api/register", {"name": name, "team": team, "platform": platform})
        return resp["agent_id"], resp["api_key"]


# ═══════════════════════════════════════════════════════════════
# Registration Tests
# ═══════════════════════════════════════════════════════════════


class TestHubRegistration(TestServerBase):

    def test_register_agent(self):
        """POST /api/register returns agent_id and api_key."""
        resp, status = self._post("/api/register", {"name": "alice", "team": "red", "platform": "GH200"})
        self.assertEqual(status, 200)
        self.assertIn("agent_id", resp)
        self.assertIn("api_key", resp)
        self.assertTrue(resp["agent_id"].startswith("alice-"))
        self.assertTrue(resp["api_key"].startswith("rr_"))

    def test_register_returns_unique_ids(self):
        """Two registrations with same name get different agent_ids."""
        r1, _ = self._post("/api/register", {"name": "bob"})
        r2, _ = self._post("/api/register", {"name": "bob"})
        self.assertNotEqual(r1["agent_id"], r2["agent_id"])
        self.assertNotEqual(r1["api_key"], r2["api_key"])

    def test_register_missing_name(self):
        """POST with empty body returns 422."""
        try:
            self._post("/api/register", {})
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 422)

    def test_register_default_platform(self):
        """POST with name only defaults platform to 'unknown'."""
        resp, _ = self._post("/api/register", {"name": "carol"})
        # Verify by listing agents
        agents, _ = self._get("/api/agents")
        agent = [a for a in agents if a["id"] == resp["agent_id"]][0]
        self.assertEqual(agent["platform"], "unknown")


# ═══════════════════════════════════════════════════════════════
# Events (Unified API) Tests
# ═══════════════════════════════════════════════════════════════


class TestHubEvents(TestServerBase):

    def test_post_event(self):
        """POST /api/events with valid type stores the event."""
        aid, key = self._register()
        resp, status = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "hello"}, "tags": ["test"]}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["event"]["type"], "CLAIM")
        self.assertEqual(resp["event"]["agent_id"], aid)

    def test_post_event_unknown_type(self):
        """POST with type=BOGUS returns 400."""
        _, key = self._register()
        try:
            self._post("/api/events", {"type": "BOGUS", "payload": {}}, key=key)
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_post_event_requires_auth(self):
        """POST without Authorization header returns 401."""
        try:
            self._post("/api/events", {"type": "CLAIM", "payload": {}})
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_post_event_invalid_key(self):
        """POST with wrong Bearer token returns 401."""
        try:
            self._post("/api/events", {"type": "CLAIM", "payload": {}}, key="rr_boguskey")
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_get_events_all(self):
        """GET /api/events returns all posted events."""
        _, key = self._register()
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "a"}}, key=key)
        self._post("/api/events", {"type": "REQUEST", "payload": {"message": "b"}}, key=key)
        events, _ = self._get("/api/events")
        self.assertGreaterEqual(len(events), 2)

    def test_get_events_filter_type(self):
        """GET with ?types=CLAIM returns only CLAIMs."""
        _, key = self._register()
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "a"}}, key=key)
        self._post("/api/events", {"type": "REQUEST", "payload": {"message": "b"}}, key=key)
        events, _ = self._get("/api/events?types=CLAIM")
        for e in events:
            self.assertEqual(e["type"], "CLAIM")

    def test_get_events_filter_type_csv(self):
        """GET with ?types=CLAIM,REQUEST returns both types."""
        _, key = self._register()
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "a"}}, key=key)
        self._post("/api/events", {"type": "REQUEST", "payload": {"message": "b"}}, key=key)
        self._post("/api/events", {"type": "FACT", "payload": {"content": "c"}}, key=key)
        events, _ = self._get("/api/events?types=CLAIM,REQUEST")
        types = {e["type"] for e in events}
        self.assertTrue(types.issubset({"CLAIM", "REQUEST"}))
        self.assertGreaterEqual(len(events), 2)

    def test_get_events_filter_agent(self):
        """GET with ?agent=<id> returns only that agent's events."""
        aid1, key1 = self._register(name="agent1")
        aid2, key2 = self._register(name="agent2")
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "a"}}, key=key1)
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "b"}}, key=key2)
        events, _ = self._get(f"/api/events?agent={aid1}")
        for e in events:
            self.assertEqual(e["agent_id"], aid1)

    def test_get_events_filter_platform(self):
        """GET with ?platform=GH200 returns only GH200 events."""
        _, key = self._register(name="gpu-agent", platform="GH200")
        _, key2 = self._register(name="cpu-agent", platform="CPU")
        self._post("/api/events", {"type": "CLAIM", "payload": {}}, key=key)
        self._post("/api/events", {"type": "CLAIM", "payload": {}}, key=key2)
        events, _ = self._get("/api/events?platform=GH200")
        for e in events:
            self.assertEqual(e["platform"], "GH200")

    def test_get_events_filter_tags(self):
        """GET with ?tags=optimizer returns only tagged events."""
        _, key = self._register()
        self._post("/api/events", {"type": "CLAIM", "payload": {}, "tags": ["optimizer"]}, key=key)
        self._post("/api/events", {"type": "CLAIM", "payload": {}, "tags": ["scheduler"]}, key=key)
        events, _ = self._get("/api/events?tags=optimizer")
        for e in events:
            self.assertIn("optimizer", e["tags"])

    def test_get_events_since_id(self):
        """GET with ?since_id=N returns only events with id > N."""
        _, key = self._register()
        r1, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "first"}}, key=key)
        first_id = r1["event"]["id"]
        self._post("/api/events", {"type": "CLAIM", "payload": {"message": "second"}}, key=key)
        events, _ = self._get(f"/api/events?since_id={first_id}")
        for e in events:
            self.assertGreater(e["id"], first_id)

    def test_get_events_pagination(self):
        """GET with ?limit=2 returns at most 2 events."""
        _, key = self._register()
        for i in range(5):
            self._post("/api/events", {"type": "CLAIM", "payload": {"message": f"msg{i}"}}, key=key)
        events, _ = self._get("/api/events?limit=2")
        self.assertLessEqual(len(events), 2)

    def test_get_events_empty_db(self):
        """GET on empty database returns empty list."""
        events, _ = self._get("/api/events")
        self.assertEqual(events, [])


# ═══════════════════════════════════════════════════════════════
# Results Tests
# ═══════════════════════════════════════════════════════════════


class TestHubResults(TestServerBase):

    def test_post_result(self):
        """POST /api/results with score and description succeeds."""
        _, key = self._register()
        resp, status = self._post("/api/results", {"score": 1.05, "status": "keep", "description": "test run"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_result_keep(self):
        """POST with status=keep appears in GET /api/results."""
        _, key = self._register()
        self._post("/api/results", {"score": 1.05, "status": "keep", "description": "kept result"}, key=key)
        results, _ = self._get("/api/results")
        self.assertTrue(any(r["description"] == "kept result" for r in results))

    def test_post_result_discard(self):
        """POST with status=discard appears when filtered."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "discard", "description": "bad run"}, key=key)
        results, _ = self._get("/api/results?status=discard")
        self.assertTrue(any(r["description"] == "bad run" for r in results))

    def test_get_results_by_agent(self):
        """GET /api/results?agent=<id> returns only that agent's results."""
        aid1, key1 = self._register(name="agent1")
        aid2, key2 = self._register(name="agent2")
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "a1"}, key=key1)
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "a2"}, key=key2)
        results, _ = self._get(f"/api/results?agent={aid1}")
        for r in results:
            self.assertEqual(r["agent_id"], aid1)

    def test_leaderboard_ranking(self):
        """Leaderboard returns agents sorted by best_score ascending."""
        _, key1 = self._register(name="fast")
        _, key2 = self._register(name="slow")
        self._post("/api/results", {"score": 0.5, "status": "keep", "description": "good"}, key=key1)
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "bad"}, key=key2)
        lb, _ = self._get("/api/results/leaderboard")
        self.assertGreaterEqual(len(lb), 2)
        self.assertLessEqual(lb[0]["best_score"], lb[1]["best_score"])

    def test_leaderboard_ignores_discard(self):
        """Leaderboard best_score only counts keep results."""
        _, key = self._register()
        self._post("/api/results", {"score": 0.1, "status": "discard", "description": "discarded low"}, key=key)
        self._post("/api/results", {"score": 5.0, "status": "keep", "description": "kept high"}, key=key)
        lb, _ = self._get("/api/results/leaderboard")
        self.assertEqual(len(lb), 1)
        self.assertEqual(lb[0]["best_score"], 5.0)

    def test_leaderboard_platform_filter(self):
        """Leaderboard with ?platform=X shows only that platform."""
        _, key1 = self._register(name="gh", platform="GH200")
        _, key2 = self._register(name="cpu", platform="CPU")
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "gh run"}, key=key1)
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "cpu run"}, key=key2)
        lb, _ = self._get("/api/results/leaderboard?platform=GH200")
        for entry in lb:
            self.assertEqual(entry["platform"], "GH200")

    def test_leaderboard_hit_rate(self):
        """hit_rate = keeps / total experiments (not always 100%)."""
        _, key = self._register()
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "k1"}, key=key)
        self._post("/api/results", {"score": 1.1, "status": "keep", "description": "k2"}, key=key)
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": "d1"}, key=key)
        lb, _ = self._get("/api/results/leaderboard")
        self.assertEqual(len(lb), 1)
        # 2 keeps out of 3 experiments = 66%
        self.assertEqual(lb[0]["hit_rate"], "66%")

    def test_leaderboard_empty(self):
        """Leaderboard on empty DB returns empty list."""
        lb, _ = self._get("/api/results/leaderboard")
        self.assertEqual(lb, [])


# ═══════════════════════════════════════════════════════════════
# Blackboard Tests
# ═══════════════════════════════════════════════════════════════


class TestHubBlackboard(TestServerBase):

    def test_post_claim(self):
        """POST /api/blackboard type=CLAIM stores the claim."""
        _, key = self._register()
        resp, status = self._post("/api/blackboard", {"type": "CLAIM", "message": "LR=0.01 is optimal"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_request(self):
        """POST type=REQUEST with target preserves target field."""
        aid, key = self._register()
        aid2, _ = self._register(name="other")
        self._post("/api/blackboard", {"type": "REQUEST", "message": "try batch=256", "target": aid2}, key=key)
        bb, _ = self._get("/api/blackboard?type=REQUEST")
        self.assertTrue(any(b["target"] == aid2 for b in bb))

    def test_post_response_in_reply_to(self):
        """POST type=RESPONSE with in_reply_to links to original."""
        _, key = self._register()
        resp1, _ = self._post("/api/blackboard", {"type": "CLAIM", "message": "hypothesis A"}, key=key)
        claim_id = resp1["event_id"]
        self._post("/api/blackboard", {"type": "RESPONSE", "message": "agreed", "in_reply_to": claim_id}, key=key)
        bb, _ = self._get("/api/blackboard?type=RESPONSE")
        self.assertTrue(any(b["in_reply_to"] == claim_id for b in bb))

    def test_post_refute(self):
        """POST type=REFUTE stores correctly."""
        _, key = self._register()
        resp, status = self._post("/api/blackboard", {"type": "REFUTE", "message": "wrong conclusion"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_blackboard_invalid_type(self):
        """POST type=RESULT returns 400 (not a blackboard type)."""
        _, key = self._register()
        try:
            self._post("/api/blackboard", {"type": "RESULT", "message": "test"}, key=key)
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_get_blackboard_all(self):
        """GET /api/blackboard returns all blackboard types."""
        _, key = self._register()
        self._post("/api/blackboard", {"type": "CLAIM", "message": "c1"}, key=key)
        self._post("/api/blackboard", {"type": "REQUEST", "message": "r1"}, key=key)
        self._post("/api/blackboard", {"type": "REFUTE", "message": "f1"}, key=key)
        bb, _ = self._get("/api/blackboard")
        types = {b["type"] for b in bb}
        self.assertTrue(types.issuperset({"CLAIM", "REQUEST", "REFUTE"}))

    def test_get_blackboard_filter_type(self):
        """GET ?type=CLAIM returns only claims."""
        _, key = self._register()
        self._post("/api/blackboard", {"type": "CLAIM", "message": "c1"}, key=key)
        self._post("/api/blackboard", {"type": "REQUEST", "message": "r1"}, key=key)
        bb, _ = self._get("/api/blackboard?type=CLAIM")
        for b in bb:
            self.assertEqual(b["type"], "CLAIM")

    def test_get_blackboard_threading(self):
        """RESPONSE's in_reply_to matches parent CLAIM id."""
        _, key = self._register()
        resp1, _ = self._post("/api/blackboard", {"type": "CLAIM", "message": "parent"}, key=key)
        parent_id = resp1["event_id"]
        self._post("/api/blackboard", {"type": "RESPONSE", "message": "child", "in_reply_to": parent_id}, key=key)
        bb, _ = self._get("/api/blackboard?type=RESPONSE")
        child = [b for b in bb if b["message"] == "child"][0]
        self.assertEqual(child["in_reply_to"], parent_id)

    def test_get_blackboard_since_id(self):
        """GET ?since_id=N returns only newer messages."""
        _, key = self._register()
        resp1, _ = self._post("/api/blackboard", {"type": "CLAIM", "message": "old"}, key=key)
        old_id = resp1["event_id"]
        self._post("/api/blackboard", {"type": "CLAIM", "message": "new"}, key=key)
        bb, _ = self._get(f"/api/blackboard?since_id={old_id}")
        for b in bb:
            self.assertGreater(b["id"], old_id)


# ═══════════════════════════════════════════════════════════════
# Memory Tests
# ═══════════════════════════════════════════════════════════════


class TestHubMemory(TestServerBase):

    def test_post_fact(self):
        """POST /api/memory type=fact stores as FACT event."""
        _, key = self._register()
        resp, status = self._post("/api/memory", {"type": "fact", "content": "LR=0.01 works"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_failure(self):
        """POST type=failure stores as FAILURE event."""
        _, key = self._register()
        resp, status = self._post("/api/memory", {"type": "failure", "content": "OOM at batch=1024"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_hunch(self):
        """POST type=hunch stores as HUNCH event."""
        _, key = self._register()
        resp, status = self._post("/api/memory", {"type": "hunch", "content": "cosine schedule might help"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])

    def test_post_memory_invalid_type(self):
        """POST type=bogus returns 400."""
        _, key = self._register()
        try:
            self._post("/api/memory", {"type": "bogus", "content": "test"}, key=key)
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_get_memory_all(self):
        """GET /api/memory returns facts, failures, and hunches."""
        _, key = self._register()
        self._post("/api/memory", {"type": "fact", "content": "f1"}, key=key)
        self._post("/api/memory", {"type": "failure", "content": "f2"}, key=key)
        self._post("/api/memory", {"type": "hunch", "content": "h1"}, key=key)
        mem, _ = self._get("/api/memory")
        types = {m["type"] for m in mem}
        self.assertTrue(types.issuperset({"fact", "failure", "hunch"}))

    def test_get_memory_filter_type(self):
        """GET ?type=failure returns only failures."""
        _, key = self._register()
        self._post("/api/memory", {"type": "fact", "content": "f1"}, key=key)
        self._post("/api/memory", {"type": "failure", "content": "f2"}, key=key)
        mem, _ = self._get("/api/memory?type=failure")
        for m in mem:
            self.assertEqual(m["type"], "failure")

    def test_memory_content_preserved(self):
        """Content string round-trips exactly."""
        _, key = self._register()
        content = "LR=0.01, batch=256, warmup=100 steps"
        self._post("/api/memory", {"type": "fact", "content": content}, key=key)
        mem, _ = self._get("/api/memory?type=fact")
        self.assertTrue(any(m["content"] == content for m in mem))


# ═══════════════════════════════════════════════════════════════
# Commits Tests
# ═══════════════════════════════════════════════════════════════


class TestHubCommits(TestServerBase):

    def test_post_commit(self):
        """POST /api/commits with hash and message succeeds."""
        pass

    def test_get_commits(self):
        """GET /api/commits returns posted commits."""
        pass

    def test_get_commits_by_agent(self):
        """GET ?agent=<id> returns only that agent's commits."""
        pass

    def test_commit_fields(self):
        """All fields (parent, memory_gb, status, score) preserved."""
        pass


# ═══════════════════════════════════════════════════════════════
# Posts (Channels) Tests
# ═══════════════════════════════════════════════════════════════


class TestHubPosts(TestServerBase):

    def test_post_to_channel(self):
        """POST /api/posts to channel=results succeeds."""
        pass

    def test_get_posts(self):
        """GET /api/posts returns posted content."""
        pass

    def test_get_posts_filter_channel(self):
        """GET ?channel=results returns only that channel."""
        pass

    def test_get_posts_since_id(self):
        """GET ?since_id=N returns only newer posts."""
        pass


# ═══════════════════════════════════════════════════════════════
# Operator Tests
# ═══════════════════════════════════════════════════════════════


class TestHubOperator(TestServerBase):

    def test_operator_strategy(self):
        """POST /api/operator/strategy creates OPERATOR event."""
        resp, status = self._post("/api/operator/strategy", {"content": "Phase 2: exploit"})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        events, _ = self._get("/api/events?types=OPERATOR")
        strategy = [e for e in events if e["agent_id"] == "OPERATOR" and "STRATEGY" in e["payload"].get("message", "")]
        self.assertTrue(len(strategy) >= 1)

    def test_operator_ban(self):
        """POST /api/operator/ban creates FAILURE with [OPERATOR BAN] prefix."""
        resp, status = self._post("/api/operator/ban", {"content": "depth 12"})
        self.assertEqual(status, 200)
        events, _ = self._get("/api/events?types=FAILURE")
        bans = [e for e in events if e["agent_id"] == "OPERATOR" and "[OPERATOR BAN]" in e["payload"].get("content", "")]
        self.assertTrue(len(bans) >= 1)
        self.assertIn("depth 12", bans[0]["payload"]["content"])

    def test_operator_directive(self):
        """POST /api/operator/directive with target creates OPERATOR event."""
        aid, _ = self._register(name="target-agent")
        resp, status = self._post("/api/operator/directive", {"message": "focus LR", "target": aid})
        self.assertEqual(status, 200)
        events, _ = self._get("/api/events?types=OPERATOR")
        directives = [e for e in events if e["agent_id"] == "OPERATOR" and e["payload"].get("subtype") == "directive"]
        self.assertTrue(len(directives) >= 1)
        self.assertEqual(directives[0]["payload"]["target"], aid)

    def test_operator_claim(self):
        """POST /api/operator/claim creates OPERATOR event with subtype=claim."""
        resp, status = self._post("/api/operator/claim", {"message": "batch=2^17 better"})
        self.assertEqual(status, 200)
        events, _ = self._get("/api/events?types=OPERATOR")
        claims = [e for e in events if e["agent_id"] == "OPERATOR" and e["payload"].get("subtype") == "claim"]
        self.assertTrue(len(claims) >= 1)
        self.assertEqual(claims[0]["payload"]["message"], "batch=2^17 better")

    def test_operator_no_auth_required(self):
        """Operator endpoints work without Bearer token."""
        # All operator endpoints should work without auth headers
        resp, status = self._post("/api/operator/strategy", {"content": "no auth needed"})
        self.assertEqual(status, 200)
        resp, status = self._post("/api/operator/ban", {"content": "banned"})
        self.assertEqual(status, 200)
        resp, status = self._post("/api/operator/directive", {"message": "do this"})
        self.assertEqual(status, 200)
        resp, status = self._post("/api/operator/claim", {"message": "claim this"})
        self.assertEqual(status, 200)


# ═══════════════════════════════════════════════════════════════
# Reactions Tests
# ═══════════════════════════════════════════════════════════════


class TestHubReactions(TestServerBase):

    def test_confirm_event(self):
        """POST /api/events/{id}/confirm creates CONFIRM event."""
        _, key = self._register()
        r, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "test claim"}}, key=key)
        event_id = r["event"]["id"]
        resp, status = self._post(f"/api/events/{event_id}/confirm", {"reason": "looks correct"}, key=key)
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["event"]["type"], "CONFIRM")

    def test_contradict_event(self):
        """POST /api/events/{id}/contradict creates CONTRADICT event."""
        _, key = self._register()
        r, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "bad claim"}}, key=key)
        event_id = r["event"]["id"]
        resp, status = self._post(f"/api/events/{event_id}/contradict", {"reason": "wrong"}, key=key)
        self.assertEqual(status, 200)
        self.assertEqual(resp["event"]["type"], "CONTRADICT")

    def test_adopt_event(self):
        """POST /api/events/{id}/adopt creates ADOPT event."""
        _, key = self._register()
        r, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "good idea"}}, key=key)
        event_id = r["event"]["id"]
        resp, status = self._post(f"/api/events/{event_id}/adopt", {"reason": "trying this"}, key=key)
        self.assertEqual(status, 200)
        self.assertEqual(resp["event"]["type"], "ADOPT")

    def test_react_to_nonexistent(self):
        """POST confirm on nonexistent event returns 404."""
        _, key = self._register()
        try:
            self._post("/api/events/99999/confirm", {"reason": "test"}, key=key)
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_reaction_with_reason(self):
        """Reason text preserved in reaction payload."""
        _, key = self._register()
        r, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "test"}}, key=key)
        event_id = r["event"]["id"]
        resp, _ = self._post(f"/api/events/{event_id}/confirm", {"reason": "verified independently"}, key=key)
        self.assertEqual(resp["event"]["payload"]["reason"], "verified independently")

    def test_reaction_counts_on_results(self):
        """Multiple confirms counted correctly in dashboard queries."""
        _, key1 = self._register(name="a1")
        _, key2 = self._register(name="a2")
        _, key3 = self._register(name="a3")
        r, _ = self._post("/api/events", {"type": "CLAIM", "payload": {"message": "shared finding"}}, key=key1)
        event_id = r["event"]["id"]
        self._post(f"/api/events/{event_id}/confirm", {"reason": "agree"}, key=key2)
        self._post(f"/api/events/{event_id}/confirm", {"reason": "me too"}, key=key3)
        # Verify by checking CONFIRM events referencing this event
        events, _ = self._get("/api/events?types=CONFIRM")
        confirms = [e for e in events if e["reply_to"] == event_id]
        self.assertEqual(len(confirms), 2)


# ═══════════════════════════════════════════════════════════════
# Verification Tests
# ═══════════════════════════════════════════════════════════════


class TestHubVerification(TestServerBase):

    def test_verify_queue_populated(self):
        """New-best result triggers VERIFY request in queue."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "new best"}, key=key)
        queue, _ = self._get("/api/verify/queue")
        self.assertTrue(len(queue) >= 1)
        self.assertIn("original_score", queue[0])

    def test_verify_queue_empty(self):
        """Fresh database has empty verify queue."""
        queue, _ = self._get("/api/verify/queue")
        self.assertEqual(queue, [])

    def test_post_verification_confirmed(self):
        """POST /api/verify with confirmed creates VERIFY result + auto CONFIRM."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "new best"}, key=key)
        queue, _ = self._get("/api/verify/queue")
        vr_id = queue[0]["id"]
        _, key2 = self._register(name="verifier")
        resp, status = self._post(
            f"/api/verify?verify_request_id={vr_id}&reproduced_score=1.05&verdict=confirmed&notes=reproduced",
            {},
            key=key2,
        )
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        # Check auto CONFIRM was created
        events, _ = self._get("/api/events?types=CONFIRM")
        self.assertTrue(len(events) >= 1)

    def test_post_verification_contradicted(self):
        """POST /api/verify with contradicted creates VERIFY result + auto CONTRADICT."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "new best"}, key=key)
        queue, _ = self._get("/api/verify/queue")
        vr_id = queue[0]["id"]
        _, key2 = self._register(name="verifier")
        resp, status = self._post(
            f"/api/verify?verify_request_id={vr_id}&reproduced_score=3.0&verdict=contradicted&notes=could+not+reproduce",
            {},
            key=key2,
        )
        self.assertEqual(status, 200)
        events, _ = self._get("/api/events?types=CONTRADICT")
        self.assertTrue(len(events) >= 1)

    def test_verify_nonexistent_request(self):
        """POST with bad verify_request_id returns 404."""
        _, key = self._register()
        try:
            self._post(
                "/api/verify?verify_request_id=99999&reproduced_score=1.0&verdict=confirmed&notes=test",
                {},
                key=key,
            )
            self.fail("Expected HTTPError")
        except HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_verify_queue_platform_filter(self):
        """Queue filtered by platform shows only matching entries."""
        _, key1 = self._register(name="gh", platform="GH200")
        _, key2 = self._register(name="a100", platform="A100")
        # Create verification requests on both platforms
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "gh baseline"}, key=key1)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "gh best"}, key=key1)
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "a100 baseline"}, key=key2)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "a100 best"}, key=key2)
        queue, _ = self._get("/api/verify/queue?platform=GH200")
        for entry in queue:
            self.assertEqual(entry["platform"], "GH200")

    def test_verify_shows_verifications(self):
        """Queue entry shows verified=True after posting verification."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "new best"}, key=key)
        queue, _ = self._get("/api/verify/queue")
        vr_id = queue[0]["id"]
        self.assertFalse(queue[0]["verified"])
        _, key2 = self._register(name="verifier")
        self._post(
            f"/api/verify?verify_request_id={vr_id}&reproduced_score=1.05&verdict=confirmed&notes=ok",
            {},
            key=key2,
        )
        queue2, _ = self._get("/api/verify/queue")
        entry = [q for q in queue2 if q["id"] == vr_id][0]
        self.assertTrue(entry["verified"])


# ═══════════════════════════════════════════════════════════════
# HAI Cards Tests
# ═══════════════════════════════════════════════════════════════


class TestHubHAICards(TestServerBase):

    def test_hai_card_empty(self):
        """GET /api/hai-card on fresh db returns card with zero counts."""
        pass

    def test_hai_card_autonomy_level_a(self):
        """No operator events = autonomy level A."""
        pass

    def test_hai_card_autonomy_level_h(self):
        """Many operator events = autonomy level H."""
        pass

    def test_hai_card_autonomy_level_c(self):
        """Both operator and agent events = level C."""
        pass

    def test_hai_card_agent_filter(self):
        """Filter by agent_id shows only that agent's data."""
        pass

    def test_hai_card_markdown(self):
        """GET /api/hai-card/markdown returns markdown string."""
        pass

    def test_hai_card_best_result(self):
        """best_result shows the lowest score."""
        pass


# ═══════════════════════════════════════════════════════════════
# SSE Stream Tests
# ═══════════════════════════════════════════════════════════════


class TestHubSSE(TestServerBase):

    def test_sse_connection(self):
        """GET /api/stream returns text/event-stream content type."""
        pass

    def test_sse_receives_event(self):
        """Stream yields event posted in another thread."""
        pass

    def test_sse_type_filter(self):
        """Stream with ?types=CLAIM only yields CLAIMs."""
        pass

    def test_sse_since_id(self):
        """Stream with ?since_id=N skips older events."""
        pass

    def test_sse_event_format(self):
        """SSE lines include id:, event:, and data: fields."""
        pass


# ═══════════════════════════════════════════════════════════════
# Playbook Tests
# ═══════════════════════════════════════════════════════════════


class TestHubPlaybooks(TestServerBase):

    # Dead End Detector
    def test_dead_end_detector_triggers(self):
        """2 agents discard same config prefix -> FAILURE auto-created."""
        _, key1 = self._register(name="a1")
        _, key2 = self._register(name="a2")
        desc = "learning_rate=0.1 with cosine schedule failed badly"
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": desc}, key=key1)
        self._post("/api/results", {"score": 6.0, "status": "discard", "description": desc}, key=key2)
        # Check for PLAYBOOK FAILURE event
        events, _ = self._get("/api/events?types=FAILURE")
        playbook_events = [e for e in events if e["agent_id"] == "PLAYBOOK"]
        self.assertTrue(len(playbook_events) >= 1)
        self.assertIn("Dead end", playbook_events[0]["payload"].get("content", ""))

    def test_dead_end_detector_no_trigger(self):
        """1 agent discard -> no PLAYBOOK event."""
        _, key = self._register()
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": "some long description here"}, key=key)
        events, _ = self._get("/api/events?types=FAILURE")
        playbook_events = [e for e in events if e["agent_id"] == "PLAYBOOK" and "Dead end" in e["payload"].get("content", "")]
        self.assertEqual(len(playbook_events), 0)

    def test_dead_end_detector_short_desc(self):
        """Discard with <5 char description -> no trigger."""
        _, key1 = self._register(name="a1")
        _, key2 = self._register(name="a2")
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": "ab"}, key=key1)
        self._post("/api/results", {"score": 6.0, "status": "discard", "description": "ab"}, key=key2)
        events, _ = self._get("/api/events?types=FAILURE")
        playbook_events = [e for e in events if e["agent_id"] == "PLAYBOOK" and "Dead end" in e["payload"].get("content", "")]
        self.assertEqual(len(playbook_events), 0)

    # Convergence Signal
    def test_convergence_signal_triggers(self):
        """3 agents' bests within 1% -> OPERATOR convergence event."""
        _, key1 = self._register(name="c1")
        _, key2 = self._register(name="c2")
        _, key3 = self._register(name="c3")
        # Post keeps with very close scores
        self._post("/api/results", {"score": 1.000, "status": "keep", "description": "run1"}, key=key1)
        self._post("/api/results", {"score": 1.005, "status": "keep", "description": "run2"}, key=key2)
        self._post("/api/results", {"score": 1.009, "status": "keep", "description": "run3"}, key=key3)
        events, _ = self._get("/api/events?types=OPERATOR")
        convergence = [e for e in events if e["agent_id"] == "PLAYBOOK" and "CONVERGENCE" in e["payload"].get("message", "")]
        self.assertTrue(len(convergence) >= 1)

    def test_convergence_signal_no_trigger(self):
        """Only 2 agents -> no convergence signal."""
        _, key1 = self._register(name="c1")
        _, key2 = self._register(name="c2")
        self._post("/api/results", {"score": 1.000, "status": "keep", "description": "run1"}, key=key1)
        self._post("/api/results", {"score": 1.005, "status": "keep", "description": "run2"}, key=key2)
        events, _ = self._get("/api/events?types=OPERATOR")
        convergence = [e for e in events if e["agent_id"] == "PLAYBOOK" and "CONVERGENCE" in e["payload"].get("message", "")]
        self.assertEqual(len(convergence), 0)

    def test_convergence_dedup(self):
        """Second trigger within 1 hour is suppressed."""
        _, key1 = self._register(name="c1")
        _, key2 = self._register(name="c2")
        _, key3 = self._register(name="c3")
        # First trigger
        self._post("/api/results", {"score": 1.000, "status": "keep", "description": "run1"}, key=key1)
        self._post("/api/results", {"score": 1.005, "status": "keep", "description": "run2"}, key=key2)
        self._post("/api/results", {"score": 1.009, "status": "keep", "description": "run3"}, key=key3)
        # Second trigger attempt (post another close result)
        _, key4 = self._register(name="c4")
        self._post("/api/results", {"score": 1.002, "status": "keep", "description": "run4"}, key=key4)
        events, _ = self._get("/api/events?types=OPERATOR")
        convergence = [e for e in events if e["agent_id"] == "PLAYBOOK" and "CONVERGENCE" in e["payload"].get("message", "")]
        # Should be exactly 1 (dedup suppresses the second)
        self.assertEqual(len(convergence), 1)

    # Platform Mismatch
    def test_platform_mismatch_warns(self):
        """Results from 2 platforms -> OPERATOR platform-mismatch."""
        _, key1 = self._register(name="gh", platform="GH200")
        _, key2 = self._register(name="a100", platform="A100")
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "gh run"}, key=key1)
        self._post("/api/results", {"score": 1.1, "status": "keep", "description": "a100 run"}, key=key2)
        events, _ = self._get("/api/events?types=OPERATOR")
        mismatch = [e for e in events if e["agent_id"] == "PLAYBOOK" and "PLATFORM" in e["payload"].get("message", "")]
        self.assertTrue(len(mismatch) >= 1)

    def test_platform_mismatch_single_platform(self):
        """All results same platform -> no warning."""
        _, key1 = self._register(name="a1", platform="GH200")
        _, key2 = self._register(name="a2", platform="GH200")
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "run1"}, key=key1)
        self._post("/api/results", {"score": 1.1, "status": "keep", "description": "run2"}, key=key2)
        events, _ = self._get("/api/events?types=OPERATOR")
        mismatch = [e for e in events if e["agent_id"] == "PLAYBOOK" and "PLATFORM" in e["payload"].get("message", "")]
        self.assertEqual(len(mismatch), 0)

    # Verification Request
    def test_verification_request_new_best(self):
        """New best score -> VERIFY request event."""
        _, key = self._register()
        self._post("/api/results", {"score": 2.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "new best"}, key=key)
        events, _ = self._get("/api/events?types=VERIFY")
        verify_requests = [e for e in events if e["agent_id"] == "PLAYBOOK" and e["payload"].get("subtype") == "request"]
        self.assertTrue(len(verify_requests) >= 1)

    def test_verification_request_first_result(self):
        """First result ever -> no verification request."""
        _, key = self._register()
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "first"}, key=key)
        events, _ = self._get("/api/events?types=VERIFY")
        verify_requests = [e for e in events if e["agent_id"] == "PLAYBOOK" and e["payload"].get("subtype") == "request"]
        self.assertEqual(len(verify_requests), 0)

    def test_verification_request_worse_result(self):
        """Worse result -> no verification request."""
        _, key = self._register()
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "baseline"}, key=key)
        self._post("/api/results", {"score": 3.0, "status": "keep", "description": "worse"}, key=key)
        events, _ = self._get("/api/events?types=VERIFY")
        verify_requests = [e for e in events if e["agent_id"] == "PLAYBOOK" and e["payload"].get("subtype") == "request"]
        self.assertEqual(len(verify_requests), 0)

    # Revision Prompt
    def test_revision_prompt_on_discard(self):
        """Discard with description -> HUNCH revision event."""
        _, key = self._register()
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": "cosine schedule with warmup"}, key=key)
        events, _ = self._get("/api/events?types=HUNCH")
        revision = [e for e in events if e["agent_id"] == "PLAYBOOK" and "REVISE" in e["payload"].get("content", "")]
        self.assertTrue(len(revision) >= 1)

    def test_revision_prompt_on_crash(self):
        """Crash result -> HUNCH revision event."""
        _, key = self._register()
        self._post("/api/results", {"score": None, "status": "crash", "description": "OOM at batch 2048 with large model"}, key=key)
        events, _ = self._get("/api/events?types=HUNCH")
        revision = [e for e in events if e["agent_id"] == "PLAYBOOK" and "REVISE" in e["payload"].get("content", "")]
        self.assertTrue(len(revision) >= 1)

    def test_revision_prompt_on_keep(self):
        """Keep result -> no revision prompt."""
        _, key = self._register()
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": "good config works well"}, key=key)
        events, _ = self._get("/api/events?types=HUNCH")
        revision = [e for e in events if e["agent_id"] == "PLAYBOOK" and "REVISE" in e["payload"].get("content", "")]
        self.assertEqual(len(revision), 0)

    def test_revision_prompt_short_desc(self):
        """Discard with <5 char description -> no trigger."""
        _, key = self._register()
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": "ab"}, key=key)
        events, _ = self._get("/api/events?types=HUNCH")
        revision = [e for e in events if e["agent_id"] == "PLAYBOOK" and "REVISE" in e["payload"].get("content", "")]
        self.assertEqual(len(revision), 0)

    # General
    def test_playbook_events_have_playbook_agent(self):
        """Auto-generated events have agent_id='PLAYBOOK'."""
        _, key1 = self._register(name="a1")
        _, key2 = self._register(name="a2")
        desc = "this is a long enough description for dead end"
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": desc}, key=key1)
        self._post("/api/results", {"score": 6.0, "status": "discard", "description": desc}, key=key2)
        events, _ = self._get("/api/events")
        playbook_events = [e for e in events if e["agent_id"] == "PLAYBOOK"]
        self.assertTrue(len(playbook_events) >= 1)
        for e in playbook_events:
            self.assertEqual(e["agent_id"], "PLAYBOOK")

    def test_playbooks_non_recursive(self):
        """Events from PLAYBOOK agent do not trigger more playbooks."""
        _, key1 = self._register(name="a1")
        _, key2 = self._register(name="a2")
        desc = "recursive test description that is long enough"
        self._post("/api/results", {"score": 5.0, "status": "discard", "description": desc}, key=key1)
        self._post("/api/results", {"score": 6.0, "status": "discard", "description": desc}, key=key2)
        # PLAYBOOK events should not trigger further PLAYBOOK events
        events, _ = self._get("/api/events")
        playbook_events = [e for e in events if e["agent_id"] == "PLAYBOOK"]
        # Each playbook fires at most once per trigger; no chain reactions
        # The dead-end detector should fire once, revision-prompt fires once per discard
        # But no PLAYBOOK event should have triggered another PLAYBOOK event
        for pe in playbook_events:
            # No PLAYBOOK event should reply_to another PLAYBOOK event
            if pe.get("reply_to"):
                parent = [e for e in events if e["id"] == pe["reply_to"]]
                if parent:
                    self.assertNotEqual(parent[0]["agent_id"], "PLAYBOOK")


# ═══════════════════════════════════════════════════════════════
# Agent Endpoint Tests
# ═══════════════════════════════════════════════════════════════


class TestHubAgents(TestServerBase):

    def test_list_agents(self):
        """GET /api/agents returns registered agents with stats."""
        pass

    def test_get_agent_detail(self):
        """GET /api/agents/{id} returns agent with stats dict."""
        pass

    def test_get_agent_not_found(self):
        """GET /api/agents/bogus returns 404."""
        pass

    def test_agent_last_seen_updates(self):
        """Posting an event updates agent's last_seen."""
        pass


# ═══════════════════════════════════════════════════════════════
# Edge Cases & Regression Tests
# ═══════════════════════════════════════════════════════════════


class TestHubEdgeCases(TestServerBase):

    # Regression: hit_rate was always 100%
    def test_hit_rate_mixed_keep_discard(self):
        """3 keep + 2 discard = 60% hit_rate (not 100%)."""
        _, key = self._register()
        for i in range(3):
            self._post("/api/results", {"score": 1.0 + i * 0.01, "status": "keep", "description": f"keep{i}"}, key=key)
        for i in range(2):
            self._post("/api/results", {"score": 5.0 + i, "status": "discard", "description": f"discard{i}"}, key=key)
        lb, _ = self._get("/api/results/leaderboard")
        self.assertEqual(len(lb), 1)
        self.assertEqual(lb[0]["hit_rate"], "60%")

    # Regression: convergence dedup used wrong field
    def test_convergence_dedup_query_correct(self):
        """Convergence dedup queries agent_id='PLAYBOOK', not type='PLAYBOOK'."""
        # Trigger convergence signal
        _, key1 = self._register(name="c1")
        _, key2 = self._register(name="c2")
        _, key3 = self._register(name="c3")
        self._post("/api/results", {"score": 1.000, "status": "keep", "description": "r1"}, key=key1)
        self._post("/api/results", {"score": 1.005, "status": "keep", "description": "r2"}, key=key2)
        self._post("/api/results", {"score": 1.009, "status": "keep", "description": "r3"}, key=key3)
        # Verify the convergence event has agent_id='PLAYBOOK'
        events, _ = self._get("/api/events?types=OPERATOR")
        convergence = [e for e in events if e["agent_id"] == "PLAYBOOK" and "CONVERGENCE" in e["payload"].get("message", "")]
        self.assertTrue(len(convergence) >= 1)
        self.assertEqual(convergence[0]["agent_id"], "PLAYBOOK")

    # Concurrency
    def test_concurrent_registration(self):
        """10 threads register simultaneously, all succeed."""
        pass

    def test_concurrent_result_posting(self):
        """Multiple agents post results in parallel."""
        pass

    # Boundary conditions
    def test_empty_db_all_endpoints(self):
        """GET every list endpoint on fresh db returns 200."""
        endpoints = [
            "/api/events",
            "/api/results",
            "/api/results/leaderboard",
            "/api/blackboard",
            "/api/memory",
            "/api/commits",
            "/api/posts",
            "/api/agents",
            "/api/verify/queue",
        ]
        for ep in endpoints:
            resp, status = self._get(ep)
            self.assertEqual(status, 200, f"{ep} returned {status}")

    def test_large_payload(self):
        """Post 1000 events, verify all stored."""
        pass

    def test_special_chars_in_description(self):
        """Descriptions with quotes, angle brackets, unicode survive."""
        _, key = self._register()
        desc = 'LR=0.01 "quoted" <html> & unicode: \u00e9\u00e0\u00fc \U0001f600'
        self._post("/api/results", {"score": 1.0, "status": "keep", "description": desc}, key=key)
        results, _ = self._get("/api/results")
        self.assertTrue(any(r["description"] == desc for r in results))

    def test_null_score(self):
        """Result with score=None stored but not on leaderboard."""
        pass

    def test_platform_filter_no_match(self):
        """Leaderboard with nonexistent platform returns empty list."""
        pass


# ═══════════════════════════════════════════════════════════════
# Dashboard Tests
# ═══════════════════════════════════════════════════════════════


class TestHubDashboard(TestServerBase):

    def test_dashboard_returns_html(self):
        """GET /dashboard returns 200 with HTML content."""
        url = f"{TEST_URL}/dashboard"
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("content-type", "")
            self.assertIn("text/html", content_type)

    def test_dashboard_empty_db(self):
        """Dashboard on fresh db renders without error."""
        url = f"{TEST_URL}/dashboard"
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            body = resp.read().decode()
            self.assertIn("researchRalph Hub", body)

    def test_dashboard_with_data(self):
        """Dashboard after posting data includes agent names and scores."""
        _, key = self._register(name="dashboard-agent")
        self._post("/api/results", {"score": 1.234, "status": "keep", "description": "dashboard test"}, key=key)
        url = f"{TEST_URL}/dashboard"
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            self.assertIn("dashboard-agent", body)
            self.assertIn("1.234", body)

    def test_sse_endpoint_content_type(self):
        """GET /api/stream returns text/event-stream."""
        url = f"{TEST_URL}/api/stream"
        req = Request(url)
        with urlopen(req, timeout=3) as resp:
            content_type = resp.headers.get("content-type", "")
            self.assertIn("text/event-stream", content_type)


if __name__ == "__main__":
    unittest.main()

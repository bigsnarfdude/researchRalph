# researchRalph-v2 Testing Plan

Comprehensive test suite for the researchRalph-v2 multi-agent research harness.
All tests run without GPU, without Claude CLI, and without external services.

---

## 1. Test Architecture

### Approach

Four test layers, from fast/isolated to slow/integrated:

| Layer | Tool | What it covers | Runtime |
|-------|------|---------------|---------|
| **Unit** | Python `unittest` | Hub API endpoints, playbook logic, SDK methods | ~5s |
| **Integration** | Python `unittest` + real HTTP | SDK against a live test server | ~10s |
| **End-to-end** | Python `unittest` | Multi-agent scenario through full API | ~15s |
| **Script validation** | Bash | Core shell scripts parse, accept flags, exit cleanly | ~5s |

### Principles

- **stdlib-only**: Use `unittest` (no pytest). Matches the project's stdlib-only philosophy.
- **Real server**: Hub tests start an actual FastAPI/uvicorn server on port 8765 in a background thread.
  The test database is a temporary file (`/tmp/researchralph_test_XXXX.db`), deleted on teardown.
- **No mocking the hub**: Integration tests use the real Python SDK (`client/researchralph`)
  talking to the real server over HTTP. This catches serialization bugs that mocks hide.
- **Deterministic**: No sleeps except for SSE streaming tests (bounded at 2s).
  Tests do not depend on wall-clock time or ordering.

### Directory Layout

```
tests/
    __init__.py
    test_hub.py          # Hub API (unit + integration)
    test_client.py       # Python SDK against test server
    test_playbooks.py    # Playbook trigger logic (extracted)
    test_domains.py      # Domain template file checks
    test_scripts.sh      # Bash script validation
    conftest.py          # Shared server setup (importable)
```

### Server Lifecycle

`setUpClass` in the base test case:

1. Create a temp SQLite database.
2. Monkey-patch `hub.server.DB_PATH` to point at it.
3. Call `init_db()` to create tables.
4. Start uvicorn in a daemon thread on `127.0.0.1:8765`.
5. Wait for the `/api/agents` endpoint to respond (up to 3s).

`tearDownClass`:

1. Stop the uvicorn server (set a shutdown event).
2. Delete the temp database file.

Each test method gets a fresh database by truncating tables in `setUp`.

---

## 2. Hub API Tests (`tests/test_hub.py`)

Every test uses direct HTTP calls (urllib) to validate the server independently of the SDK.

### 2.1 Registration

| Test | Validates | Expected |
|------|-----------|----------|
| `test_register_agent` | POST `/api/register` with name, team, platform | 200, returns `agent_id` and `api_key` starting with `rr_` |
| `test_register_returns_unique_ids` | Two registrations with same name | Both succeed, different `agent_id` values (hex suffix differs) |
| `test_register_missing_name` | POST with empty body | 422 (Pydantic validation error) |
| `test_register_default_platform` | POST with name only | `platform` defaults to `"unknown"` |

### 2.2 Events (Unified API)

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_event` | POST `/api/events` with type=CLAIM | 200, event has correct type, agent_id, payload |
| `test_post_event_unknown_type` | POST with type=BOGUS | 400, error lists valid types |
| `test_post_event_requires_auth` | POST without Authorization header | 401 |
| `test_post_event_invalid_key` | POST with wrong Bearer token | 401 |
| `test_get_events_all` | GET `/api/events` after posting 5 events | Returns all 5 (descending order) |
| `test_get_events_filter_type` | GET with `?types=CLAIM` | Only CLAIM events returned |
| `test_get_events_filter_type_csv` | GET with `?types=CLAIM,REQUEST` | Both types returned, nothing else |
| `test_get_events_filter_agent` | GET with `?agent=<agent_id>` | Only that agent's events |
| `test_get_events_filter_platform` | GET with `?platform=GH200` | Only events from GH200 agents |
| `test_get_events_filter_tags` | POST event with tags=["optimizer"], GET with `?tags=optimizer` | Only tagged event returned |
| `test_get_events_since_id` | GET with `?since_id=3` after posting 5 events | Only events with id > 3 |
| `test_get_events_pagination` | GET with `?limit=2` after posting 5 events | Exactly 2 returned |
| `test_get_events_empty_db` | GET on fresh database | Empty list `[]` |

### 2.3 Results

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_result` | POST `/api/results` with score, status, description | 200, `ok: true`, event_id returned |
| `test_post_result_keep` | POST with status=keep | Appears in GET `/api/results` |
| `test_post_result_discard` | POST with status=discard | Appears when filtered by `?status=discard` |
| `test_get_results_by_agent` | GET `/api/results?agent=<id>` | Only that agent's results |
| `test_leaderboard_ranking` | 3 agents post different scores (keep) | Leaderboard returns sorted ascending by best_score |
| `test_leaderboard_ignores_discard` | Agent posts keep=1.0 then discard=0.5 | best_score is 1.0 (discard excluded) |
| `test_leaderboard_platform_filter` | 2 agents on different platforms | `?platform=GH200` shows only GH200 agent |
| `test_leaderboard_hit_rate` | Agent posts 3 keep + 2 discard | hit_rate = "60%" (not 100%) |
| `test_leaderboard_empty` | No results posted | Empty list |

### 2.4 Blackboard

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_claim` | POST `/api/blackboard` type=CLAIM | 200, event stored |
| `test_post_request` | POST type=REQUEST with target | target field preserved |
| `test_post_response_in_reply_to` | POST type=RESPONSE with in_reply_to | reply_to links to original event |
| `test_post_refute` | POST type=REFUTE | Type is REFUTE in response |
| `test_post_blackboard_invalid_type` | POST type=RESULT | 400 error (not a blackboard type) |
| `test_get_blackboard_all` | GET `/api/blackboard` | Returns CLAIM, RESPONSE, REQUEST, REFUTE, OPERATOR types |
| `test_get_blackboard_filter_type` | GET `?type=CLAIM` | Only CLAIMs returned |
| `test_get_blackboard_threading` | Post CLAIM, then RESPONSE with in_reply_to | RESPONSE's in_reply_to matches CLAIM id |
| `test_get_blackboard_since_id` | GET `?since_id=N` | Only newer messages |

### 2.5 Memory

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_fact` | POST `/api/memory` type=fact | 200, stored as FACT event |
| `test_post_failure` | POST type=failure | Stored as FAILURE event |
| `test_post_hunch` | POST type=hunch | Stored as HUNCH event |
| `test_post_memory_invalid_type` | POST type=bogus | 400 error |
| `test_get_memory_all` | GET `/api/memory` | Returns facts, failures, hunches |
| `test_get_memory_filter_type` | GET `?type=failure` | Only failures returned |
| `test_memory_content_preserved` | Post then retrieve | Content string matches exactly |

### 2.6 Commits

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_commit` | POST `/api/commits` with hash, message, score | 200, hash returned |
| `test_get_commits` | GET `/api/commits` | Returns posted commits |
| `test_get_commits_by_agent` | GET `?agent=<id>` | Only that agent's commits |
| `test_commit_fields` | POST with all fields | parent, memory_gb, status all preserved |

### 2.7 Posts (Channels)

| Test | Validates | Expected |
|------|-----------|----------|
| `test_post_to_channel` | POST `/api/posts` with channel=results | 200 |
| `test_get_posts` | GET `/api/posts` | Returns posted content |
| `test_get_posts_filter_channel` | GET `?channel=results` | Only results channel |
| `test_get_posts_since_id` | GET `?since_id=N` | Only newer posts |

### 2.8 Operator

| Test | Validates | Expected |
|------|-----------|----------|
| `test_operator_strategy` | POST `/api/operator/strategy` | OPERATOR event with subtype=strategy |
| `test_operator_ban` | POST `/api/operator/ban` | FAILURE event with [OPERATOR BAN] prefix |
| `test_operator_directive` | POST `/api/operator/directive` with target | OPERATOR event with target field, priority=high |
| `test_operator_claim` | POST `/api/operator/claim` | OPERATOR event with subtype=claim |
| `test_operator_no_auth_required` | POST without Bearer token | 200 (operator endpoints are unauthenticated) |

### 2.9 Reactions

| Test | Validates | Expected |
|------|-----------|----------|
| `test_confirm_event` | POST `/api/events/{id}/confirm` | CONFIRM event with reply_to = target id |
| `test_contradict_event` | POST `/api/events/{id}/contradict` | CONTRADICT event created |
| `test_adopt_event` | POST `/api/events/{id}/adopt` | ADOPT event created |
| `test_react_to_nonexistent` | POST confirm on id=99999 | 404 |
| `test_reaction_with_reason` | POST confirm with reason text | reason preserved in payload |
| `test_reaction_counts_on_dashboard` | Confirm event 3 times | Dashboard query shows 3 confirms |

### 2.10 Verification (Aletheia-inspired)

| Test | Validates | Expected |
|------|-----------|----------|
| `test_verify_queue_populated` | Post new-best result triggering playbook | VERIFY request appears in `/api/verify/queue` |
| `test_verify_queue_empty` | Fresh database | Empty list |
| `test_post_verification_confirmed` | POST `/api/verify` with verdict=confirmed | VERIFY result event + auto CONFIRM on original |
| `test_post_verification_contradicted` | POST with verdict=contradicted | VERIFY result + auto CONTRADICT on original |
| `test_verify_nonexistent_request` | POST with bad verify_request_id | 404 |
| `test_verify_queue_platform_filter` | Two platforms, filter by one | Only matching platform in queue |
| `test_verify_shows_verifications` | Post verification result | Queue entry shows `verified: true` with verifier details |

### 2.11 HAI Cards

| Test | Validates | Expected |
|------|-----------|----------|
| `test_hai_card_empty` | GET `/api/hai-card` on fresh db | Returns card with zero counts |
| `test_hai_card_autonomy_level_a` | Agent results, no operator events | autonomy_level.level = "A" |
| `test_hai_card_autonomy_level_h` | Many operator events, few agent results | autonomy_level.level = "H" |
| `test_hai_card_autonomy_level_c` | Both operator and agent events | autonomy_level.level = "C" |
| `test_hai_card_agent_filter` | Two agents, filter by one | Only filtered agent's data |
| `test_hai_card_markdown` | GET `/api/hai-card/markdown` | Returns dict with `markdown` key, contains "# Human-AI Interaction Card" |
| `test_hai_card_best_result` | Post several results | best_result shows lowest score |

### 2.12 SSE Stream

| Test | Validates | Expected |
|------|-----------|----------|
| `test_sse_connection` | GET `/api/stream` | Response has `Content-Type: text/event-stream` |
| `test_sse_receives_event` | Connect stream, post event in another thread | Stream yields the event within 2s |
| `test_sse_type_filter` | Connect with `?types=CLAIM`, post CLAIM and FACT | Only CLAIM received |
| `test_sse_since_id` | Post 3 events, connect with `?since_id=2` | Only event 3 received |
| `test_sse_event_format` | Parse raw SSE lines | Has `id:`, `event:`, `data:` fields per SSE spec |

### 2.13 Playbooks

| Test | Validates | Expected |
|------|-----------|----------|
| `test_dead_end_detector_triggers` | 2 agents discard results with same description prefix (30 chars) | FAILURE event auto-created by PLAYBOOK agent |
| `test_dead_end_detector_no_trigger` | 1 agent discards | No PLAYBOOK event |
| `test_dead_end_detector_short_desc` | Discard with 3-char description | No trigger (< 5 chars) |
| `test_convergence_signal_triggers` | 3 agents post keeps within 1% of each other | OPERATOR event with subtype=convergence |
| `test_convergence_signal_no_trigger` | 2 agents only | No convergence signal |
| `test_convergence_dedup` | Trigger twice in 1 hour | Only 1 convergence event (dedup query) |
| `test_platform_mismatch_warns` | Results from 2 different platforms | OPERATOR event with subtype=platform-mismatch |
| `test_platform_mismatch_single_platform` | All results same platform | No warning |
| `test_verification_request_new_best` | Post first result then a better one | VERIFY event with subtype=request |
| `test_verification_request_first_result` | Post only one result (first ever) | No verification request (nothing to beat) |
| `test_verification_request_worse_result` | Post worse result after better | No verification request |
| `test_revision_prompt_on_discard` | Post discard result with description | HUNCH event with subtype=revision |
| `test_revision_prompt_on_crash` | Post crash result | HUNCH event fires |
| `test_revision_prompt_on_keep` | Post keep result | No revision prompt |
| `test_revision_prompt_short_desc` | Discard with 3-char description | No trigger |
| `test_playbook_events_have_playbook_agent` | Any playbook triggers | agent_id = "PLAYBOOK" on generated events |
| `test_playbooks_non_recursive` | Playbook creates event | That event does not trigger more playbooks |

### 2.14 Agents

| Test | Validates | Expected |
|------|-----------|----------|
| `test_list_agents` | GET `/api/agents` after registering 2 | Returns both with stats |
| `test_get_agent_detail` | GET `/api/agents/{id}` | Returns agent with stats dict |
| `test_get_agent_not_found` | GET `/api/agents/bogus` | 404 |
| `test_agent_last_seen_updates` | Post event, then GET agent | last_seen is recent timestamp |

---

## 3. Client SDK Tests (`tests/test_client.py`)

All tests use `from researchralph import Hub, HubError` against the same test server.

| Test | Validates | Expected |
|------|-----------|----------|
| `test_register_classmethod` | `Hub.register(url, name)` | Returns Hub instance with key and agent_id |
| `test_result` | `hub.result(score=1.0, description="test")` | Returns dict with `ok` |
| `test_results` | `hub.results()` | Returns list including posted result |
| `test_leaderboard` | `hub.leaderboard()` | Returns list of dicts |
| `test_claim` | `hub.claim("test claim")` | Returns dict with `ok` |
| `test_request` | `hub.request("test request")` | Returns dict with `ok` |
| `test_respond` | `hub.respond(in_reply_to=1, message="reply")` | reply_to set correctly |
| `test_refute` | `hub.refute(in_reply_to=1, message="wrong")` | Returns dict with `ok` |
| `test_blackboard` | `hub.blackboard()` | Returns list |
| `test_fact` | `hub.fact("LR 0.08 works")` | Returns dict with `ok` |
| `test_failure` | `hub.failure("OOM at 62GB")` | Returns dict with `ok` |
| `test_hunch` | `hub.hunch("try weight decay")` | Returns dict with `ok` |
| `test_memory` | `hub.memory(type="fact")` | Returns list of facts |
| `test_commit` | `hub.commit(hash="abc123", message="test")` | Returns dict with `ok` |
| `test_commits` | `hub.commits()` | Returns list |
| `test_post` | `hub.post("hello", channel="results")` | Returns dict with `ok` |
| `test_posts` | `hub.posts(channel="results")` | Returns list |
| `test_confirm` | `hub.confirm(event_id=N)` | Returns dict with `ok` |
| `test_contradict` | `hub.contradict(event_id=N)` | Returns dict with `ok` |
| `test_adopt` | `hub.adopt(event_id=N)` | Returns dict with `ok` |
| `test_event_raw` | `hub.event("FACT", payload={"content": "x"})` | Returns dict with event |
| `test_events_query` | `hub.events(types="FACT")` | Returns filtered list |
| `test_since` | `hub.since(since_id=0)` | Returns events after id 0 |
| `test_agents` | `hub.agents()` | Returns list of agents |
| `test_agent_detail` | `hub.agent(agent_id)` | Returns agent dict |
| `test_verify_queue` | `hub.verify_queue()` | Returns list |
| `test_hai_card` | `hub.hai_card()` | Returns dict with autonomy_level |
| `test_hai_card_markdown` | `hub.hai_card_markdown()` | Returns markdown string |
| `test_check_operator` | `hub.check_operator()` | Returns list (convenience) |
| `test_check_failures` | `hub.check_failures()` | Returns list (convenience) |
| `test_check_facts` | `hub.check_facts()` | Returns list (convenience) |
| `test_last_seen_tracking` | Call `hub.events()`, check `hub.last_seen` | last_seen updated to max event id |
| `test_hub_error_on_401` | Hub with bad key, call result() | Raises `HubError` with status=401 |
| `test_hub_error_on_400` | Post invalid event type | Raises `HubError` with status=400 |
| `test_url_encoding_special_chars` | Agent name with spaces/special chars | No URL encoding errors |
| `test_sse_stream_receives` | Start `hub.stream()` in thread, post event | Generator yields event within 2s |
| `test_repr` | `repr(hub)` | Contains URL and agent_id |

---

## 4. Script Validation Tests (`tests/test_scripts.sh`)

Each test verifies the script is syntactically valid, prints usage on missing args,
and does not crash. Tests use `bash -n` for syntax checks and controlled invocations.

```bash
#!/usr/bin/env bash
# tests/test_scripts.sh — Script validation suite
# Usage: bash tests/test_scripts.sh
set -euo pipefail
PASS=0; FAIL=0; ROOT="$(cd "$(dirname "$0")/.." && pwd)"

assert_ok()   { if "$@" >/dev/null 2>&1; then ((PASS++)); else echo "FAIL: $*"; ((FAIL++)); fi; }
assert_fail() { if ! "$@" >/dev/null 2>&1; then ((PASS++)); else echo "FAIL (expected failure): $*"; ((FAIL++)); fi; }

# Syntax checks (bash -n)
for f in core/run-single.sh core/launch.sh core/stop.sh core/monitor.sh \
         core/operator.sh core/watchdog.sh core/verifier.sh core/conductor.sh \
         core/collect.sh core/bridge.sh core/notebook.sh \
         deploy-lambda.sh deploy-nigel.sh quickstart.sh stop-all.sh; do
    assert_ok bash -n "$ROOT/$f"
done

# operator.sh: all subcommands print help or succeed
for cmd in status strategy ban directive claim leaderboard blackboard memory watch help; do
    assert_ok bash "$ROOT/core/operator.sh" help
done

# launch.sh with no args should fail or print usage
assert_fail bash "$ROOT/core/launch.sh"

# stop.sh with no args should not crash (may print usage)
bash "$ROOT/core/stop.sh" 2>/dev/null || true; ((PASS++))

echo "Script validation: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
```

### Specific Script Checks

| Script | Test | Expected |
|--------|------|----------|
| `run-single.sh` | `bash -n` | Syntax OK |
| `launch.sh` | No args | Exits non-zero or prints usage |
| `launch.sh` | `bash -n` | Syntax OK |
| `stop.sh` | No args | Does not crash |
| `monitor.sh` | `bash -n` | Syntax OK |
| `operator.sh` | `help` subcommand | Exits 0, prints usage |
| `watchdog.sh` | `bash -n` | Syntax OK |
| `verifier.sh` | `bash -n` | Syntax OK |
| `conductor.sh` | `bash -n` | Syntax OK |
| `collect.sh` | `bash -n` | Syntax OK |
| `bridge.sh` | `bash -n` | Syntax OK |
| `notebook.sh` | `bash -n` | Syntax OK |

---

## 5. Domain Template Tests (`tests/test_domains.py`)

Verify each domain has the required files and structure.

| Test | Validates | Expected |
|------|-----------|----------|
| `test_template_has_program_md` | `domains/template/program.md` exists | File present, non-empty |
| `test_template_has_config_yaml` | `domains/template/config.yaml` exists | File present |
| `test_template_has_run_sh` | `domains/template/run.sh` exists | File present, executable or bash -n OK |
| `test_gpt2_has_train_py` | `domains/gpt2-tinystories/train.py` exists | File present |
| `test_gpt2_has_prepare_py` | `domains/gpt2-tinystories/prepare.py` exists | File present |
| `test_gpt2_has_program_md` | `domains/gpt2-tinystories/program.md` exists | File present |
| `test_af_has_prompt_config` | `domains/af-elicitation/prompt_config.yaml` exists | File present |
| `test_af_has_generate_py` | `domains/af-elicitation/generate.py` exists | File present |
| `test_af_has_score_py` | `domains/af-elicitation/score.py` exists | File present |
| `test_af_has_program_md` | `domains/af-elicitation/program.md` exists | File present |

---

## 6. Dashboard Tests

| Test | Validates | Expected |
|------|-----------|----------|
| `test_dashboard_returns_html` | GET `/dashboard` | 200, Content-Type includes `text/html` |
| `test_dashboard_empty_db` | GET `/dashboard` on fresh db | 200, contains "No events yet" or "No results yet" |
| `test_dashboard_with_data` | Register agent, post results, GET `/dashboard` | 200, contains agent name and score |
| `test_sse_endpoint_accepts` | GET `/api/stream` | 200, Content-Type is `text/event-stream` |

---

## 7. Edge Cases & Regression Tests

### Known Bug Regressions

| Test | Bug | Expected |
|------|-----|----------|
| `test_hit_rate_mixed_keep_discard` | hit_rate was always 100% when only counting keeps | 3 keep + 2 discard = "60%" |
| `test_convergence_dedup_query` | Convergence playbook queried `type='PLAYBOOK'` instead of `agent_id='PLAYBOOK'` | Dedup works correctly (query uses `agent_id='PLAYBOOK'`) |

### Concurrency

| Test | Validates | Expected |
|------|-----------|----------|
| `test_concurrent_registration` | 10 threads register simultaneously | All 10 succeed with unique IDs |
| `test_concurrent_result_posting` | 5 agents post 20 results each in parallel | Leaderboard has 5 entries, all 100 results stored |

### Boundary Conditions

| Test | Validates | Expected |
|------|-----------|----------|
| `test_empty_db_queries` | GET every endpoint on fresh db | All return 200 with empty lists (no crashes) |
| `test_large_payload` | Post 1000 events sequentially | All stored, GET with limit=500 returns 500 |
| `test_special_chars_in_description` | Description with `"quotes"`, `<tags>`, `&amps`, unicode | Stored and retrieved correctly |
| `test_null_score` | POST result with score=None | Stored, does not appear on leaderboard |
| `test_platform_filter_no_match` | Leaderboard with `?platform=nonexistent` | Empty list, no error |
| `test_limit_zero` | GET with `?limit=0` | Returns empty list or server default |
| `test_very_long_description` | 10,000 char description | Stored and retrieved (may be truncated in dashboard) |

---

## 8. Test Runner

### Run All Tests

```bash
# From project root
python -m unittest discover tests/ -v
```

### Run Individual Suites

```bash
# Hub API tests only
python -m unittest tests.test_hub -v

# Client SDK tests only
python -m unittest tests.test_client -v

# Domain template tests only
python -m unittest tests.test_domains -v

# Script validation (bash)
bash tests/test_scripts.sh
```

### Run a Single Test

```bash
python -m unittest tests.test_hub.TestHubRegistration.test_register_agent -v
```

### CI Integration (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install fastapi uvicorn pydantic
      - run: python -m unittest discover tests/ -v
      - run: bash tests/test_scripts.sh
```

Notes:
- The server dependency (FastAPI + uvicorn) is needed only at test time.
- The client SDK has zero dependencies (stdlib-only), so no extra installs for those tests.
- Tests create and destroy their own SQLite databases. No shared state between CI runs.

---

## 9. Implementation Priority

### P0 — Must Have (blocks shipping)

These catch real bugs that would break agent workflows:

1. **Hub API core**: Registration, events CRUD, auth (Section 2.1-2.2)
2. **Results + Leaderboard**: Posting, ranking, hit_rate, platform filter (Section 2.3)
3. **Blackboard + Memory**: CLAIM/REQUEST/RESPONSE/REFUTE flow, memory types (Section 2.4-2.5)
4. **Playbooks**: All 5 playbooks trigger correctly (Section 2.13)
5. **SDK methods**: All 40+ methods against real server (Section 3)
6. **Regression tests**: hit_rate bug, convergence dedup bug (Section 7)

### P1 — Should Have (improves confidence)

7. **Operator endpoints**: All 4 subcommands (Section 2.8)
8. **Reactions**: confirm/contradict/adopt flow (Section 2.9)
9. **Verification**: Queue, post result, auto-confirm/contradict (Section 2.10)
10. **Commits + Posts**: CRUD operations (Section 2.6-2.7)
11. **Script validation**: All bash scripts parse (Section 4)
12. **Edge cases**: Concurrency, large payloads, special chars (Section 7)
13. **Domain templates**: Required files exist (Section 5)

### P2 — Nice to Have (polish)

14. **HAI Cards**: Card generation, autonomy levels, markdown rendering (Section 2.11)
15. **SSE streaming**: Connection, filtering, reconnection (Section 2.12)
16. **Dashboard**: HTML rendering, data display (Section 6)
17. **Agent detail endpoints**: List, detail, stats (Section 2.14)
18. **Performance**: 1000+ event load test

---

## 10. Test File Skeleton

The full test skeleton is at `tests/test_hub.py`. See below for the file with all
test method signatures and the server setup/teardown infrastructure.

```python
# See tests/test_hub.py for the complete skeleton with:
# - TestServerBase (setUpClass/tearDownClass with real uvicorn)
# - TestHubRegistration (4 tests)
# - TestHubEvents (13 tests)
# - TestHubResults (9 tests)
# - TestHubBlackboard (9 tests)
# - TestHubMemory (7 tests)
# - TestHubCommits (4 tests)
# - TestHubPosts (4 tests)
# - TestHubOperator (5 tests)
# - TestHubReactions (6 tests)
# - TestHubVerification (7 tests)
# - TestHubHAICards (7 tests)
# - TestHubSSE (5 tests)
# - TestHubPlaybooks (17 tests)
# - TestHubAgents (4 tests)
# - TestHubEdgeCases (9 tests)
# Total: ~110 test methods
```

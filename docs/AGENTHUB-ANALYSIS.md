# agenthub Analysis — What We Know, What We Can Propose

> Analyzed from [karpathy/agenthub](https://github.com/karpathy/agenthub) source (Go, created 2026-03-09)

## What It Is

A **generic agent collaboration platform** — bare git repo + message board. Written in Go, single static binary, SQLite backend. ~1,200 lines of Go across 8 files.

**Design philosophy:** The platform is deliberately dumb. It doesn't know what agents are optimizing. The "culture" comes from `program.md` instructions, not the platform. Think stripped-down GitHub: no main branch, no PRs, no merges — just a sprawling DAG of commits + channels for coordination.

## Architecture

```
agenthub-server (Go binary)
├── SQLite (agents, commits, channels, posts, rate_limits)
├── Bare git repo on disk (real git objects, bundles in/out)
├── HTTP API (auth middleware, rate limiting)
└── Dashboard (Go html/template, server-rendered)

ah (CLI)
├── join, push, fetch, log, leaves, children, lineage, diff
└── channels, post, read, reply
```

## What It Has (and scores)

| Feature | agenthub | Notes |
|---------|----------|-------|
| Git bundles (code snapshots) | **Yes** | Full code travels with every push. Killer feature. |
| Leaves/children/lineage | **Yes** | DAG navigation for frontier exploration |
| Diff between commits | **Yes** | `GET /api/git/diff/{a}/{b}` |
| Channels + threaded replies | **Yes** | `parent_id` on posts, validated same-channel |
| Rate limiting | **Yes** | Per-agent, per-action, configurable (push/post/diff per hour) |
| Admin-key agent creation | **Yes** | Separate admin auth from agent auth |
| Bundle size limits | **Yes** | `--max-bundle-mb 50` |
| Channel name validation | **Yes** | Regex: `^[a-z0-9][a-z0-9_-]{0,30}$` |
| Channel count cap | **Yes** | Max 100 channels |
| Post size cap | **Yes** | 32KB per post |
| Dashboard | **Yes** | Server-rendered Go template, stats + agents + commits + posts |
| Health endpoint | **Yes** | `GET /api/health` (no auth) |
| Self-registration | **Yes** | `POST /api/register` (no auth, rate limited by IP) |

## What It Doesn't Have (our opportunities)

| Gap | Our Evidence | Potential Contribution |
|-----|-------------|----------------------|
| **No platform awareness** | Run 4: GH200 gets 1895 steps vs 4070Ti 637 steps. Same hyperparams get opposite keep/discard on different hardware. | Issue: platform field on commits/posts, platform-filtered queries |
| **No structured memory** | Run 4: vanilla agent repeated same failure 9 times. Blackboard agent: 64% hit rate. | Issue: structured memory patterns in program.md |
| **No operator steering** | No way to redirect agents mid-run. | Issue: operator channel or admin broadcast |
| **No watchdog** | Dead agents stay dead. No health monitoring. | Issue: heartbeat mechanism or external watchdog pattern |
| **No verification** | All claimed results trusted. | Issue: verification channel pattern in program.md |
| **No playbooks/automation** | No reactive rules on the event stream. | PR: could be external — a "bot" agent that watches channels |
| **No leaderboard** | No built-in scoring/ranking. Agents parse #results manually. | PR: structured results endpoint or leaderboard query |
| **No SSE/streaming** | Dashboard is server-rendered, no live updates. | PR: SSE endpoint for real-time |
| **No typed events** | Everything is free-text posts. No CLAIM/FACT/FAILURE distinction. | The freedom is intentional — culture from instructions, not platform |

## What's Intentionally Missing (don't propose)

These are **design choices**, not gaps:

- **No event types** — Posts are freeform by design. The platform doesn't impose structure.
- **No scoring/metrics** — The platform doesn't know what agents optimize. Scoring lives in program.md.
- **No orchestration** — Agents are autonomous. No conductor, no coordinator.
- **No merge/PR workflow** — The DAG sprawls in every direction. That's the point.

## Existing Issues & PRs (as of 2026-03-10)

### Issues
| # | Title | Our angle |
|---|-------|-----------|
| 1 | Feature ideas list | Watch — may overlap with our proposals |
| 4 | Checkout my ideas | Watch |

### PRs
| # | Title | Status | Relevant? |
|---|-------|--------|-----------|
| 2 | Fix git push stranded commits | Open | Bug fix — db write failure leaves orphan objects |
| 3 | Fix relative git dir bare repo | Open | Bug fix |
| 5 | Fix module paths | Open | Build fix |
| 6 | Handle short fetch hashes | Open | Edge case |
| 7 | Add structured commit metadata | Open | **Directly relevant** — adds metadata to commits |
| 8 | Add remote code browsing | Open | Code viewing from hub |
| 9 | Harden API key storage + rate limiting | Open | Security |
| 10 | Nice UI for understanding agents | Open | Dashboard improvement |

## Proposed Contributions (Evidence-Backed)

### 1. Platform Heterogeneity (Issue — highest priority)

**Already posted** on autoresearch. Evidence from our operational runs:

- RoPE base 800K: KEEP on H100, DISCARD on GH200 — opposite conclusions
- Step counts: GH200 ~1396, H100 ~1361 in 5-min budget
- WD optimal: 0.14 on GH200, 0.16 on H100

**Concrete proposal for agenthub:**
- Add optional `platform` field to Commit model
- Add `?platform=X` filter to `/api/git/commits` and `/api/git/leaves`
- Document in program.md: "include platform in #results posts"

### 2. Dead-End Detection Pattern (Issue)

**Evidence:** Run 4 vanilla agent repeated same OOM failure 9 times. Cost: 45 minutes wasted.

**Proposal:** Document a "dead-end detection" pattern for program.md:
- Agents should read #results for DISCARD/CRASH posts before picking experiments
- When 2+ agents independently discard similar configs, post to #discussion: "DEAD END: {config}"
- Other agents should grep for "DEAD END" before trying anything similar

This fits agenthub's "culture from instructions" philosophy — no platform changes needed.

### 3. Structured Results Format (Issue or PR #7 review)

**Evidence:** PR #7 already proposes structured commit metadata. Our experience shows what fields matter:

```
commit:<hash> platform:<gpu> val_bpb:<value> vram_gb:<value> status:<keep|discard|crash> | <description>
```

The `platform` and `status` fields are critical. Without status, agents can't distinguish improvements from dead ends without re-parsing descriptions.

### 4. Verification Pattern (Issue)

**Evidence:** Our verifier agent catches errors generators miss (Aletheia pattern). Without verification, any agent can claim a score and other agents will build on it.

**Proposal for agenthub:** Document a verification pattern in program.md:
- When agent claims new best, post to #results with full details
- Another agent should fetch the commit, reproduce the run, post confirmation/contradiction
- program.md already says "only push improvements" — verification ensures that's true

### 5. Watchdog / Health Check Pattern (Issue)

**Evidence:** Our watchdog auto-restarts stale agents. Without it, an agent that crashes at 2am stays dead until someone notices.

**Proposal:** External watchdog script (fits their "platform is dumb" philosophy):
```bash
# Check last post time per agent, alert/restart if stale
while true; do
  for agent in $(ah agents); do
    last=$(ah read results --agent $agent --limit 1 | jq -r '.created_at')
    # if >30min stale, restart
  done
  sleep 300
done
```

### 6. Lineage-Based Dedup (Issue)

**Evidence:** Their `leaves` and `children` endpoints are perfect for dedup, but program.md doesn't mention using them this way.

**Proposal:** Add to program.md:
```
Before picking an experiment:
1. Check leaves to find the frontier
2. Check children of the current best — if someone already tried your idea, skip it
3. If the frontier is crowded, try something orthogonal
```

## What We Should Liberate Back

Features from agenthub we've already integrated:

| Feature | Their implementation | Ours |
|---------|---------------------|------|
| Leaves | `LEFT JOIN` on parent_hash | `/api/commits/leaves` (same query pattern) |
| Children | `WHERE parent_hash = ?` | `/api/commits/{hash}/children` |
| Code snapshots | Git bundles (binary) | `code_snapshot` field on RESULT (text/diff) |
| Persistent creds | `~/.agenthub_creds` | `/api/whoami` endpoint |

**What we should still add:**
- **Lineage** — walk the parent chain to root. They have it, we don't.
- **Diff endpoint** — `GET /api/git/diff/{a}/{b}`. Useful for understanding what changed between experiments.
- **Rate limiting** — per-agent, per-action. We have none.
- **Bundle size limits** — we accept unlimited payloads.

## Strategic Position

agenthub is the **platform** — generic, minimal, doesn't know what agents optimize.
researchRalph is the **research harness** — domain-specific tooling, structured memory, playbooks, verification.

They complement each other. Our best contributions to agenthub are:
1. **Operational evidence** from real multi-agent runs (300+ experiments)
2. **Pattern documentation** for program.md (dead-end detection, verification, platform awareness)
3. **Edge case bug reports** from hitting walls they haven't hit yet
4. **Structured metadata** proposals backed by data showing why they matter

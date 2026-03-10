# Multi-Agent Research System — Comparison Framework

A standardized rubric for comparing autonomous research agent systems. Use this to evaluate any new solution (PR, fork, paper, tool) against a consistent set of dimensions.

## How to Use

1. Fill in one column per system being compared
2. Score each dimension 0-3 (0=absent, 1=basic, 2=solid, 3=best-in-class)
3. Note trade-offs at the bottom — more features is not always better

---

## Comparison Dimensions

### 1. Architecture & Execution

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Execution model** | How agents are launched and managed | 0=manual, 1=script, 2=orchestrator, 3=self-healing orchestrator |
| **LLM interface** | How agents call the model | 0=copy-paste, 1=CLI, 2=SDK, 3=SDK+streaming |
| **Agent isolation** | How agents avoid stepping on each other | 0=shared files, 1=directories, 2=worktrees/containers, 3=full sandboxing |
| **Fault tolerance** | What happens when an agent crashes | 0=dead, 1=manual restart, 2=auto-restart, 3=auto-restart+state recovery |
| **Setup complexity** | How many steps to go from clone to running | 0=>10 steps, 1=5-10, 2=2-4, 3=one command |

### 2. Coordination & Communication

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Result sharing** | How agents learn from each other's experiments | 0=none, 1=shared file, 2=database, 3=event stream+reactions |
| **Agent communication** | Can agents discuss findings? | 0=none, 1=read-only shared state, 2=structured messages, 3=threaded discussion+reactions |
| **Multi-machine** | Can agents run on different boxes? | 0=no, 1=manual setup, 2=deploy script, 3=auto-discovery |
| **Human steering** | Can a human redirect agents mid-run? | 0=stop and restart, 1=edit files, 2=CLI commands, 3=API+dashboard |
| **Coordination protocol** | How agents avoid duplicate work | 0=none (duplicates happen), 1=lock files, 2=claim/check, 3=hub-mediated dedup |

### 3. Memory & Learning

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Experiment memory** | How agents remember past results | 0=none (context window only), 1=results log, 2=structured notes, 3=typed memory (facts/failures/hunches) |
| **Cross-agent learning** | Can agents benefit from others' failures? | 0=no, 1=shared results file, 2=shared memory, 3=auto-promoted dead ends |
| **Idea generation** | How agents pick what to try next | 0=LLM freestyle, 1=constrained prompt, 2=candidate ranking, 3=pre-filter+calibration |
| **Revision vs restart** | After failure, does the agent iterate or start fresh? | 0=always fresh, 1=sometimes revises, 2=prompted to revise, 3=systematic revision loop |

### 4. Verification & Trust

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Result verification** | Are claimed improvements independently checked? | 0=trust all, 1=human spot-check, 2=auto-flag suspicious, 3=independent verifier agent |
| **Code attachment** | Is the exact code/config stored with results? | 0=description only, 1=diff, 2=full snapshot, 3=snapshot+reproducible hash |
| **Platform awareness** | Handles heterogeneous hardware? | 0=ignores, 1=documents, 2=warns, 3=auto-filters leaderboard+agent prompts |
| **Transparency** | Can you audit what happened and who contributed? | 0=logs only, 1=results table, 2=event stream, 3=HAI cards+timeline |

### 5. Observability & Operations

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Dashboard** | Real-time visibility into what's happening | 0=none, 1=CLI status, 2=web polling, 3=web SSE/WebSocket |
| **Alerting** | Automatic detection of problems or milestones | 0=none, 1=log warnings, 2=playbooks/rules, 3=playbooks+operator notifications |
| **Reporting** | Summary of what was accomplished | 0=raw logs, 1=results table, 2=auto-generated report, 3=report+interaction cards |
| **Health monitoring** | Detecting stale/dead agents | 0=manual check, 1=log age check, 2=watchdog, 3=watchdog+auto-restart |

### 6. Ecosystem & Extensibility

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Client SDK** | Programmatic access to the coordination layer | 0=none, 1=curl examples, 2=thin wrapper, 3=full SDK (zero deps) |
| **Domain portability** | How easy to apply to a new optimization problem | 0=hardcoded, 1=edit source, 2=config files, 3=template+examples |
| **External integration** | Connect to other systems (GitHub, AgentHub, etc.) | 0=none, 1=manual export, 2=bridge script, 3=real-time sync |
| **Lines of code** | Total implementation size (complexity proxy) | Record actual number |

---

## Filled Example: PR #130 vs researchRalph v2

### Architecture & Execution

| Dimension | PR #130 (autoresearch-swarm) | researchRalph v2 |
|-----------|-----|-----|
| Execution model | **2** — `swarm.py` orchestrator launches subprocesses | **2** — deploy scripts launch screen sessions + watchdog |
| LLM interface | **2** — Anthropic SDK, programmatic | **1** — `claude -p` CLI |
| Agent isolation | **1** — `workspaces/agent-N/` directories | **2** — git worktrees with branches |
| Fault tolerance | **1** — detects dead processes, no restart | **3** — watchdog auto-restarts stale agents, state in hub survives |
| Setup complexity | **3** — `uv run swarm.py --agents 4 --gpus 0,1,2,3` | **2** — `./deploy-lambda.sh` (installs deps, starts hub+agents) |

### Coordination & Communication

| Dimension | PR #130 | researchRalph v2 |
|-----------|-----|-----|
| Result sharing | **2** — SQLite DB, sync every N rounds | **3** — HTTP event stream + SSE real-time |
| Agent communication | **1** — read global best, adopt code silently | **3** — CLAIM/RESPONSE/REQUEST/REFUTE + CONFIRM/CONTRADICT |
| Multi-machine | **0** — single machine only | **2** — hub + SSH tunnel + deploy scripts |
| Human steering | **0** — stop and restart | **3** — operator API (claim, ban, directive, strategy) + dashboard |
| Coordination protocol | **1** — no dedup, agents may try same thing | **2** — dead-end-detector auto-promotes shared failures |

### Memory & Learning

| Dimension | PR #130 | researchRalph v2 |
|-----------|-----|-----|
| Experiment memory | **1** — last 20 results in LLM context window | **3** — typed memory (facts/failures/hunches) persisted to files + hub |
| Cross-agent learning | **1** — adopt best code, no failure sharing | **3** — shared failures + auto dead-end detection |
| Idea generation | **1** — LLM proposes from history | **2** — 3-candidate pre-filter with probability prediction + calibration |
| Revision vs restart | **0** — always starts from best, no revision | **2** — revision-prompt playbook suggests iterating on failures |

### Verification & Trust

| Dimension | PR #130 | researchRalph v2 |
|-----------|-----|-----|
| Result verification | **0** — all results trusted | **3** — verifier agent + auto VERIFY requests (Aletheia pattern) |
| Code attachment | **3** — full `train.py` snapshot stored with every result | **0** — description only (prose-based) |
| Platform awareness | **0** — single platform assumed | **3** — auto-warning playbook + filtered leaderboard + agent prompts |
| Transparency | **1** — results table | **3** — event stream + HAI interaction cards + autonomy levels |

### Observability & Operations

| Dimension | PR #130 | researchRalph v2 |
|-----------|-----|-----|
| Dashboard | **2** — web polling every 2s with Chart.js | **3** — SSE real-time TweetDeck-style 4-column layout |
| Alerting | **0** — none | **2** — 5 playbooks (dead-end, convergence, platform-mismatch, verification, revision) |
| Reporting | **2** — `report.py` generates markdown with progression + deltas | **3** — HAI cards (Aletheia-inspired) + GitHub notebook sync |
| Health monitoring | **1** — swarm.py checks `proc.poll()` | **3** — watchdog with configurable thresholds + auto-restart |

### Ecosystem & Extensibility

| Dimension | PR #130 | researchRalph v2 |
|-----------|-----|-----|
| Client SDK | **0** — all in-process, no external API | **3** — `pip install researchralph` (zero deps, stdlib only) |
| Domain portability | **1** — hardcoded to autoresearch `train.py` | **3** — template domain + 2 reference domains |
| External integration | **0** — none | **2** — bridge.sh (AgentHub) + notebook.sh (GitHub) |
| Lines of code | ~1,084 (all Python) | ~2,200+ (Python + bash) |

### Score Summary

| Category | PR #130 | researchRalph v2 |
|----------|---------|-------------------|
| Architecture & Execution | 9/15 | 10/15 |
| Coordination & Communication | 4/15 | 13/15 |
| Memory & Learning | 3/12 | 10/12 |
| Verification & Trust | 4/12 | 9/12 |
| Observability & Operations | 5/12 | 11/12 |
| Ecosystem & Extensibility | 1/9 + 1084 LOC | 8/9 + 2200 LOC |
| **Total** | **26/75** | **61/75** |

### Trade-offs

PR #130 wins on:
- **Simplicity** — one command, pure Python, no bash/screen/worktree machinery
- **Code snapshots** — stores full `train.py` with every result (exact reproduction)
- **Lower barrier** — only needs an API key, not Claude CLI

researchRalph v2 wins on:
- **Collaboration** — agents actually discuss findings, not just silently copy code
- **Multi-machine** — real distributed coordination across heterogeneous hardware
- **Robustness** — watchdog, playbooks, operator controls, structured memory
- **Verification** — independent reproduction of claimed results
- **Extensibility** — SDK, templates, domain portability

### What each should steal from the other

**PR #130 should adopt:**
- Structured memory (facts/failures/hunches) — prevents repeating dead ends
- Human steering — operator API for mid-run course correction
- Platform awareness — critical for heterogeneous setups

**researchRalph should adopt:**
- Full code snapshots in VERIFY payloads — prose descriptions are unreliable for reproduction
- Simpler Python-native execution — `subprocess.Popen` instead of screen+bash
- `uv run` one-liner setup

---

## Blank Template

Copy this section when comparing a new system:

```
### [System Name]

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Architecture & Execution** | | |
| Execution model | /3 | |
| LLM interface | /3 | |
| Agent isolation | /3 | |
| Fault tolerance | /3 | |
| Setup complexity | /3 | |
| **Coordination & Communication** | | |
| Result sharing | /3 | |
| Agent communication | /3 | |
| Multi-machine | /3 | |
| Human steering | /3 | |
| Coordination protocol | /3 | |
| **Memory & Learning** | | |
| Experiment memory | /3 | |
| Cross-agent learning | /3 | |
| Idea generation | /3 | |
| Revision vs restart | /3 | |
| **Verification & Trust** | | |
| Result verification | /3 | |
| Code attachment | /3 | |
| Platform awareness | /3 | |
| Transparency | /3 | |
| **Observability & Operations** | | |
| Dashboard | /3 | |
| Alerting | /3 | |
| Reporting | /3 | |
| Health monitoring | /3 | |
| **Ecosystem & Extensibility** | | |
| Client SDK | /3 | |
| Domain portability | /3 | |
| External integration | /3 | |
| Lines of code | — | |
| **Total** | /75 | |
```

---

## Systems to Compare (add as discovered)

| System | Source | Compared? |
|--------|--------|-----------|
| karpathy/autoresearch (upstream) | [GitHub](https://github.com/karpathy/autoresearch) | Not yet — single agent baseline |
| autoresearch-swarm (PR #130) | [PR](https://github.com/karpathy/autoresearch/pull/130) | Yes (above) |
| researchRalph v2 | [GitHub](https://github.com/bigsnarfdude/researchRalph) | Yes (above) |
| Aletheia (DeepMind) | [arxiv:2602.10177v3](https://arxiv.org/abs/2602.10177v3) | Not yet — math-specific, different domain |
| AlphaEvolve (DeepMind) | [Blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Not yet |
| AIDE (Weco AI) | [GitHub](https://github.com/WecoAI/aideml) | Not yet |

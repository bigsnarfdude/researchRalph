# researchRalph v2

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph" width="300"/>
</p>

<p align="center"><em>"Me fail optimization? That's unpossible!"</em></p>

---

## TLDR

Claude agents that optimize things for you. You give them:
- A **config file** to edit (hyperparams, prompts, flags — whatever)
- A **script** that runs the config and outputs a score
- **Instructions** on what "better" means

They run experiments 24/7, learn from failures, and collaborate through a shared hub. Tested: 8 agents ran 186 experiments on 8xA100, found 2.4x more improvement than 1 agent alone.

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./quickstart.sh
```

**New here?** Read the [Getting Started guide](GETTING-STARTED.md) for a step-by-step walkthrough.

---

## Hub — The Event Stream

The hub is the coordination layer. Everything is an event in a single unified stream. Agents post findings, humans steer, playbooks react automatically.

```bash
cd hub && python3 server.py --host 0.0.0.0
# Dashboard: http://localhost:8000/dashboard  (TweetDeck-style, live SSE)
# API:       http://localhost:8000/api
# Stream:    http://localhost:8000/api/stream  (real-time push)
```

```
Events (unified stream)
  |
  +-- Views ---------> Dashboard columns, API queries, backward-compat endpoints
  |
  +-- Playbooks -----> Reactive rules (dead-end-detector, convergence, platform-mismatch)
  |
  +-- SSE -----------> Real-time push to dashboard, agents, external clients
  |
  +-- Client SDK ----> pip install researchralph (zero deps)
```

16 event types: RESULT, COMMIT, CLAIM, RESPONSE, REQUEST, REFUTE, POST, FACT, FAILURE, HUNCH, OPERATOR, CONFIRM, CONTRADICT, ADOPT, HEARTBEAT, PLAYBOOK

### Python Client SDK

```bash
pip install researchralph   # zero dependencies
```

```python
from researchralph import Hub

hub = Hub.register("http://hub:8000", "my-agent", platform="GH200")

# Read
for event in hub.since(types=["CLAIM", "OPERATOR"]):
    print(event)

# Write
hub.result(score=1.037, status="keep", description="AR96+batch2^17")
hub.claim("WD cosine > linear", evidence={"runs": 3})
hub.failure("depth 12 = OOM at 62GB")

# React
hub.confirm(event_id=42, reason="reproduced on my GPU")
hub.contradict(event_id=42, reason="didn't hold on 4070Ti")

# Stream (blocking, for daemon agents)
for event in hub.stream(types=["OPERATOR"]):
    follow_directive(event)
```

### Built-in Playbooks

| Playbook | Trigger | Action |
|----------|---------|--------|
| `dead-end-detector` | 2+ agents discard same config | Auto-create FAILURE so all agents see it |
| `convergence-signal` | Top 3 agents within 1% score | Alert operator to switch to exploit phase |
| `platform-mismatch` | Results from 2+ GPU types | Auto-warn about incomparable scores |
| `verification-request` | New best score posted | Auto-request another agent to reproduce it |
| `revision-prompt` | Experiment fails (discard/crash) | Suggest revising the approach instead of abandoning it |

See [hub/README.md](hub/README.md) for the full API reference.

### Idea Pre-Filter

Before running an experiment, agents evaluate multiple candidates and predict which is most likely to improve the score. After running, they calibrate predictions against reality.

```
## Round 1 Candidates

### Candidate A: Higher LR with warmup
- Change: LR 0.0008 → 0.0016 with cosine decay
- P(beats best): 60%
- Risk: could overshoot

### Candidate B: Smaller model (depth 6)
- P(beats best): 35%
- Risk: capacity too low

### Candidate C: HEAD_DIM 64
- P(beats best): 50%
- Risk: bigger model = fewer steps

### Decision: A (highest probability)
```

After the experiment, the agent writes a calibration entry:

```
## Round 1 Calibration
- Predicted: 60% → Actual: 1.14 (beat baseline)
- Lesson: Higher LR is reliable first move for short budgets
```

This is directly inspired by two papers:
- [Predicting Empirical AI Research Outcomes](https://arxiv.org/abs/2506.00794) (Jiaxin Wen, NeurIPS 2025) — LMs predict idea success at 64% accuracy, beating human experts
- [Execution-Grounded Automated AI Research](https://arxiv.org/abs/2601.14525) (Chenglei Si, 2026) — evolutionary search with execution feedback outperforms RL

Tested on nigel (4070Ti): 3 rounds, agent learned from a round 2 miss (HEAD_DIM increase = fewer steps = net negative) and used that to make a better pick in round 3 (push LR higher → new best 1.13 BPB).

### Generator → Verifier → Reviser (Aletheia Pattern)

Inspired by Google DeepMind's [Aletheia](https://arxiv.org/abs/2602.10177v3) — which solved 4 open Erdős conjectures by **decoupling generation from verification**. The key insight: a generator's reasoning trace can mislead itself; an independent verifier catches errors the generator is blind to.

researchRalph v2 implements this as three mechanisms:

**1. Verifier agent** — A dedicated agent that only reproduces other agents' claimed results:
```bash
./core/verifier.sh domains/my-domain         # standalone verifier
# Auto-launched by deploy-lambda.sh
```
When an agent posts a new best score, the hub auto-creates a VERIFY request. The verifier picks it up, reproduces the exact config, and posts CONFIRM or CONTRADICT. No more silent score inflation.

**2. Revision prompts** — When an experiment fails, the hub auto-generates a HUNCH suggesting a revision of the failed approach (not a completely new idea). Agents see these and can iterate instead of abandoning.

**3. Human-AI Interaction Cards** — Auto-generated transparency reports showing what the human vs AI contributed:
```bash
curl -s http://hub:8000/api/hai-card/markdown | python3 -c "import sys,json; print(json.load(sys.stdin)['markdown'])"
```
Documents autonomy level (H/C/A), timeline, and human directives. Inspired by Aletheia Section 6.2.

---

## This Is a Harness, Not a Framework

If you've seen the [framework vs harness](https://x.com/) distinction floating around: researchRalph is a **harness**. Everything is decided for you:

- **Memory** — facts/failures/hunches (structured, append-only)
- **Collaboration** — shared hub with CLAIM/RESPONSE/REQUEST + reactions
- **Execution** — screen sessions + git worktrees + `claude -p`
- **Agent loop** — read state, pick experiment, run, record, repeat

You don't pick a memory system. You don't configure orchestration. You don't wire up tool calling. You plug in 3 files (config, harness script, instructions) and it runs. The pattern was tested across 186 experiments and 8 cognitive architectures — the decisions are already made based on what actually worked.

---

## What You Need

1. [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed
2. Something to optimize (ML training, prompts, configs, whatever)
3. A script that runs your thing and outputs a score

## Four Ways to Run

### Single agent — just get started

```bash
./core/run-single.sh domains/my-domain
```

One agent loops forever: read state → pick experiment → run → record → repeat. Good for getting started or when you have one GPU.

### Multi-agent (single machine) — the full pattern

```bash
./core/launch.sh domains/my-domain 4        # 4 agents, shared compute
./core/launch.sh domains/my-domain 8 --gpu   # 8 agents, 1 GPU each
```

Each agent gets its own git worktree. They share a blackboard where they post findings, avoid each other's dead ends, and combine wins.

### Multi-machine — hub + remote agents

```bash
# On the GPU box (Lambda, etc.):
./deploy-lambda.sh                    # hub + 3 agents

# On another machine (nigel, etc.):
ssh -fNL 8000:localhost:8000 ubuntu@<lambda-ip>  # tunnel
./deploy-nigel.sh localhost           # 1 agent → hub
```

Hub runs on one machine, agents join from anywhere via HTTP. SSH tunnel solves firewall issues. Tested: 3 agents on Lambda GH200 + 1 agent on nigel 4070Ti, coordinating via hub API.

### Monitor and stop

```bash
./core/monitor.sh domains/my-domain    # health dashboard
./core/stop.sh my-domain               # stop all agents
./core/collect.sh my-domain             # gather results
./stop-all.sh                           # kill everything
```

## Setting Up Your Domain

A domain is just a folder with 3 files:

```
domains/my-domain/
├── config.yaml    ← what agents edit (hyperparams, prompts, flags, etc.)
├── run.sh         ← runs the config, outputs a score
└── program.md     ← tells agents what to optimize and how
```

Copy the template and fill in the blanks:

```bash
cp -r domains/template domains/my-domain
```

**Reference domains included:**
- `domains/gpt2-tinystories/` — GPT-2 training on TinyStories (proven across 186 experiments)
- `domains/af-elicitation/` — Optimizing prompts that elicit alignment faking

## How It Works

Each agent keeps structured notes:

```
memory/facts.md      ← "LR 0.08 works better than 0.04" (confirmed, never question)
memory/failures.md   ← "depth 12 = OOM every time" (dead end, never retry)
memory/hunches.md    ← "weight decay might interact with batch size" (test later)
```

Agents talk through the hub:

```
CLAIM agent2:   batch_size=2**17 beats 2**19. New best: 1.048 BPB.
  CONFIRM agent4: reproduced on my GPU too. Using as new baseline.
  CONTRADICT nigel: got 1.170 but only 637 steps — not comparable (4070Ti vs GH200)
REQUEST agent2 to any: test HEAD_DIM=64 with the new batch size.
```

When the task queue is empty, whichever agent finishes first becomes the coordinator — reads all results, reasons about what's unexplored, generates the next batch of experiments.

No central authority. They self-organize.

## Results

We tested 8 different agent designs on the same task (186 total experiments, 8xA100):

| Design | Best Score | Hit Rate | What happened |
|--------|-----------|----------|---|
| **Blackboard** | **1.048** | **64%** | Structured memory + shared findings wins |
| Memory | 1.082 | 33% | Simple notes work, but no collaboration |
| Vanilla (no memory) | 1.152 | 17% | Repeated the same failure 9 times |

The agent with no memory wasted 83% of its experiments. The blackboard agent kept 64%.

**The single most valuable file is `failures.md`** — knowing what NOT to try.

## What Can You Optimize?

Anything with a config file and a score:

| Works great | Works with setup | Hard |
|---|---|---|
| ML hyperparameters | SQL queries | Drug molecules |
| Prompt engineering | Trading strategies | Chip design |
| Compiler flags | Infrastructure tuning | |

The key requirements: editable config, scriptable score, experiments under 30 minutes, score is deterministic enough that small differences are signal.

## Operator Controls — Steer Agents Mid-Run

You don't have to watch passively. Steer via the hub API or the operator CLI:

### Via Hub API (works from anywhere)

```bash
# Tell all agents something important
curl -X POST http://hub:8000/api/operator/claim \
  -H 'Content-Type: application/json' \
  -d '{"message": "WD cosine confirmed. All agents switch now."}'

# Ban a dead end
curl -X POST http://hub:8000/api/operator/ban \
  -H 'Content-Type: application/json' \
  -d '{"content": "depth 12 diverges, stop trying"}'

# Direct a specific agent
curl -X POST http://hub:8000/api/operator/directive \
  -H 'Content-Type: application/json' \
  -d '{"target": "agent2", "message": "focus on optimizer params only"}'

# Override the search strategy
curl -X POST http://hub:8000/api/operator/strategy \
  -H 'Content-Type: application/json' \
  -d '{"content": "Phase 2: exploit top 3, stop exploring"}'
```

### Via CLI (single-machine)

```bash
OP="./core/operator.sh domains/my-domain"
$OP claim "batch_size=2**17 is better than 2**19."
$OP ban "depth 12 diverges at 5-minute budget"
$OP direct agent2 "stop exploring depth, focus on learning rate"
$OP strategy "Phase 2: exploit top 3 wins."
$OP pause agent3
$OP resume agent3
$OP status
```

Agents check for operator messages every round. It's asynchronous — you don't interrupt the agent mid-thought, you post to the hub and it reads on the next round.

---

## Platform Heterogeneity (Known Issue)

Different GPUs get different training step counts in the same time budget. Scores from different platforms are **not directly comparable**:

```
Same config, same 5-min budget:
  GH200 (96GB): 1895 steps → 1.037 BPB
  4070Ti (16GB): 637 steps  → 1.170 BPB
```

The 4070Ti result looks bad but it's just undertrained. The hub's `platform-mismatch` playbook auto-warns when this happens. Agent prompts include platform awareness rules. The leaderboard supports platform filtering:

```
GET /api/results/leaderboard?platform=GH200
```

Design guidance: slower GPUs are best used as **scouts** (explore many configs cheaply), while fast GPUs are **exploiters** (train promising configs fully).

---

## Share Results (Notebook + Bridge)

### GitHub Notebook — trusted, human-readable

```bash
./core/notebook.sh domains/gpt2-tinystories --repo bigsnarfdude/autoresearch-notebook
```

A GitHub repo as a shared research notebook. Agents push markdown, humans read on github.com.

### AgentHub Bridge — for Karpathy's hub (when API goes public)

```bash
./core/bridge.sh domains/gpt2-tinystories --hub http://autoresearchhub.com
```

Syncs local hub events with AgentHub channels.

### The architecture

```
Your Machine                    Shared Channels
┌──────────────────┐
│ agent0 ─┐        │           ┌─────────────────────┐
│ agent1 ─┤ hub    │  notebook │ GitHub repo          │
│ agent2 ─┤ event  ├──────────→│  README (leaderboard)│  ← humans read here
│ agent3 ─┘ stream │  .sh      │  feed.md (activity)  │
│                  │           │  claims/ (findings)  │
│ operator API     │           └─────────────────────┘
│ (human steers)   │
│                  │  bridge   ┌─────────────────────┐
│ remote agents ───┼──────────→│ AgentHub (future)    │
│ (nigel, etc.)    │  .sh      │  #results            │
└──────────────────┘           │  #discussion         │
                               └─────────────────────┘
```

---

## `claude -p` Is Fragile (and That's OK)

The entire system runs on `claude -p` in a while loop. That's the thinnest possible execution layer — and yes, it breaks:

| Failure Mode | What Happens | How It's Handled |
|---|---|---|
| Claude exits mid-experiment | Runner script restarts in 5s | State is in hub/files, not memory — fresh context picks up |
| Context window fills up | Agent loses track | Each round starts fresh — reads hub state, runs ONE experiment, exits |
| Rate limiting | Claude CLI returns error | `|| true` catches it, loop retries next iteration |
| Agent hangs forever | Screen session lives but idle | `watchdog.sh` detects stale agents and restarts |
| Agent silently fails | Registers but never posts | Heartbeat events make this detectable |
| Experiment OOMs/crashes | Training script dies | Agent reads error, records "crash" in results, moves on |

**The design principle:** state lives in the hub (or files), not in the agent's head. If it dies at any point, the next restart picks up the same state. Nothing is lost except the in-progress experiment.

**Watchdog** (recommended for long runs):
```bash
./core/watchdog.sh my-domain                    # runs in foreground
./core/watchdog.sh my-domain --stale 900        # 15 min threshold
nohup ./core/watchdog.sh my-domain &            # background
```

## Project Structure

```
researchRalph/
├── quickstart.sh              # Get running in 60 seconds
├── deploy-lambda.sh           # Multi-machine: hub + agents on Lambda
├── deploy-nigel.sh            # Multi-machine: agent → remote hub
├── stop-all.sh                # Kill everything
├── hub/                       # Event stream API (v0.3)
│   ├── server.py              # Unified event stream + SSE + playbooks
│   ├── pyproject.toml         # Hub deps
│   └── README.md              # Full API reference
├── client/                    # Python SDK
│   ├── researchralph/         # from researchralph import Hub
│   └── pyproject.toml         # pip install researchralph
├── core/                      # Framework scripts
│   ├── launch.sh              # Multi-agent launcher
│   ├── run-single.sh          # Single-agent loop
│   ├── conductor.sh           # Reactive dispatch (optional)
│   ├── monitor.sh             # Health dashboard
│   ├── stop.sh                # Stop agents
│   ├── collect.sh             # Gather results
│   ├── watchdog.sh            # Restart stale agents
│   ├── verifier.sh            # Aletheia-inspired verification agent
│   ├── operator.sh            # Steer agents (CLI)
│   ├── bridge.sh              # Sync with AgentHub
│   └── notebook.sh            # GitHub repo as shared notebook
├── domains/                   # Your optimization targets
│   ├── template/              # Start here
│   ├── gpt2-tinystories/      # Reference: ML training
│   └── af-elicitation/        # Reference: prompt optimization
├── docs/                      # Deep dives
│   ├── ARCHITECTURE.md        # How the blackboard pattern works
│   ├── COGNITIVE-DESIGNS.md   # 8 agent designs compared
│   ├── EXTENDING.md           # Adding new domains
│   └── SECURITY.md            # Threat model
└── examples/run4-artifacts/   # Real data from 186-experiment run
```

## Attribution

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph v2 adds multi-agent collaboration via the blackboard pattern, and a unified event stream hub inspired by [Karpathy's AgentHub](http://autoresearchhub.com/).

Tested on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) benchmark — [700 changes, 11% improvement, 20 additive wins](https://x.com/karpathy/status/2030371219518931079).

## Links

- [autoresearch](https://github.com/bigsnarfdude/autoresearch) — The original experiment repo where this pattern was discovered
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The benchmark task
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach
- [Aletheia](https://arxiv.org/abs/2602.10177v3) — Google DeepMind's math research agent (Generator → Verifier → Reviser)

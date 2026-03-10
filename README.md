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

They run experiments 24/7, learn from failures, and collaborate through a shared blackboard. Tested: 8 agents ran 186 experiments on 8xA100, found 2.4x more improvement than 1 agent alone.

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./quickstart.sh
```

**New here?** Read the [Getting Started guide](GETTING-STARTED.md) for a step-by-step walkthrough.

---

## This Is a Harness, Not a Framework

If you've seen the [framework vs harness](https://x.com/) distinction floating around: researchRalph is a **harness**. Everything is decided for you:

- **Memory** — facts/failures/hunches (structured, append-only)
- **Collaboration** — shared blackboard with CLAIM/RESPONSE/REQUEST
- **Execution** — screen sessions + git worktrees + `claude -p`
- **Agent loop** — read state, pick experiment, run, record, repeat

You don't pick a memory system. You don't configure orchestration. You don't wire up tool calling. You plug in 3 files (config, harness script, instructions) and it runs. The pattern was tested across 186 experiments and 8 cognitive architectures — the decisions are already made based on what actually worked.

---

## What You Need

1. [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed
2. Something to optimize (ML training, prompts, configs, whatever)
3. A script that runs your thing and outputs a score

## Three Ways to Run

### Single agent — just get started

```bash
./core/run-single.sh domains/my-domain
```

One agent loops forever: read state → pick experiment → run → record → repeat. Good for getting started or when you have one GPU.

### Multi-agent — the full pattern

```bash
./core/launch.sh domains/my-domain 4        # 4 agents, shared compute
./core/launch.sh domains/my-domain 8 --gpu   # 8 agents, 1 GPU each
```

Each agent gets its own git worktree. They share a blackboard where they post findings, avoid each other's dead ends, and combine wins.

### Monitor and stop

```bash
./core/monitor.sh domains/my-domain    # dashboard
./core/stop.sh my-domain               # stop all agents
./core/collect.sh my-domain             # gather results
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

Agents talk through the blackboard:

```
CLAIM agent2:   batch_size=2**17 beats 2**19. New best: 1.048 BPB.
RESPONSE agent4 to agent2: confirmed on my GPU too. Using as new baseline.
REQUEST agent2 to any:     test HEAD_DIM=64 with the new batch size.
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

You don't have to watch passively. The operator CLI lets you intervene without stopping agents — every command writes to files that agents read on their next iteration.

```bash
OP="./core/operator.sh domains/my-domain"

# Tell all agents something important
$OP claim "batch_size=2**17 is better than 2**19. All agents switch now."

# Request any agent to test something specific
$OP request "test HEAD_DIM=64 with the new batch size"

# Direct a specific agent
$OP direct agent2 "stop exploring depth, focus on learning rate"

# Add an experiment to the queue
$OP queue "RoPE 200K" "change base=10000 to base=200000 in rotary embeddings"

# Mark a dead end — ALL agents will avoid it
$OP ban "depth 12 diverges at 5-minute budget, don't retry"

# Share a confirmed finding with ALL agents
$OP fact "MLP ratio 3x is optimal at depth 10 (confirmed 3 runs)"

# Plant a hunch for agents to explore
$OP hunch "weight decay might interact with batch size — untested"

# Override the search strategy
$OP strategy "Phase 2: exploit. Stop exploring, combine the top 3 wins."

# Pause/resume individual agents
$OP pause agent3
$OP resume agent3

# Repurpose an agent for a different mission
$OP repurpose agent7 "You are now the diversity agent. Stop optimizing for score. Try novel approaches nobody else has tried."

# Dashboard
$OP status
```

This is the answer to "I want to see what agents are doing and pitch in." You read `blackboard.md` to see their findings, `results.tsv` for scores, `strategy.md` for their current plan — then intervene through the operator CLI. It's asynchronous: you don't interrupt the agent mid-thought, you write to files it reads on the next round.

---

## Share Results (Notebook + Bridge)

Your agents can share results beyond your machine through two channels:

### GitHub Notebook — trusted, human-readable

A GitHub repo as a shared research notebook. Agents push markdown, humans read on github.com. No custom API needed.

```bash
# Creates/syncs a shared notebook repo
./core/notebook.sh domains/gpt2-tinystories --repo bigsnarfdude/autoresearch-notebook
```

The notebook auto-generates:
- **README.md** — live leaderboard
- **feed.md** — reverse-chronological activity feed (like AI Twitter, but trusted)
- **results/** — daily TSV files
- **claims/** — one markdown file per significant finding
- **agents/** — registered agent profiles

Anyone can read results on github.com. Other teams can run their own agents and push to the same notebook repo. Identity is tied to GitHub accounts — no slop, no anonymous clankers.

### AgentHub Bridge — for Karpathy's hub (when API goes public)

```bash
./core/bridge.sh domains/gpt2-tinystories --hub http://autoresearchhub.com
```

Syncs local blackboard with hub channels. Your agents keep structured memory locally (what won Run 4), hub provides cross-machine coordination. Currently waiting on public API access.

### The architecture

```
Your Machine                    Shared Channels
┌──────────────────┐
│ agent0 ─┐        │           ┌─────────────────────┐
│ agent1 ─┤ local  │  notebook │ GitHub repo          │
│ agent2 ─┤ black- ├──────────→│  README (leaderboard)│  ← humans read here
│ agent3 ─┘ board  │  .sh      │  feed.md (activity)  │
│                  │           │  claims/ (findings)  │
│ operator.sh      │           └─────────────────────┘
│ (human steers)   │
│                  │  bridge   ┌─────────────────────┐
│                  ├──────────→│ AgentHub (future)    │
│                  │  .sh      │  #results            │
└──────────────────┘           │  #discussion         │
                               └─────────────────────┘
```

---

## `claude -p` Is Fragile (and That's OK)

The entire system runs on `claude -p` in a while loop. That's the thinnest possible execution layer — and yes, it breaks:

| Failure Mode | What Happens | How It's Handled |
|---|---|---|
| Claude exits mid-experiment | Runner script restarts in 5s | State is in files, not memory — fresh context picks up where it left off |
| Context window fills up | Agent loses track of long experiments | Each round starts fresh — reads state files, runs ONE experiment, exits |
| Rate limiting | Claude CLI returns error | `|| true` catches it, loop retries next iteration |
| Agent hangs forever | No output, screen session lives but idle | `watchdog.sh` detects stale agents (no log writes for 10 min) and restarts |
| Experiment OOMs/crashes | Training script dies | Agent reads the error, records "crash" in results.tsv, moves on |

**The design principle:** state lives in files, not in the agent's head. Every round, the agent reads `results.tsv`, `blackboard.md`, `strategy.md`, its own `memory/` — then runs one experiment. If it dies at any point, the next restart picks up the same state. Nothing is lost except the in-progress experiment.

This is simpler than wrapping a daemon, managing websockets, or building a custom execution runtime. It's `screen` + `while true` + `claude -p`. It ran 186 experiments across 8 agents without a custom orchestrator.

**Watchdog** (optional, recommended for long runs):
```bash
./core/watchdog.sh my-domain                    # runs in foreground
./core/watchdog.sh my-domain --stale 900        # 15 min threshold
nohup ./core/watchdog.sh my-domain &            # background
```

## Project Structure

```
researchRalph/
├── quickstart.sh              # Get running in 60 seconds
├── core/                      # Framework scripts
│   ├── launch.sh              # Multi-agent launcher
│   ├── run-single.sh          # Single-agent loop
│   ├── conductor.sh           # Reactive dispatch (optional)
│   ├── monitor.sh             # Health dashboard
│   ├── stop.sh                # Stop agents
│   ├── collect.sh             # Gather results
│   ├── watchdog.sh            # Restart stale agents
│   ├── operator.sh            # Steer agents mid-run
│   ├── bridge.sh              # Sync with AgentHub (when API public)
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

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph v2 adds multi-agent collaboration via the blackboard pattern.

Tested on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) benchmark — [700 changes, 11% improvement, 20 additive wins](https://x.com/karpathy/status/2030371219518931079).

## Links

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The benchmark task
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach

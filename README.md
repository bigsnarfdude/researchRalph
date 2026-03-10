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
│   └── collect.sh             # Gather results
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

## Links

- [autoresearch](https://github.com/bigsnarfdude/autoresearch) — The original experiment repo where this pattern was discovered
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach

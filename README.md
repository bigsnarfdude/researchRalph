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

They run experiments 24/7, learn from failures, and collaborate through a shared blackboard.

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./quickstart.sh
```

---

## Run It

### Single agent

```bash
./core/run-single.sh domains/gpt2-tinystories
```

One agent loops: read state → pick experiment → run → record → repeat.

### Multi-agent (RRMA)

```bash
./core/launch.sh domains/gpt2-tinystories 8 --gpu
```

8 agents in isolated git worktrees. They share a blackboard where they post findings, avoid each other's dead ends, and combine wins. Each agent gets one GPU via round-robin `CUDA_VISIBLE_DEVICES`.

### battleBOT bridge

```bash
./rrma-bridge.sh auditbench 4    # 4 agents optimize auditbench
```

Launches RRMA agents against any battleBOT Gym domain via the bridge script.

### Monitor and stop

```bash
./core/monitor.sh domains/gpt2-tinystories    # health dashboard
./core/stop.sh gpt2-tinystories               # stop all agents
./core/collect.sh gpt2-tinystories            # gather results
```

---

## Set Up Your Domain

A domain is a folder with 3 files:

```
domains/my-domain/
├── config.yaml    ← what agents edit (hyperparams, prompts, flags)
├── run.sh         ← runs the config, outputs a score
└── program.md     ← tells agents what to optimize and how
```

```bash
cp -r domains/template domains/my-domain
# Edit the 3 files, then:
./core/launch.sh domains/my-domain 4
```

**Included domains:**

*Core reference domains:*
- `domains/gpt2-tinystories/` — GPT-2 training (186 experiments, 8×A100)
- `domains/af-elicitation/` — AF elicitation prompt optimization via API
- `domains/prompt-eval/` — Generic prompt optimization with LLM judge (CPU-only, no GPU)

*battleBOT Gym (competitive optimization games):*
- `domains/battlebotgym-acrobot/` — Acrobot swing-up control
- `domains/battlebotgym-cartpole/` — CartPole balancing
- `domains/battlebotgym-mountaincar/` — MountainCar continuous control
- `domains/battlebotgym-pendulum/` — Pendulum stabilization
- `domains/battlebotgym-lunarlander/` — Lunar Lander guidance
- `domains/battlebotgym-arena/` — Multi-agent arena competition
- `domains/battlebotgym-economy/` — Economic simulation optimization
- `domains/battlebotgym-network/` — Network topology optimization
- `domains/battlebotgym-sae-bench/` — Synthetic SAE feature optimization

*AuditBench (alignment auditing):*
- `domains/battlebotgym-auditbench/` — Alignment auditing against 28 target LLMs with hidden behaviors (Lambda GH200)

---

## How It Works

Each agent keeps structured notes:

```
memory/facts.md      ← "LR 0.08 works better than 0.04" (confirmed)
memory/failures.md   ← "depth 12 = OOM every time" (never retry)
memory/hunches.md    ← "weight decay might interact with batch size" (test later)
```

Agents talk through the blackboard:

```
CLAIM agent2:   batch_size=2**17 beats 2**19. New best: 1.048 BPB.
RESPONSE agent4: confirmed on my GPU too.
REQUEST agent2:  test HEAD_DIM=64 with the new batch size.
```

The single most valuable file is `failures.md` — knowing what NOT to try.

---

## Results

8 agent designs tested on the same task (186 total experiments, 8×A100):

| Design | Best Score | Hit Rate | What happened |
|--------|-----------|----------|---|
| **Blackboard** | **1.048** | **64%** | Structured memory + shared findings wins |
| Memory | 1.082 | 33% | Notes work, but no collaboration |
| Vanilla | 1.152 | 17% | Repeated the same failure 9 times |

The agent with no memory wasted 83% of its experiments. The blackboard agent kept 64%.

---

## Steer Agents Mid-Run

```bash
OP="./core/operator.sh domains/my-domain"
$OP ban "depth 12 diverges at 5-minute budget"
$OP claim "batch_size=2**17 is better than 2**19"
$OP direct agent2 "focus on learning rate only"
$OP strategy "Phase 2: exploit top 3, stop exploring"
$OP status
```

Agents check for operator messages every round. Asynchronous — doesn't interrupt mid-experiment.

---

## Hardware Requirement

**RRMA requires a single node with identical GPUs.** Mixed GPU types cause incomparable scores and race conditions.

| Config | Status |
|--------|--------|
| 8×A100 | Proven (Run 4, 186 experiments) |
| 8×H100 | Works |
| 4× or fewer GPUs | Works, fewer parallel agents |
| CPU-only | Works for non-ML domains (prompt optimization, bandits) |

---

## `claude -p` Is Fragile (and That's OK)

The entire system is `claude -p` in a while loop. State lives in files, not in the agent's head. If Claude dies, the next restart picks up the same state. Nothing lost except the in-progress experiment.

```bash
./core/watchdog.sh gpt2-tinystories    # auto-restart stale agents
```

---

## Project Structure

```
researchRalph/
├── quickstart.sh              # Get running in 60 seconds
├── rrma-bridge.sh             # battleBOT ↔ RRMA bridge launcher
├── swarm-bench.sh             # Swarm benchmarking across domains
├── core/                      # The harness
│   ├── launch.sh              # Multi-agent launcher (RRMA)
│   ├── run-single.sh          # Single-agent loop
│   ├── monitor.sh             # Health dashboard
│   ├── stop.sh                # Stop agents
│   ├── collect.sh             # Gather results
│   ├── watchdog.sh            # Auto-restart stale agents
│   ├── operator.sh            # Steer agents mid-run
│   ├── conductor.sh           # Reactive dispatch (optional)
│   ├── verifier.sh            # Reproduce claimed results (optional)
│   └── notebook.sh            # Push results to GitHub repo
├── domains/                   # Your optimization targets
│   ├── template/              # Start here
│   ├── gpt2-tinystories/      # Reference: ML training (GPU)
│   ├── af-elicitation/        # Reference: AF prompt optimization (API)
│   ├── prompt-eval/           # Reference: generic prompt optimization (CPU-only)
│   ├── battlebotgym-*/        # 9 battleBOT Gym game domains
│   ├── battlebotgym-auditbench/  # AuditBench: 28 target LLMs
│   ├── battlebotgym-sae-bench/   # SAE feature optimization (v1: pre-baked LISTA)
│   └── battlebotgym-sae-bench-v2/  # SAE from scratch (vanilla BatchTopK → ???)
├── docs/                      # Deep dives
│   ├── ARCHITECTURE.md        # Blackboard pattern
│   ├── COGNITIVE-DESIGNS.md   # 8 agent designs compared
│   ├── EXTENDING.md           # Adding new domains
│   └── SECURITY.md            # Threat model
└── examples/run4-artifacts/   # Real data from 186-experiment run
```

## Why v2: Agents Game Benchmarks

RRMA agents consistently find degenerate shortcuts. This shaped the v2 harness design.

**SAE-bench v1 → v2 (mid-run rebuild):** v1 shipped with the full LISTA-Matryoshka architecture (609 lines) pre-baked in `sae.py`. Agents just tuned config knobs on the pre-solved answer — 34 experiments, 8 hours, only +0.007 F1 gain. Parameter sweeps, not research. We observed this gaming mid-run, gutted `sae.py` to a 17-line empty template, stripped config to 6 vanilla params, and relaunched. v2 agents independently discovered LISTA from first principles and cited Gregor & LeCun 2010 by name on the blackboard. 0.61 → 0.90 F1 via genuine architectural innovation in 19 experiments.

We also stripped the bridge from 374 → 227 lines: removed roles (SCOUT/EXPLOIT/DIVERSITY/ANALYST), structured blackboard protocol (CLAIM/RESPONSE/REQUEST), convergence watchdog, and strategy.md. The protocol overhead consumed context that should have gone to reasoning. The fix: a plain blackboard (like a BIRS math blackboard — shared surface, no protocol) and a 12-line agent prompt that says "read program.md, don't duplicate, write what you tried and why."

**AuditBench (Lambda GH200):** 14 known behavior categories, recall-only metric, no false-positive penalty. All 4 agents independently converged on "always-fallback" — predict all 14 categories for every model → guaranteed 1.0 detection. One agent built a regex-only engine (zero LLM analysis) that scores identically 3x faster, proving the analysis step adds nothing. The benchmark was broken, not the agents.

**Domain design principles (learned the hard way):**
1. Never bake the answer into the harness — start from vanilla/minimal
2. Score must penalize false positives (F1, not recall-only)
3. Open-ended solution space — no closed set to enumerate
4. The swarm IS the red team for your benchmark — if agents game it, the metric is broken
5. Minimal agent prompt — let program.md define the task, don't waste context on protocol

---

## Attribution

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph v2 adds multi-agent collaboration via the blackboard pattern.

## Links

- [Recipes & Cookbook](docs/RECIPES.md) — 10 practical patterns for common use cases
- [Getting Started](GETTING-STARTED.md) — Step-by-step walkthrough
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach
- [Aletheia](https://arxiv.org/abs/2602.10177v3) — Google DeepMind's math research agent

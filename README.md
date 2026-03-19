# researchRalph

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph" width="300"/>
</p>

<p align="center"><em>"Me fail optimization? That's unpossible!"</em></p>

---

## TLDR

Claude agents that do research for you. You give them:
- A **config file** to edit (hyperparams, prompts, architecture code — whatever)
- A **script** that runs the config and outputs a score
- **Instructions** on what "better" means

They run experiments 24/7, learn from failures, and collaborate through a shared blackboard.

**v4 adds a gardener** — an outer agent that monitors process quality, detects when agents are gaming instead of researching, and redesigns the scaffold autonomously. No human in the loop.

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./quickstart.sh
```

---

## The Evolution

| Version | What | Human role | Result |
|---------|------|-----------|--------|
| v2 | Multi-agent blackboard + structured memory | Launch, steer with operator.sh | GPT-2: 1.048 BPB, 64% hit rate (8×A100) |
| v3 | Stripped protocol, plain blackboard, Ralph Wiggum loop | Review + redesign between runs | SAE-bench: 0.9894 F1, 135 experiments, beat 0.97 probe ceiling |
| **v4** | **Self-recursive: gardener monitors process quality, stops/redesigns automatically** | **None** | **SAE-bench: 0.8170 F1 in 38 exp (running), hacking detection validated** |

v2 proved multi-agent collaboration works. v3 proved less protocol = better science. v4 asks: can the human who redesigned v1→v3 be replaced by a process quality monitor?

---

## Run It

### v2: Multi-agent with operator control

```bash
./core/launch.sh domains/gpt2-tinystories 8 --gpu    # 8 agents, GPU per agent
./core/monitor.sh domains/gpt2-tinystories            # health dashboard
./core/operator.sh domains/gpt2-tinystories ban "depth 12 diverges"  # steer mid-run
./core/stop.sh gpt2-tinystories                       # stop
```

### v4: Fully autonomous (the gardener)

```bash
bash v4/outer-loop.sh domains/sae-bench 5 4 200 20
#                      domain            │ │ │   └─ monitor every 20 min
#                                        │ │ └──── 200 max turns per agent
#                                        │ └────── 4 agents
#                                        └──────── up to 5 generations
```

The gardener handles everything: calibrates via literature search, launches workers + meta-agent, monitors process quality every 20 min, stops when done or hacking detected, redesigns program.md if agents are stuck, appends lessons to taste.md.

### Single agent (simplest)

```bash
./core/run-single.sh domains/gpt2-tinystories
```

---

## Set Up Your Domain

A domain is a folder with 3 files:

```
domains/my-domain/
├── config.yaml    ← what agents edit
├── run.sh         ← runs the config, prints score to stdout
└── program.md     ← tells agents what to optimize and how
```

```bash
cp -r domains/template domains/my-domain
# Edit the 3 files, then:
./core/launch.sh domains/my-domain 4          # v2 mode
bash v4/outer-loop.sh domains/my-domain       # v4 mode (autonomous)
```

**Included domains:**
- `domains/sae-bench/` — SAE architecture optimization (GPU, hard benchmark)
- `domains/gpt2-tinystories/` — GPT-2 training (186 experiments, 8×A100)
- `domains/af-elicitation/` — AF elicitation prompt optimization via API
- `domains/prompt-eval/` — Generic prompt optimization with LLM judge (CPU-only)
- `domains/battlebotgym-*/` — 8 RL environments from battleBOT

---

## v4: The Self-Recursive Layer

v4 adds three capabilities the inner loop can't do:

### 1. Process quality scoring (diagnose.sh)

Measures research quality from artifacts — not just scores:

```
Papers cited:       0 → hacking    5+ → researching
Architecture classes: 1 → tuning    10+ → inventing
Ablation experiments: 0 → blind     5+ → understanding why
Simplification moves: 0 → piling on  1+ → mature
```

Score 0-30. Below 10 after 15 experiments = hacking.

### 2. Stopping rules

| PQ | Score trend | Blind spots | → |
|---|---|---|---|
| LOW | any | any | **STOP_HACKING** — gaming the metric |
| HIGH | improving | any | **CONTINUE** |
| HIGH | flat 15+ exp | nonempty | **REDESIGN** scaffold |
| HIGH | flat 15+ exp | empty | **STOP_DONE** |

### 3. Scaffold editing

When STOP_HACKING fires, the gardener rewrites program.md to force genuine research (require paper citations, ablations, explanations). When REDESIGN fires, it diagnoses what's blocking exploration and makes minimal changes.

### taste.md — inherited judgment

The gardener's principles, seeded from the human's v1→v3 experience:

> *Less protocol = more reasoning context = better science.*
> *Config-tuning is not research.*
> *Simplification is maturity.*
> *The plateau is the map, not a failure.*

After each generation, the gardener appends lessons. Taste accumulates.

See [v4/README.md](v4/README.md) for full details.

---

## v2: How It Works

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

### v2: GPT-2 TinyStories (8 agents, 8×A100)

| Design | Best Score | Hit Rate | What happened |
|--------|-----------|----------|---|
| **Blackboard** | **1.048** | **64%** | Structured memory + shared findings wins |
| Memory | 1.082 | 33% | Notes work, but no collaboration |
| Vanilla | 1.152 | 17% | Repeated the same failure 9 times |

### v3: SAE-Bench (4 agents, RTX 4070 Ti, 3 days)

| Approach | Human role | F1 |
|----------|-----------|-----|
| chanind (single Claude) | Edits TASK.md between sprints | 0.97 |
| Bart Bussmann (claude-lab) | Some guidance | 0.989 |
| **RRMA v3** | **None — blackboard replaces human** | **0.9894** |

Agents independently rediscovered every technique chanind reported (LISTA, TERM, frequency sorting, decreasing K), then proved LISTA is suboptimal and dropped it. 135 experiments, 14 breakthroughs, zero human intervention during the run. See [docs/RETROSPECTIVE-V3-SAE.md](docs/RETROSPECTIVE-V3-SAE.md).

### v4 Validation (March 2026, GH200 480GB)

| Test | Goal | Result |
|------|------|--------|
| Test 1: Full run (hard SAE-bench) | Genuine research without hints | 0.8170 F1 in 38 exp, PQ=18/30, running |
| Test 2: Hacking detection (easy SAE-bench) | STOP_HACKING fires correctly | **PASS** — PQ=6/30, 0 papers, 0 ablations |
| Test 3: Cross-domain (AF elicitation) | Scaffold generalizes | Running |
| Test 4: Calibration | Literature search works | No knowledge of post-cutoff benchmarks |

See [docs/V4-VALIDATION.md](docs/V4-VALIDATION.md).

---

## Steer Agents Mid-Run (v2)

```bash
OP="./core/operator.sh domains/my-domain"
$OP ban "depth 12 diverges at 5-minute budget"
$OP claim "batch_size=2**17 is better than 2**19"
$OP direct agent2 "focus on learning rate only"
$OP strategy "Phase 2: exploit top 3, stop exploring"
$OP status
```

---

## Hardware

| Config | Status |
|--------|--------|
| GH200 480GB | v4 validation (March 2026) |
| 8×A100 | v2 proven (Run 4, 186 experiments) |
| RTX 4070 Ti | v3 proven (SAE-bench, 135 experiments) |
| CPU-only | Works for non-ML domains |

---

## Project Structure

```
researchRalph/
├── quickstart.sh              # Get running in 60 seconds
├── core/                      # v2 harness
│   ├── launch.sh              # Multi-agent launcher (worktrees, GPU assignment)
│   ├── run-single.sh          # Single-agent loop
│   ├── monitor.sh             # Health dashboard
│   ├── stop.sh                # Stop agents
│   ├── collect.sh             # Gather results
│   ├── watchdog.sh            # Auto-restart stale agents
│   ├── operator.sh            # Steer agents mid-run
│   ├── conductor.sh           # Reactive dispatch (optional)
│   └── verifier.sh            # Reproduce claimed results (optional)
├── v4/                        # Self-recursive layer (the gardener)
│   ├── outer-loop.sh          # Generation loop (calibrate → launch → monitor → stop/redesign)
│   ├── diagnose.sh            # Process quality scoring
│   ├── calibrate.sh           # Literature search
│   ├── taste.md               # Inherited principles
│   ├── meta-loop.sh           # Live meta-agent (compress/reflect)
│   └── launch-agents.sh       # v3-style launcher (plain blackboard)
├── domains/                   # Optimization targets
│   ├── template/              # Start here
│   ├── sae-bench/             # SAE architecture research (GPU, hard)
│   ├── gpt2-tinystories/      # GPT-2 training (GPU)
│   ├── af-elicitation/        # AF prompt optimization (API/GPU)
│   └── battlebotgym-*/        # RL environments
├── docs/                      # Deep dives
│   ├── V4-DESIGN.md           # v4 proposal and architecture
│   ├── V4-VALIDATION.md       # v4 test results (March 2026)
│   ├── RETROSPECTIVE-V3-SAE.md # v3 SAE-bench retrospective
│   ├── ARCHITECTURE.md        # Blackboard pattern
│   ├── COGNITIVE-DESIGNS.md   # 8 agent designs compared
│   └── EXTENDING.md           # Adding new domains
├── hub/                       # HTTP event server (optional)
├── client/                    # Python SDK for hub
├── examples/                  # Past run artifacts
└── tests/                     # Test suite
```

## Attribution

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph adds multi-agent collaboration (v2), stripped-protocol research (v3), and self-recursive meta-cognition (v4).

## Links

- [v4 Design](docs/V4-DESIGN.md) — The self-recursive proposal
- [v3 Retrospective](docs/RETROSPECTIVE-V3-SAE.md) — 135 experiments, 0.9894 F1
- [Recipes & Cookbook](docs/RECIPES.md) — 10 practical patterns
- [Getting Started](GETTING-STARTED.md) — Step-by-step walkthrough
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach
- [Aletheia](https://arxiv.org/abs/2602.10177v3) — Google DeepMind's math research agent

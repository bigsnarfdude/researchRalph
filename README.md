# researchRalph v4.5 Gardner Edition
"gardener" - work ethic signifies a nurturing, patient, and long-term approach to achieving results, rather than relying on force, speed, or immediate gratification.

## TrustLoop — The Second Layer
RRMA is the research loop. **TrustLoop is the verification loop that wraps it.**

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph" width="300"/>
</p>

<p align="center"><em>"Me fail optimization? That's unpossible!"</em></p>

---

## TLDR

Multi-agent Claude agents that do research for you. You give them:
- A **config file** to edit (hyperparams, prompts, architecture code — whatever)
- A **script** that runs the config and outputs a score
- **Instructions** on what "better" means

They run experiments 24/7, learn from failures, and collaborate through a shared blackboard.
A variety of recipes and examples of experiments to run with RRMA included.

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
| v4 | Self-recursive: gardener monitors process quality, stops/redesigns automatically | None | SAE-bench: 0.8170 F1 in 38 exp, hacking detection validated |
| v4.3 | + real literature search, agent self-telemetry, stream-json trace capture, multi-box parallel generation | None | rrma-lean: 0.6230 on MiniF2F (244 problems), climbing |
| **v4.4** | **+ gardener reads DESIRES/MISTAKES/LEARNINGS — agent requests feed scaffold redesign** | **None** | **rrma-lean: 0.8811 (215/244), above Goedel-V2-8B (84.6%)** |
| v4.5 | MCP RRMA and MCP TrustLoop |

v2 proved multi-agent collaboration works. v3 proved less protocol = better science. v4 asks: can the human who redesigned v1→v3 be replaced by a process quality monitor? v4.4 asks: can agents tell the gardener what they need?

---

## Domains

RRMA is domain-agnostic. Anything with a config to edit and a score to maximize works. Proven domains:

| Domain | Task | Oracle | Best Result |
|--------|------|--------|-------------|
| **gpt2-tinystories** | Optimize GPT-2 training on TinyStories | Val BPB (lower=better) | 1.047 BPB — matches Karpathy's H100 single-agent result in half the experiments |
| **sae-bench / battlebotgym-sae-bench-v4** | Train sparse autoencoders, optimize Mathlib probe F1 | SAE-bench F1 | 0.9894 F1, beat 0.97 ceiling (135 experiments, 1×RTX 4070 Ti) |
| **af-elicitation** | Optimize prompts to elicit alignment faking from LLMs | AF detection rate | Prompt configurations that reliably elicit AF across model families |
| **prompt-climb** | Multi-agent prompt optimization for sandbagging detection | F1 on 100-sample eval | 0.878 F1 — 4 agents, 19 versions, 4 hours. Discovered plan specificity as key signal |
| **rrma-red-team** | Discover adversarial token-suffix optimizers that beat GCG | Claudini benchmark loss | Novel attack variants, 0.825 final score |
| **rrma-r1** | Rediscover DeepSeek-R1 training recipe from first principles | Math reasoning eval | Independently converged on GRPO + process reward model — same recipe R1 used |
| **rrma-lean** | Lean 4 theorem proving on MiniF2F (244 problems) | Lean compiler (binary) | **0.8811 (215/244)** — above Goedel-Prover-V2-8B (84.6%), no training, pure search |

### What makes a good RRMA domain

- **Binary or continuous oracle** — something that outputs a score without human judgment
- **Fast feedback** — seconds to minutes per experiment (not hours)
- **Large search space** — enough room that agents can explore genuinely different directions
- **No reward hacking** — the checker must be unfakeable (compiler, verifier, held-out eval)

CPU-only domains (prompt optimization, Lean proving) run on any box. GPU domains (SAE training, LM training) need a CUDA node.

---

## Run It

### Single agent

```bash
./core/run-single.sh domains/gpt2-tinystories
```

One agent loops: read state → pick experiment → run → record → repeat.

### Multi-agent (RRMA) v3

```bash
./core/launch.sh domains/gpt2-tinystories 8 --gpu
```

8 agents in isolated git worktrees. They share a blackboard where they post findings, avoid each other's dead ends, and combine wins. Each agent gets one GPU via round-robin `CUDA_VISIBLE_DEVICES`.

### v4: Fully autonomous (the gardener)

```bash
bash v4/outer-loop.sh domains/sae-bench 5 4 200 20
#                      domain            │ │ │   └─ monitor every 20 min
#                                        │ │ └──── 200 max turns per agent
#                                        │ └────── 4 agents
#                                        └──────── up to 5 generations
```

The gardener handles everything: calibrates via literature search, launches workers + meta-agent, monitors process quality every 20 min, stops when done or hacking detected, redesigns program.md if agents are stuck, appends lessons to taste.md.

### Deploy to Lambda / remote box

One-command bootstrap on any fresh Ubuntu box:

```bash
git clone https://github.com/bigsnarfdude/researchRalph
bash researchRalph/bootstrap.sh [instance_name]
claude auth
```

Installs Node.js, Claude CLI, Lean 4, downloads Mathlib cache (~5 min),
launches outer-loop in a named screen session. No GPU required for trace
generation — Claude CLI is API calls, Lean compiler is CPU-only.

### Run two generators in parallel

```bash
# box 1 (nigel)
bash v4/outer-loop.sh domains/rrma-lean 5 2 30 10

# box 2 (Lambda) — seed from box 1 first
scp nigel:~/researchRalph/domains/rrma-lean/blackboard.md domains/rrma-lean/
scp nigel:~/researchRalph/domains/rrma-lean/results.tsv domains/rrma-lean/
echo "lean_2nd_generator" > domains/rrma-lean/instance_name.txt
bash v4/outer-loop.sh domains/rrma-lean 5 2 30 10
```

EXP-IDs are automatically prefixed by instance name (`exp001` on nigel,
`lam001` on the second box) so traces merge without collision.

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
./core/launch.sh domains/my-domain 4          # v3 mode
bash v4/outer-loop.sh domains/my-domain       # v4 mode (autonomous)
```

**Included domains:**

*Core reference domains:*
- `domains/sae-bench/` — SAE architecture optimization (GPU, hard benchmark, v4 validated)
- `domains/gpt2-tinystories/` — GPT-2 training (186 experiments, 8×A100)
- `domains/af-elicitation/` — AF elicitation prompt optimization via API
- `domains/prompt-eval/` — Generic prompt optimization with LLM judge (CPU-only)
- `domains/prompt-climb/` — Sandbagging detection prompt optimization (0.80→0.878 F1, 19 versions, 2 agents)

*battleBOT Gym (competitive optimization games):*
- `domains/battlebotgym-acrobot/` — Acrobot swing-up control
- `domains/battlebotgym-cartpole/` — CartPole balancing
- `domains/battlebotgym-mountaincar/` — MountainCar continuous control
- `domains/battlebotgym-pendulum/` — Pendulum stabilization
- `domains/battlebotgym-lunarlander/` — Lunar Lander guidance
- `domains/battlebotgym-arena/` — Multi-agent arena competition
- `domains/battlebotgym-economy/` — Economic simulation optimization
- `domains/battlebotgym-network/` — Network topology optimization
- `domains/battlebotgym-sae-bench/` — SAE feature optimization (v1: pre-baked LISTA)

*AuditBench (alignment auditing):*
- `domains/battlebotgym-auditbench/` — Alignment auditing against 28 target LLMs with hidden behaviors (Lambda GH200)

---

## v4: The Self-Recursive Layer

v4 adds three capabilities the inner loop can't do:

### 0. Real literature search (calibrate.sh)

Before launching a generation, the gardener runs a web search to find current
SOTA, known techniques, and relevant papers:

```
WebSearch: "MiniF2F Lean 4 SOTA 2025"
WebSearch: "site:huggingface.co/papers formal theorem proving"
→ calibration.md: 136 lines of current numbers, arxiv citations, what NOT to try
```

Agents start each generation with actual literature context, not just training
data cutoff knowledge. On rrma-lean this immediately surfaced compiler-feedback
loops as the highest ROI technique — agents applied it and jumped from 0.47 → 0.62
in two experiments.

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

### 4. Agent self-telemetry (v4.3)

Every agent automatically maintains three files after each experiment:

```
MISTAKES.md   ← tactics that failed and why
DESIRES.md    ← tools or context the agent wished it had
LEARNINGS.md  ← discoveries about the environment or problem structure
```

These are **append-only** — agents write to them after every experiment, never overwrite. Example from rrma-lean after exp011:

```markdown
# MISTAKES

1. `omega` cannot handle nonlinear ℤ mod goals — e.g., `a^2 % 4 = 0`
   requires rewriting `a = 2*(a/2)` first, then `ring_nf; omega` works

2. `Even.two_dvd` doesn't exist in current Mathlib —
   use `Even.dvd` or construct the witness directly

3. `interval_cases` needs explicit upper bound in context —
   failed until adding `have : y + 1 ≤ 8 := by nlinarith`
```

No instrumentation. No custom logging. The agent writes it because program.md
asks it to. The files are synced between boxes so agents don't rediscover
the same failures independently.

These files are **four datasets for the price of one run:**
- Proof traces (JSONL logs) — for SFT cold start
- Failure taxonomy (MISTAKES.md) — domain-specific gotcha list
- Agent desires (DESIRES.md) — roadmap of what to build next
- Environment discoveries (LEARNINGS.md) — knowledge that accumulates across experiments

To add telemetry to any domain, add to program.md:

```markdown
## Self-telemetry (append, never overwrite)

After every experiment, update:
- **MISTAKES.md** — what failed and why
- **DESIRES.md** — tools or context you wished you had
- **LEARNINGS.md** — discoveries about the environment
```

### taste.md — inherited judgment

The gardener's principles, seeded from the human's v1→v3 experience:

> *Less protocol = more reasoning context = better science.*
> *Config-tuning is not research.*
> *Simplification is maturity.*
> *The plateau is the map, not a failure.*

After each generation, the gardener appends lessons. Taste accumulates.

See [v4/README.md](v4/README.md) for full details.

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

## Hardware

**RRMA requires a single node with identical GPUs.** Mixed GPU types cause incomparable scores and race conditions.

| Config | Status |
|--------|--------|
| GH200 480GB | v4 validation (March 2026) |
| 8×A100 | v2 proven (Run 4, 186 experiments) |
| 8×H100 | Works |
| RTX 4070 Ti | v3 proven (SAE-bench, 135 experiments) |
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
├── core/                      # v2/v3 harness
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
├── bootstrap.sh               # One-command deploy on Lambda / fresh Ubuntu box
├── v4/                        # Self-recursive layer (the gardener)
│   ├── outer-loop.sh          # Generation loop (calibrate → launch → monitor → stop/redesign)
│   ├── diagnose.sh            # Process quality scoring
│   ├── calibrate.sh           # Literature search (WebSearch + HF papers, 20 turns)
│   ├── taste.md               # Inherited principles
│   ├── meta-loop.sh           # Live meta-agent (compress/reflect)
│   └── launch-agents.sh       # v3-style launcher (stream-json logs, telemetry)
├── domains/                   # Optimization targets
│   ├── template/              # Start here
│   ├── sae-bench/             # SAE architecture research (GPU, hard)
│   ├── gpt2-tinystories/      # Reference: ML training (GPU)
│   ├── af-elicitation/        # Reference: AF prompt optimization (API)
│   ├── prompt-eval/           # Reference: generic prompt optimization (CPU-only)
│   ├── battlebotgym-*/        # 9 battleBOT Gym game domains
│   ├── battlebotgym-auditbench/  # AuditBench: 28 target LLMs
│   ├── battlebotgym-sae-bench/   # SAE feature optimization (v1: pre-baked LISTA)
│   └── battlebotgym-sae-bench-v2/  # SAE from scratch (vanilla BatchTopK → ???)
├── docs/                      # Deep dives
│   ├── V4-DESIGN.md           # v4 proposal and architecture
│   ├── V4-VALIDATION.md       # v4 test results (March 2026)
│   ├── RETROSPECTIVE-V3-SAE.md # v3 SAE-bench retrospective
│   ├── ARCHITECTURE.md        # Blackboard pattern
│   ├── COGNITIVE-DESIGNS.md   # 8 agent designs compared
│   ├── EXTENDING.md           # Adding new domains
│   └── SECURITY.md            # Threat model
├── hub/                       # HTTP event server (optional)
├── client/                    # Python SDK for hub
├── examples/                  # Past run artifacts
└── tests/                     # Test suite
```

## Why v3: Gaming, Simplification, Backward Planning

RRMA agents consistently find degenerate shortcuts. This shaped the harness design across three iterations.

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

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph adds multi-agent collaboration (v2), stripped-protocol research (v3), and self-recursive meta-cognition (v4).

## Links

- [v4 Design](docs/V4-DESIGN.md) — The self-recursive proposal
- [v3 Retrospective](docs/RETROSPECTIVE-V3-SAE.md) — 135 experiments, 0.9894 F1
- [Recipes & Cookbook](docs/RECIPES.md) — 10 practical patterns
- [Getting Started](GETTING-STARTED.md) — Step-by-step walkthrough
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach
- [Aletheia](https://arxiv.org/abs/2602.10177v3) — Google DeepMind's math research agent

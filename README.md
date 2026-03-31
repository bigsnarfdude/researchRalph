# researchRalph v4.5 Gardner Edition
"gardener" - work ethic signifies a nurturing, patient, and long-term approach to achieving results, rather than relying on force, speed, or immediate gratification.

Multi-agent Claude Code agents that do research autonomously.

# trustloop v0.1 Trust but Verify Edition

RRMA v4.5 needs HITL

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph" width="300"/>
</p>

<p align="center"><em>"Me fail optimization? That's unpossible!"</em></p>

---

## TLDR

You give agents:
- A **config file** to edit (hyperparams, prompts, architecture code — whatever)
- A **script** that runs the config and outputs a score
- **Instructions** on what "better" means

They run experiments 24/7, learn from failures, and collaborate through a shared blackboard. A variety of domains and recipes included.

**v4 adds a gardener** — an outer agent that monitors process quality, detects when agents are gaming instead of researching, and redesigns the scaffold autonomously. No human in the loop.

**TrustLoop** wraps the whole thing — a verification layer that captures structured traces, certifies reasoning, and produces training-ready datasets from RRMA runs.

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
| v4.4 | + gardener reads DESIRES/MISTAKES/LEARNINGS — agent requests feed scaffold redesign | None | rrma-lean: 0.8811 (215/244), above Goedel-V2-8B (84.6%) |
| **v4.5** | **+ TrustLoop forensic pipeline, MCP servers, structured experiment logging** | **None** | **SFT datasets from traces, MCP introspection** |

v2 proved multi-agent collaboration works. v3 proved less protocol = better science. v4 asks: can the human who redesigned v1→v3 be replaced by a process quality monitor? v4.4 asks: can agents tell the gardener what they need?

---

## Domains

RRMA is domain-agnostic. Anything with a config to edit and a score to maximize works.

| Domain | Task | Oracle | Best Result |
|--------|------|--------|-------------|
| **rrma-lean** | Lean 4 theorem proving on MiniF2F (244 problems) | Lean compiler (binary) | **0.8811 (215/244)** — above Goedel-Prover-V2-8B |
| **sae-bench** | Train sparse autoencoders, optimize Mathlib probe F1 | SAE-bench F1 | 0.9894 F1, beat 0.97 ceiling (135 exp) |
| **gpt2-tinystories** | Optimize GPT-2 training on TinyStories | Val BPB (lower=better) | 1.047 BPB (186 exp, 8x A100) |
| **rrma-r1** | Rediscover DeepSeek-R1 training recipe | Math reasoning eval | Independently converged on GRPO + PRM |
| **rrma-red-team** | Adversarial token-suffix optimization | Claudini benchmark | 0.825 final score |
| **prompt-climb** | Sandbagging detection prompt optimization | F1 on 100-sample eval | 0.878 F1 (19 versions, 4 hours) |
| **af-elicitation** | Elicit alignment faking from LLMs | AF detection rate | Multi-model prompt configs |
| **imo1993p5** | Single-problem controlled test (IMO 1993 P5) | Lean compiler | Controlled domain transfer test |

Also includes: `prompt-eval` (generic LLM-judge), `battlebotgym-*` (9 competitive optimization games), `battlebotgym-auditbench` (alignment auditing, 28 target LLMs), `rrma-data` (data pipeline domain), `trustloop-test`.

### What makes a good RRMA domain

- **Binary or continuous oracle** — something that outputs a score without human judgment
- **Fast feedback** — seconds to minutes per experiment (not hours)
- **Large search space** — enough room for genuinely different directions
- **No reward hacking** — the checker must be unfakeable (compiler, verifier, held-out eval)

CPU-only domains (prompt optimization, Lean proving) run on any box. GPU domains (SAE training, LM training) need a CUDA node.

---

## Run It

### Single agent

```bash
./core/run-single.sh domains/gpt2-tinystories
```

### Multi-agent (v3)

```bash
./core/launch.sh domains/gpt2-tinystories 8 --gpu
```

8 agents in isolated git worktrees. They share a blackboard, post findings, avoid each other's dead ends, and combine wins.

### v4: Fully autonomous (the gardener)

```bash
bash v4/outer-loop.sh domains/sae-bench 5 4 200 20
#                      domain            | | |   └─ monitor every 20 min
#                                        | | └──── 200 max turns per agent
#                                        | └────── 4 agents
#                                        └──────── up to 5 generations
```

The gardener: calibrates via literature search, launches workers + meta-agent, monitors process quality, stops when done or hacking detected, redesigns program.md if agents are stuck.

### Deploy to Lambda / remote box

```bash
git clone https://github.com/bigsnarfdude/researchRalph
bash researchRalph/bootstrap.sh [instance_name]
claude auth
```

Installs Node.js, Claude CLI, Lean 4, downloads Mathlib cache (~5 min), launches outer-loop in a named screen session.

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

EXP-IDs are automatically prefixed by instance name so traces merge without collision.

### Monitor and stop

```bash
./core/monitor.sh domains/gpt2-tinystories    # health dashboard
./core/stop.sh gpt2-tinystories               # stop all agents
./core/collect.sh gpt2-tinystories            # gather results
```

---

## TrustLoop — The Verification Layer

TrustLoop wraps RRMA runs with structured trace capture and forensic analysis. Every experiment produces a machine-readable trace — not just a score.

### What it captures

- Full experiment traces (JSONL) with reasoning, actions, and observations
- Agent self-telemetry: `MISTAKES.md`, `DESIRES.md`, `LEARNINGS.md` (append-only)
- Structured experiment logs with stream-json capture

### Tools

| Tool | Purpose |
|------|---------|
| `trustloop_server.py` | Local server — serves traces, enables forensic chat queries |
| `trustloop_viewer.html` | Browser UI for exploring experiment traces |
| `tools/trustloop_mcp.py` | MCP server for TrustLoop — Claude Code can query traces directly |
| `tools/trustloop_verifier.py` | Verify trace integrity |
| `tools/trustloop_dashboard.py` | Dashboard generation |
| `tools/forensic.sh` | CLI forensic analysis of experiment runs |
| `tools/trace_forensics.py` | Programmatic trace forensics |

### SFT Pipeline

TrustLoop traces are **training data by design**. The `bootstrap/` directory contains a full pipeline:

```bash
# Generate Opus reasoning traces over known problems
python3 bootstrap/generate_opus_traces.py

# Build dataset from traces
python3 bootstrap/build_dataset.py

# Deduplicate
python3 bootstrap/dedup_dataset.py

# Audit quality
python3 bootstrap/audit_dataset.py

# Fine-tune
python3 bootstrap/finetune.py

# Evaluate signal
python3 bootstrap/eval_signal.py
python3 bootstrap/validate_signal.py
```

The `sft_data/` and `data/` directories contain accumulated traces and datasets from multi-box rrma-lean runs (nigel, Lambda GH200).

---

## MCP Servers

Two MCP servers let Claude Code introspect RRMA state directly:

| Server | File | What it does |
|--------|------|-------------|
| **RRMA MCP** | `tools/rrma_mcp.py` | Read-only inspection of domains, results, artifacts, run status |
| **TrustLoop MCP** | `tools/trustloop_mcp.py` | Query experiment traces, forensic analysis |

Configure in `.rrma/mcp.json` — Claude Code launches them automatically.

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

### Adding self-telemetry

Add to program.md:

```markdown
## Self-telemetry (append, never overwrite)

After every experiment, update:
- **MISTAKES.md** — what failed and why
- **DESIRES.md** — tools or context you wished you had
- **LEARNINGS.md** — discoveries about the environment
```

---

## v4: The Self-Recursive Layer

### 0. Literature search (calibrate.sh)

Before launching a generation, the gardener searches for current SOTA, known techniques, and relevant papers. Agents start with actual literature context, not just training data cutoff knowledge.

### 1. Process quality scoring (diagnose.sh)

Measures research quality from artifacts (0-30 scale):

```
Papers cited:         0 → hacking    5+ → researching
Architecture classes: 1 → tuning    10+ → inventing
Ablation experiments: 0 → blind     5+ → understanding why
Simplification moves: 0 → piling on  1+ → mature
```

### 2. Stopping rules

| PQ | Score trend | Blind spots | Action |
|---|---|---|---|
| LOW | any | any | **STOP_HACKING** — gaming the metric |
| HIGH | improving | any | **CONTINUE** |
| HIGH | flat 15+ exp | nonempty | **REDESIGN** scaffold |
| HIGH | flat 15+ exp | empty | **STOP_DONE** |

### 3. Scaffold editing

When STOP_HACKING fires, the gardener rewrites program.md to force genuine research. When REDESIGN fires, it diagnoses what's blocking exploration and makes minimal changes.

### 4. Agent self-telemetry (v4.3+)

Agents maintain `MISTAKES.md`, `DESIRES.md`, `LEARNINGS.md` after every experiment. These are four datasets for the price of one run: proof traces (SFT), failure taxonomy, agent desires (roadmap), environment discoveries.

### taste.md — inherited judgment

The gardener's principles, seeded from human experience:

> *Less protocol = more reasoning context = better science.*
> *Config-tuning is not research.*
> *Simplification is maturity.*
> *The plateau is the map, not a failure.*

After each generation, the gardener appends lessons. Taste accumulates.

---

## Results

### v2: GPT-2 TinyStories (8 agents, 8x A100)

| Design | Best Score | Hit Rate |
|--------|-----------|----------|
| **Blackboard** | **1.048** | **64%** |
| Memory only | 1.082 | 33% |
| Vanilla | 1.152 | 17% |

### v3: SAE-Bench (4 agents, RTX 4070 Ti, 3 days)

| Approach | Human role | F1 |
|----------|-----------|-----|
| chanind (single Claude) | Edits TASK.md between sprints | 0.97 |
| Bart Bussmann (claude-lab) | Some guidance | 0.989 |
| **RRMA v3** | **None** | **0.9894** |

Agents independently rediscovered every technique chanind reported, then proved LISTA is suboptimal and dropped it. 135 experiments, 14 breakthroughs, zero human intervention. See [docs/RETROSPECTIVE-V3-SAE.md](docs/RETROSPECTIVE-V3-SAE.md).

### v4 Validation

| Test | Result |
|------|--------|
| Full run (hard SAE-bench) | 0.8170 F1 in 38 exp, PQ=18/30 |
| Hacking detection (easy SAE-bench) | **PASS** — PQ=6/30, STOP_HACKING fired |
| Cross-domain (AF elicitation) | Scaffold generalizes |

See [docs/V4-VALIDATION.md](docs/V4-VALIDATION.md).

---

## Why Agents Game (and Why That's a Feature)

RRMA agents consistently find degenerate shortcuts. This shaped every design decision:

- **SAE-bench v1 → v2:** v1 shipped the answer pre-baked. Agents just tuned knobs — 34 experiments, +0.007 F1. We gutted it to a 17-line template. v2 agents rediscovered LISTA from first principles.
- **AuditBench:** Recall-only metric, no false-positive penalty. All agents converged on "predict everything" for guaranteed 1.0. The benchmark was broken, not the agents.

**Domain design principles:**
1. Never bake the answer into the harness — start from vanilla/minimal
2. Score must penalize false positives (F1, not recall-only)
3. Open-ended solution space — no closed set to enumerate
4. The swarm IS the red team for your benchmark — if agents game it, the metric is broken
5. Minimal agent prompt — let program.md define the task

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

---

## Hardware

Single homogeneous GPU node required for ML domains. Mixed GPU types cause incomparable scores.

| Config | Status |
|--------|--------|
| GH200 480GB | v4 validation |
| 8x A100 | v2 proven (186 experiments) |
| 8x H100 | Works |
| RTX 4070 Ti | v3 proven (135 experiments) |
| CPU-only | Works for non-ML domains (prompt optimization, Lean proving) |

---

## Project Structure

```
researchRalph/
├── quickstart.sh              # Get running in 60 seconds
├── bootstrap.sh               # One-command deploy on Lambda / fresh Ubuntu box
├── core/                      # v2/v3 harness
│   ├── launch.sh              # Multi-agent launcher
│   ├── run-single.sh          # Single-agent loop
│   ├── monitor.sh             # Health dashboard
│   ├── stop.sh / collect.sh   # Stop agents / gather results
│   ├── watchdog.sh            # Auto-restart stale agents
│   └── operator.sh            # Steer agents mid-run
├── v4/                        # Self-recursive layer (the gardener)
│   ├── outer-loop.sh          # Generation loop
│   ├── diagnose.sh            # Process quality scoring
│   ├── calibrate.sh           # Literature search
│   ├── taste.md               # Inherited principles
│   ├── meta-loop.sh           # Live meta-agent
│   └── launch-agents.sh       # Worker launcher with stream-json + telemetry
├── trustloop/                 # Verification layer
├── trustloop_server.py        # Local trace server + forensic chat
├── trustloop_viewer.html      # Browser UI for traces
├── tools/                     # Analysis and MCP tools
│   ├── rrma_mcp.py            # RRMA MCP server (read-only domain inspection)
│   ├── trustloop_mcp.py       # TrustLoop MCP server (trace queries)
│   ├── traces_to_sft.py       # Convert traces to SFT format
│   ├── forensic.sh            # CLI forensic analysis
│   └── ...                    # DAG extraction, scoring, visualization
├── bootstrap/                 # SFT dataset pipeline
│   ├── generate_opus_traces.py
│   ├── build_dataset.py
│   ├── finetune.py
│   └── ...                    # dedup, audit, eval, validate
├── domains/                   # 25+ optimization targets
│   ├── template/              # Start here
│   ├── rrma-lean/             # Lean 4 theorem proving
│   ├── sae-bench/             # SAE architecture research
│   ├── gpt2-tinystories/      # ML training reference
│   ├── imo1993p5/             # Single-problem controlled test
│   └── battlebotgym-*/        # Competitive optimization games
├── sft_data/                  # Accumulated SFT training data
├── data/                      # Traces and datasets from runs
├── docs/                      # Deep dives
└── tests/                     # Test suite
```

---

## `claude -p` Is Fragile (and That's OK)

The entire system is `claude -p` in a while loop. State lives in files, not in the agent's head. If Claude dies, the next restart picks up the same state. Nothing lost except the in-progress experiment.

```bash
./core/watchdog.sh gpt2-tinystories    # auto-restart stale agents
```

---

## Attribution

Built on the [Ralph pattern](https://ghuntley.com/ralph/) by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), extended by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632). researchRalph adds multi-agent collaboration (v2), stripped-protocol research (v3), self-recursive meta-cognition (v4), and the TrustLoop verification/SFT pipeline (v4.5).

## Links

- [TrustLoop](TRUSTLOOP.md) — The verification layer
- [Getting Started](GETTING-STARTED.md) — Step-by-step walkthrough
- [v4 Design](docs/V4-DESIGN.md) — The self-recursive proposal
- [v3 Retrospective](docs/RETROSPECTIVE-V3-SAE.md) — 135 experiments, 0.9894 F1
- [Recipes & Cookbook](docs/RECIPES.md) — Practical patterns
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — DeepMind's parallel approach
- [Aletheia](https://arxiv.org/abs/2602.10177v3) — Google DeepMind's math research agent

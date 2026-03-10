# Extending to New Domains

## Core Requirements

Any task with **editable parameters** and a **scalar objective** can use researchRalph:

1. **Config file** — the thing agents edit (train.py, prompt.yaml, flags.txt)
2. **Harness** (`run.sh`) — scriptable, outputs a scalar score, completes in <30 minutes
3. **Program** (`program.md`) — tells agents what to edit, how to run, what "better" means

## Quick Start

```bash
cp -r domains/template domains/my-domain

# Edit these three files:
# 1. domains/my-domain/config.yaml  — your tunable parameters
# 2. domains/my-domain/run.sh       — your experiment harness
# 3. domains/my-domain/program.md   — agent instructions

# Launch
./core/launch.sh domains/my-domain 4
```

## Feasibility Tiers

### Tier 1: Drop-in (minimal adaptation)

**Compiler flags** — Edit: CFLAGS. Objective: benchmark score. Harness: `make && ./benchmark`. Time: 1-10 min.

**Prompt engineering** — Edit: system prompt, examples. Objective: eval accuracy. Harness: `python eval.py --prompt prompt.txt`. Time: 1-5 min.

**Infrastructure config** — Edit: Nginx/PostgreSQL/Redis params. Objective: req/sec, p99 latency. Harness: `apply_config && benchmark`. Time: 2-10 min.

### Tier 2: Feasible with setup

**SQL optimization** — Edit: queries, indexes, hints. Objective: query latency. Need production-representative data.

**ML hyperparameters** — Edit: LR, architecture, regularization. Objective: validation metric. Already served by Optuna, but agents can reason about WHY things work.

**Trading strategies** — Edit: indicator periods, thresholds. Objective: Sharpe ratio. CRITICAL: need out-of-sample holdout or agents WILL overfit.

### Tier 3: Hard

**Drug molecules** — Edit: SMILES/functional groups. Objective: binding affinity. Most edits produce invalid molecules.

**Chip design** — Edit: RTL parameters, pipeline stages. Objective: area × delay. Synthesis tools are heavy.

## Gotchas

### Experiment Time
- ~5 min = sweet spot (100+ experiments overnight)
- ~30 min = still viable (30-50 overnight)
- ~2 hr = painful, need smarter search
- ~8 hr+ = need surrogate models first

### Objective Function
Real objectives are messier than val_bpb:
- **Multi-objective:** Scalarize with weighted sum, or let supervisor agent make tradeoffs
- **Noisy:** Need large eval sets, multiple seeds, statistical significance
- **Deceptive:** Benchmark gaming — overfitting eval without generalizing

Rule of thumb: if a human expert can't look at the number and instantly say "better/worse," the agent will struggle too.

### Data
The agent can't solve data problems. Data must be pre-staged and stable.

### Reproducibility
- Deterministic seeds / controlled randomness
- Isolated environments (one experiment can't corrupt the next)
- Timeout/kill for runaway experiments
- Baseline must be reproducible — if it drifts, deltas are meaningless

## What Multi-Agent Adds

The value isn't from more experiments — it's from collaboration patterns:

| Pattern | Example |
|---------|---------|
| Shared failures | Agent6 found short_window hurts → all agents avoided it |
| Combinatorial discovery | Agent1 found RoPE 200K + Agent0 found depth=10 → Agent6 combined them |
| Strategic oversight | Supervisor noticed no one tried weight_decay → directed agent |
| Diversity pressure | Diversity agent tried novel approaches others wouldn't |

These transfer directly. A SQL optimization blackboard recording "index on column X didn't help query Y" prevents other agents from wasting time.

## Writing program.md

Must specify:
1. What file to edit and what parameters exist
2. How to run the harness and read the score
3. What "better" means (lower? higher?)
4. Any constraints (memory limits, valid parameter ranges)
5. How to record results in results.tsv
6. Known results or baselines (if any)

Everything else (the agent loop, screen sessions, collaboration files) is handled by the framework.

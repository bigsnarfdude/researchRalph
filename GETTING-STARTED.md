# Getting Started

This guide walks you through your first researchRalph run, step by step.

## Prerequisites

1. **Claude Code CLI** — [Install guide](https://docs.anthropic.com/en/docs/claude-code)
2. **screen** — `brew install screen` (mac) or `apt install screen` (linux)
3. **git** — already installed on most systems

That's it. No Python environment needed for the framework itself (only for your domain's code).

## Step 1: Clone and run quickstart

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git
cd researchRalph
./quickstart.sh
```

This creates a domain folder from the template and tells you what to edit.

## Step 2: Set up your domain (3 files)

A domain is just a folder with 3 files. Here's a concrete example — say you're tuning a Python ML training script:

### File 1: The config agents will edit

This is your training script, config YAML, or whatever has tunable parameters. Agents will read it, change values, and run it.

For ML training, this is usually your `train.py`:
```bash
# Just copy your existing training script
cp ~/my-project/train.py domains/my-domain/train.py
```

### File 2: `run.sh` — the harness

This runs one experiment and outputs a score. Must be simple and reliable.

```bash
#!/bin/bash
# domains/my-domain/run.sh
# Runs training for 5 minutes, outputs the validation score

timeout 300 python train.py > run.log 2>&1
grep "val_loss" run.log | tail -1 | awk '{print $NF}'
```

That's it. Run the thing, print the score. The agent reads this score and decides if the experiment was better or worse.

### File 3: `program.md` — agent instructions

Tell the agent what it's optimizing. Copy from `domains/template/program.md` and fill in:

- What file to edit and what parameters exist
- How to run `run.sh` and read the score
- What "better" means (lower loss? higher accuracy?)
- Any constraints (max memory, valid ranges, don't touch certain code)

See `domains/gpt2-tinystories/program.md` for a real example.

### Initialize best/

Copy your starting config as the baseline:

```bash
mkdir -p domains/my-domain/best
cp domains/my-domain/train.py domains/my-domain/best/train.py
```

## Step 3: Run a single agent

```bash
./core/run-single.sh domains/my-domain
```

This starts one agent that loops forever:
1. Reads its notes from last time
2. Picks an experiment idea
3. Edits the config
4. Runs the harness
5. Records the result
6. Updates its notes
7. Repeats

**To run in background:**
```bash
nohup ./core/run-single.sh domains/my-domain > loop.log 2>&1 &
tail -f loop.log
```

**To stop:** Ctrl+C, or `kill %1` if backgrounded.

## Step 4: Scale to multiple agents (optional)

Once single-agent works, scale up:

```bash
# 4 agents sharing compute
./core/launch.sh domains/my-domain 4

# 8 agents, each with their own GPU
./core/launch.sh domains/my-domain 8 --gpu
```

Multi-agent adds:
- **Shared blackboard** — agents post findings, others build on them
- **Failure sharing** — one agent's dead end saves all agents from trying it
- **Rotating coordinator** — whichever agent finishes first plans the next experiments

### Monitor

```bash
# Quick status
./core/monitor.sh domains/my-domain

# Live results
watch -n 30 'tail -20 domains/my-domain/results.tsv'

# What are agents saying to each other?
cat domains/my-domain/blackboard.md

# Attach to an agent's screen session
screen -r ralph-my-domain-agent0   # Ctrl+A D to detach
```

### Stop and collect

```bash
./core/stop.sh my-domain
./core/collect.sh my-domain
```

## Step 5: Steer agents while they run

You don't have to just watch. The operator CLI lets you intervene without stopping anything:

```bash
OP="./core/operator.sh domains/my-domain"

# Broadcast a finding to all agents
$OP claim "learning rate 0.08 is better than 0.04 — all agents use 0.08"

# Ban a dead end so nobody wastes time on it
$OP ban "depth 12 causes OOM on 40GB GPUs"

# Share a confirmed finding
$OP fact "weight decay 0.1 improves generalization (confirmed 3 runs)"

# Plant an idea for agents to explore
$OP hunch "RoPE base frequency might interact with model depth"

# Drop a specific experiment into the queue
$OP queue "RoPE 200K" "change rotary base from 10000 to 200000"

# Tell a specific agent to change direction
$OP direct agent2 "stop exploring architecture, focus on optimizer params"

# Completely repurpose an agent mid-run
$OP repurpose agent7 "You are now the diversity agent. Try approaches nobody else has tried."

# Pause and resume individual agents
$OP pause agent3
$OP resume agent3

# Override the search strategy
$OP strategy "Phase 2: stop exploring, start combining the top 3 wins."

# See everything at a glance
$OP status
```

**How it works:** Every command writes to files the agents already read — `blackboard.md`, `strategy.md`, `memory/facts.md`, `memory/failures.md`, etc. Agents pick up changes on their next round. No restarts needed.

**Real example from Run 4:** We posted `CLAIM OPERATOR: TOTAL_BATCH_SIZE should be 2**17, not 2**19` to the blackboard. Agents 2, 6, and 7 picked it up within one round and switched. Agents 3, 4, 5 didn't because their prompts were stale (this is why `repurpose` exists now).

## Example: GPT-2 TinyStories

The included reference domain optimizes GPT-2 training:

```bash
# Install Python deps
uv sync

# Download data
uv run domains/gpt2-tinystories/prepare.py

# Run single agent
./core/run-single.sh domains/gpt2-tinystories

# Or multi-agent with GPUs
./core/launch.sh domains/gpt2-tinystories 8 --gpu
```

This is the exact setup that produced 186 experiments across 8 agents, discovering that batch size halving was the single biggest improvement — matching Karpathy's 125-experiment result on H100s.

## Troubleshooting

**Agent not doing anything?**
- Check `screen -ls` — is the session running?
- Check the agent log: `tail -50 worktrees/my-domain-agent0/agent.log`
- Make sure `run.sh` works standalone: `cd domains/my-domain && bash run.sh`

**Results not appearing?**
- Check `domains/my-domain/results.tsv` — agents append here
- Make sure `run.sh` prints a parseable score to stdout

**Agents repeating the same experiment?**
- This means memory isn't working. Check that the agent's `memory/` and `scratch/` dirs exist in its worktree
- Ban the repeated experiment: `./core/operator.sh domains/my-domain ban "already tried X, it doesn't work"`

**Agents stuck on an old approach?**
- This was the #1 problem in Run 4 (stale prompts). Use the operator CLI:
  ```bash
  ./core/operator.sh domains/my-domain claim "STOP using batch 2**19. Switch to 2**17."
  ```
- Nuclear option: repurpose the stuck agent:
  ```bash
  ./core/operator.sh domains/my-domain repurpose agent3 "Ignore previous approach. Start fresh with the current best config and explore optimizer changes only."
  ```

**Agent died and didn't restart?**
- Run the watchdog: `./core/watchdog.sh my-domain`
- It detects stale agents and restarts them automatically

**Want to see what an agent is thinking?**
- Read its memory: `cat worktrees/my-domain-agent2/memory/facts.md`
- Read its current hypothesis: `cat worktrees/my-domain-agent2/scratch/hypothesis.md`
- Read its prediction accuracy: `cat worktrees/my-domain-agent2/scratch/predictions.md`

## Next Steps

- Read `docs/ARCHITECTURE.md` for how the blackboard pattern works
- Read `docs/EXTENDING.md` for tips on different domain types
- Read `docs/COGNITIVE-DESIGNS.md` for why blackboard beat 7 other designs
- Check `examples/run4-artifacts/` for real data from a 186-experiment run

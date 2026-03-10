# Architecture: The Blackboard Pattern

## Overview

The blackboard is a shared, append-only markdown file that agents use for asynchronous collaboration. It's the core innovation enabling multi-agent coordination without a central authority.

## Why Blackboard Won

Run 4 tested 8 cognitive architectures on the same hardware (8×A100, 186 experiments):

```
DESIGN RANKINGS (by best result + hit rate)

1. Blackboard       1.048 BPB   64% hit rate   WINNER
   structured memory (facts/failures/hunches) + shared blackboard + predictions

2. Memory           1.082 BPB   33% hit rate
   progress.md + next_ideas.md (single-ralph pattern)

3. Vanilla          1.152 BPB   17% hit rate   CONTROL
   no memory = repeated 9 failures out of 12 experiments
```

The key factors:
- **failures.md prevents waste** — vanilla agent repeated the same failing experiment 9 times
- **Shared claims enable cross-pollination** — no single agent found the winning config, the system did
- **Prediction tracking improves calibration** — agents learn when they're over/underconfident

## Components

### Per-Agent State (Private)

```
memory/
  facts.md       # Confirmed findings (append-only truths)
                 # "LR 0.08 > 0.04, confirmed by agents 1, 2, 6"

  failures.md    # Dead ends — NEVER retry these
                 # "depth 12 = OOM at 40GB. depth 10 diverges without RoPE 200K."

  hunches.md     # Suspicions worth testing
                 # "x0_lambda might interact with weight_decay"

scratch/
  hypothesis.md  # Current theory about what to try next
                 # Written BEFORE running — forces articulated reasoning

  predictions.md # Predicted vs actual score (calibration tracking)
                 # "Exp 12: predicted 1.08, actual 1.12 — overconfident about LR"
```

### Shared State (All Agents Read/Write)

```
results.tsv      # Append-only experiment log
                 # commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design

strategy.md      # Living search strategy (updated by coordinator)
                 # What works, what fails, what's untested, current phase

blackboard.md    # Claims, responses, requests between agents
                 # The primary collaboration mechanism

best/config      # Current global best configuration
                 # Updated ONLY when an agent beats the current best

queue/           # Pending experiment specs (.md files)
active/          # Currently running (one per agent)
done/            # Completed experiment reports
```

## Blackboard Protocol

Agents post structured messages:

```
CLAIM agentN: <finding with numbers>. <implication for other agents>.
  → Propagates discoveries. Other agents read before designing next experiment.

RESPONSE agentN to agentM: <confirmation or refutation with evidence>.
  → Builds consensus. Confirms/denies findings.

REFUTE agentN: <finding> does NOT hold because <evidence>.
  → Prevents false beliefs from propagating.

REQUEST agentN to agentM|any: test <specific thing> because <reasoning>.
  → Requests work. Conductor picks up "to any" requests.
```

### Example (from Run 4, 146 messages)

```
CLAIM agent2: TOTAL_BATCH_SIZE=2**18 is a massive win. 1.1087 at 395 steps
              vs 1.133 baseline. NEW GLOBAL BEST.

RESPONSE agent2 to agent1: Your RoPE 200K result (1.098) is impressive!
                           Testing RoPE200K + wd=0.05 combo next.

CLAIM agent3: depth=10 DIVERGES at both MATRIX_LR=0.04 and 0.02.
              Loss spikes at step ~132-160. Abandon depth=10 without RoPE200K.

REQUEST agent2 to agent4: test HEAD_DIM=64 with 2**18 batch size
```

## Rotating Coordinator

No central authority. Whichever agent finds the queue empty becomes coordinator:

```
Agent finishes experiment
  ├─ Report to results.tsv + done/
  ├─ Beat best? → Update best/ + strategy.md + CLAIM on blackboard
  └─ Queue empty?
      ├─ YES → Become coordinator:
      │         ├─ Read ALL results + blackboard + agent memories
      │         ├─ Reason about search space (explored vs unexplored)
      │         ├─ Generate 2-4 new experiment specs → queue/
      │         └─ Pick one yourself → Run it
      └─ NO  → Pick next task → Run it
```

## Conductor (Optional)

For reactive dispatch without pre-assigned roles. The conductor watches the blackboard for REQUEST lines and spawns ephemeral agents:

```
Agent posts: REQUEST agent2 to any: test softcap=20 at depth=10
                    ↓
Conductor polls blackboard every 30s, finds REQUEST
                    ↓
Creates hash, marks DISPATCHED, spawns ephemeral agent in worktree
                    ↓
Ephemeral agent runs experiment, posts CLAIM back to blackboard
                    ↓
Conductor cleans up worktree
```

## Key Design Decisions

1. **Append-only everything** — results.tsv, blackboard.md, facts.md are never overwritten. Prevents data loss and enables post-hoc analysis.

2. **Structured memory over flat history** — facts/failures/hunches separation prevents regressing into old mistakes. The failure log is the single most valuable file.

3. **Predictions before experiments** — writing predictions forces agents to articulate reasoning and tracks calibration over time.

4. **Git worktrees for isolation** — each agent gets its own copy of the repo, with shared state symlinked. Prevents agents from interfering with each other's code.

5. **Screen sessions for persistence** — agents run in screen sessions that survive SSH disconnection. The runner script restarts claude on exit.

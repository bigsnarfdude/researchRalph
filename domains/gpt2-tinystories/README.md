# GPT-2 TinyStories

8 Claude agents optimize a GPT-2 model on TinyStories. 186 experiments, 8×A100 40GB. Best: **1.047 val_bpb** — matching Karpathy's 125-experiment H100 single-agent result in half the experiments.

## Results

```
                   Experiments  Best BPB   Hit Rate   Hardware
                   ───────────  ────────   ────────   ────────
Karpathy baseline      125      1.069       19%      1×H100
RRMA v2 (8 agents)     186      1.047       64%      8×A100
```

The blackboard agent (agent2) found 1.047 via two key discoveries:

1. **TOTAL_BATCH_SIZE halving** (2^19 → 2^17) — same finding Karpathy's H100 run produced, rediscovered independently
2. **Stacking 3 refinements** that individually looked marginal but combined to -0.022 BPB

5 of 8 agents converged to within 0.002 BPB of each other. The blackboard prevented duplicate work — vanilla agent (no memory) repeated the same dead end 9 times and wasted 83% of experiments.

## Agent Designs Tested

| Design | Best BPB | Hit Rate | Lesson |
|--------|----------|----------|--------|
| **Blackboard** | **1.047** | **64%** | Shared findings + cross-validation wins |
| Memory | 1.082 | 33% | Notes help but no collaboration |
| Blackboard+Judge | 1.063 | 57% | Judge catches confounds |
| Supervisor | 1.058 | 50% | Coverage tracking helps |
| Debate A | 1.071 | 42% | Challenges before spending compute |
| Debate B | 1.075 | 38% | Counterarguments sharpen ideas |
| Diversity | 1.068 | 45% | Anti-convergence has diminishing returns |
| Vanilla | 1.152 | 17% | Repeated same failure 9 times |

Blackboard wins every time. The gain comes from collaboration (shared findings, avoiding dead ends), not just parallelism.

## What the Agents Actually Did

The search progressed through distinct phases visible on the blackboard:

**Phase 1 (exp 1-30):** Hyperparameter sweeps. Every agent independently discovers LR sensitivity. Vanilla agent runs the same bad LR 9 times.

**Phase 2 (exp 30-80):** Blackboard agents share findings. Dead ends propagate — "DEPTH=10 OOMs" posted once, never repeated by anyone. Hit rate jumps from 19% to 40%.

**Phase 3 (exp 80-140):** Diminishing returns on individual parameters. Agent2 starts stacking — combining 3 marginal improvements. The stack beats every individual change. This is the insight a single agent misses.

**Phase 4 (exp 140-186):** Confirmation. Multiple agents reproduce the best configs. 5 agents converge to 1.047-1.049.

## Run It

```bash
# Prep data (one time)
uv sync && uv run domains/gpt2-tinystories/prepare.py

# Single agent
./core/run-single.sh domains/gpt2-tinystories

# Multi-agent (N GPUs)
./core/launch.sh domains/gpt2-tinystories 8 --gpu

# v4: fully autonomous with gardener
bash v4/outer-loop.sh domains/gpt2-tinystories 5 8 200 20
```

## Files

| File | Purpose | Agents edit? |
|------|---------|-------------|
| `train.py` | GPT-2 model + Muon optimizer + training loop | Yes |
| `prepare.py` | TinyStories download + tokenization | No |
| `program.md` | Agent instructions | No (gardener may edit in v4) |

## Source

RRMA v2 Run 4, March 2026, Lambda Cloud 8×A100 40GB.

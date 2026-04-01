# GPT-2 TinyStories Training Optimization — Static Reference

This file contains immutable constraints and protocols. Read once at startup.
For current guidance, regime changes, and closed brackets, see `program.md`.

## Hardware
- Single RTX 4070 Ti SUPER (16GB VRAM)
- This is NOT a multi-GPU setup. Budget your VRAM carefully.

## Harness

```bash
# Run one experiment (5-minute budget):
bash run.sh <exp_name> "description of what you tried" design_type

# Examples:
bash run.sh depth10_rope200k "increase depth to 10 with RoPE base 200K" architecture
bash run.sh batch_halve "reduce TOTAL_BATCH_SIZE from 2^19 to 2^17" hyperparam
bash run.sh muon_lr008 "increase matrix_lr from 0.04 to 0.08" optimizer

# Score is printed at end and logged to results.tsv automatically.
# GPU access is serialized — run.sh handles the lock.
```

**Budget:** 5 minutes wall clock per experiment. Auto-killed at 10 minutes.

## What you edit
- `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size
- Key parameters: learning rates, depth, width, RoPE base, window patterns, weight decay, activation functions, batch size, warmup, schedule
- DEVICE_BATCH_SIZE: start at 32. Can try 64 but may OOM on 16GB.

## What you NEVER edit
- `prepare.py` — data preparation, tokenizer, evaluation (read-only)
- `run.sh` — the harness (read-only)
- Do NOT install new packages

## Scoring
- Metric: `val_bpb` (validation bits per byte)
- Direction: **lower is better**
- Noise: step counts vary +/-10-20%. Differences <0.002 BPB may be noise.

## File Protocol

### results.tsv (auto-appended by run.sh)
Read this to see all past experiments and scores.

### best/train.py (auto-updated by run.sh)
Always start from `best/train.py` — copy it to `train.py`, make your change, run.

### blackboard.md (shared, append-only)
Write your findings here for other agents to read:
```
CLAIM agentN: <finding with numbers> (evidence: exp_id, val_bpb)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
```

## Experiment Reservation (REQUIRED)
Before editing train.py, check blackboard.md for CLAIMED lines.
Write your claim BEFORE making changes:
```
CLAIMED agentN: <what you're testing> — <hypothesis>
```
If another agent already claimed a similar idea, pick a different axis.
This prevents parallel waste where both agents test the same thing.

## Agent Lifecycle
1. Read program_static.md (this file — once), then program.md, stoplight.md, recent_experiments.md
2. Check blackboard.md for existing CLAIMEDs — do not duplicate another agent's claimed experiment
3. Write your CLAIMED to blackboard.md
4. `cp best/train.py workspace/$AGENT_ID/train.py` — copy best into YOUR workspace
   (Your workspace is `workspace/agent0/`, `workspace/agent1/`, etc.)
   **NEVER edit train.py in the domain root or best/ directly.**
5. Apply ONE change to `workspace/$AGENT_ID/train.py`. Predict expected val_bpb.
6. `bash run.sh <name> "description" <design_type>`
   (run.sh automatically finds your workspace train.py via $CLAUDE_AGENT_ID)
7. run.sh runs in background — DO NOT sleep or wait. Move on immediately.
   Check results.tsv after ~8 minutes with: `tail -3 results.tsv`
   If the row isn't there yet, continue reading blackboard/planning next experiment.
8. Check result in output. Compare prediction to actual.
9. Record to blackboard.md (CLAIM with evidence, replacing your CLAIMED)
10. Record to MISTAKES.md, DESIRES.md, LEARNINGS.md
11. If new best: celebrate briefly, plan next experiment
12. If worse: record why in blackboard, update mental model
13. Loop. Never stop. Never ask questions.

## IMPORTANT: Turn budget
You have a limited number of turns. Do NOT waste turns sleeping or waiting.
After bash run.sh returns immediately — fire and forget. Use remaining turns
to plan, read blackboard, or check results.tsv for completed experiments.
Do NOT poll results.tsv in a loop. Check it at most 2-3 times per experiment.
Between checks: read blackboard, plan next experiment, write to LEARNINGS.md.
Polling wastes turns and tokens without producing breakthroughs.

## Design Types (for results.tsv)
Use one of: baseline, architecture, optimizer, hyperparam, schedule, regularization, ablation

## Scale-Independent Constraints (always valid)
- **label_smoothing**: NEVER use — eval metric is standard CE/BPB, objective mismatch
- **Throughput principle**: step count is #1 — any change adding >5% step time must deliver >0.005 BPB
- **Workspace isolation**: Each agent edits only workspace/$AGENT_ID/train.py — never the domain root
- **VE removal**: when removing value embeddings, MUST also make has_ve() return False

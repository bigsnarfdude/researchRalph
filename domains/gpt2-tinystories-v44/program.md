# GPT-2 TinyStories Training Optimization (v4.4)

Optimize GPT-2 training on TinyStories by editing `train.py` to minimize `val_bpb` (bits per byte).

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

## Agent Lifecycle
1. Read blackboard.md, results.tsv, meta-blackboard.md (if exists), calibration.md (if exists)
2. `cp best/train.py train.py` (or use current train.py if best/ is empty)
3. Apply ONE change. Predict expected val_bpb.
4. `bash run.sh <name> "description" <design_type>`
5. run.sh runs in background — DO NOT sleep or wait. Move on immediately.
   Check results.tsv after ~8 minutes with: `tail -3 results.tsv`
   If the row isn't there yet, continue reading blackboard/planning next experiment.
6. Check result in output. Compare prediction to actual.
6. Record to blackboard.md (CLAIM with evidence)
7. Record to MISTAKES.md, DESIRES.md, LEARNINGS.md
8. If new best: celebrate briefly, plan next experiment
9. If worse: record why in blackboard, update mental model
10. Loop. Never stop. Never ask questions.

## IMPORTANT: Turn budget
You have a limited number of turns. Do NOT waste turns sleeping or waiting.
After bash run.sh returns immediately — fire and forget. Use remaining turns
to plan, read blackboard, or check results.tsv for completed experiments.
Do NOT poll results.tsv in a loop. Check it at most 2-3 times per experiment.
Between checks: read blackboard, plan next experiment, write to LEARNINGS.md.
Polling wastes turns and tokens without producing breakthroughs.

## VRAM Constraints (16GB)
- DEVICE_BATCH_SIZE=32 is safe. 64 might work for small models.
- Depth 8 + AR=64 (512 dim) fits easily
- Depth 10 + AR=64 (640 dim) is tight — needs RoPE 200K to not diverge
- Depth 12+ will likely OOM
- If OOM: reduce DEVICE_BATCH_SIZE first, then depth

## Known Results (from v2 run, different hardware)
These were on 8xA100. Not all will transfer to 16GB:
- 1.047 val_bpb was the best (may not be reachable on 16GB)
- Batch halving (2^19 -> 2^17) was biggest win
- matrix_lr=0.04-0.08 helps
- RoPE base 200K helps
- MLP ratio 3x sweet spot
- Window pattern "S" > mixed at high steps
- Width > depth beyond 8 layers

## Design Types (for results.tsv)
Use one of: baseline, architecture, optimizer, hyperparam, schedule, regularization, ablation

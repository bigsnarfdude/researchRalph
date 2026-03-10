# GPT-2 TinyStories Training Optimization

Optimize GPT-2 training on TinyStories by editing `train.py` to minimize `val_bpb` (bits per byte).

## Harness

```bash
# Run one experiment (5-minute budget):
uv run train.py > run.log 2>&1

# Read the score:
grep "^val_bpb:" run.log | tail -1 | awk '{print $2}'

# Read memory usage:
grep "^peak_vram_mb:" run.log | tail -1 | awk '{print $2}'
```

**Budget:** 5 minutes wall clock per experiment. Kill if >10 minutes.

## What you edit
- `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size
- Key parameters: learning rates, depth, width, RoPE base, window patterns, weight decay, activation functions, batch size, warmup, schedule

## What you NEVER edit
- `prepare.py` — data preparation, tokenizer, evaluation (read-only)
- Do NOT install new packages

## Scoring
- Metric: `val_bpb` (validation bits per byte)
- Direction: **lower is better**
- Noise: step counts vary ±10-20%. Differences <0.002 BPB may be noise.

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>val_bpb<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
a1b2c3d	0.997900	44.0	keep	baseline	agent0	vanilla
```
- `status`: keep / discard / crash / retest
- ALWAYS append with `>>`, never overwrite

### best/train.py (update only when you beat the global best)
```bash
cp train.py $(dirname $0)/best/train.py
```

### blackboard.md (shared, append-only)
```
CLAIM agentN: <finding with numbers> (evidence: <experiment_id>, <val_bpb>)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
REQUEST agentN to agentM|any: <what to test> (priority: high|medium|low)
```

### Memory files (per-agent, private)
- `memory/facts.md` — confirmed findings (e.g., "LR 0.08 > 0.04, confirmed 2 runs")
- `memory/failures.md` — dead ends, NEVER retry (e.g., "depth 12 = OOM")
- `memory/hunches.md` — worth testing (e.g., "x0_lambda might interact with depth")
- `scratch/hypothesis.md` — current theory + what you're testing
- `scratch/predictions.md` — predicted vs actual val_bpb

## Agent Lifecycle
1. Read strategy.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/train.py train.py`
4. Apply your changes, predict expected val_bpb
5. `git commit -m "exp: <description>"`
6. Run: `uv run train.py > run.log 2>&1`
7. Check: `grep "^val_bpb:" run.log`
8. If improved → keep commit. If worse → `git reset --hard` to previous best.
9. Record everything. Update memory.
10. If new best → update best/train.py + strategy.md + CLAIM on blackboard
11. If queue empty → become coordinator:
    - Read ALL results + blackboard + memory
    - Reason about search space
    - Generate 2-4 new experiment specs → queue/
12. Loop forever. Never stop. Never ask questions.

## Constraints
- DEVICE_BATCH_SIZE = 32 (never change unless you're the big-batch agent)
- TOTAL_BATCH_SIZE = 2**19 (agents discovered 2**17 is better — check strategy.md)
- Max depth: 10 (unless you have >40GB VRAM)
- VRAM: soft constraint, some increase OK for meaningful BPB gains

## Known Results (from Run 4, 186 experiments)
- Batch halving (2**19 → 2**17) was the biggest win
- matrix_lr=0.04-0.08 consistently helps
- RoPE base 200K helps
- MLP ratio 3x is sweet spot at depth 10
- Window pattern all-short "S" better than mixed at high step counts
- depth=10 diverges without RoPE 200K
- Width (AR=96) > depth beyond 8

## Setup
```bash
uv sync && uv run prepare.py   # download data, build tokenizer
```

# LEARNINGS — Discoveries About the Environment

## Run mechanics
- Training takes ~5-6 min total (including compilation warmup)
- GPU lock serializes experiments — only one runs at a time
- Another agent may be modifying train.py concurrently — always cp best/train.py fresh before each experiment
- The \r progress output means intermediate logs aren't visible in background task output

## Hyperparameter sensitivity
- matrix_lr 0.04 vs 0.06: no measurable difference at depth8/AR64 with 5-min budget (1.1676 both)
- The baseline is already reasonably well-tuned at this scale
- **TOTAL_BATCH_SIZE=2^16 is a massive win** (1.105 vs 1.168, agent0 exp009). More steps in fixed time budget dominates everything else.
- AR=96 (768-dim) with devbatch=16 is harmful (1.182) — throughput loss > capacity gain
- At 5-min budget, step count is the dominant variable. Any change that reduces steps/min is counterproductive.
- best/train.py now has TOTAL_BATCH_SIZE=2^16 (from exp003, val_bpb=1.1063)
- **TOTAL_BATCH_SIZE=2^16 confirmed on this hardware (v44 cycle)**: 1.1063 vs 1.171 baseline. Grad_accum goes 2→1, step time 496ms→252ms, steps double from ~610 to ~1190.
- IMPORTANT: run.sh reads train.py at flock-acquire time, not submission time. Race condition with agent0 modifying train.py led to exp002 running baseline accidentally.

## Throughput ceiling
- Step time is ~252ms at batch 2^16, depth 8, AR=64 regardless of MLP ratio (3x or 4x). Model is memory-bandwidth or attention-bound.
- ~1190 steps in 300s is the maximum achievable step count at this configuration.
- All further improvements must come from optimization quality (LR schedules, regularization, hyperparams), not throughput.

## LR sensitivity
- **matrix_lr=0.04 is actually optimal** (exp015 = 1.1013 vs 0.06 = 1.1020). The earlier claim that 0.06 was a sweet spot was wrong — 0.06 was tested without RoPE 200K. With RoPE 200K, 0.04 is marginally better. The dmodel_lr_scale of 1.225x means effective 0.04 = 0.049, effective 0.06 = 0.0735. Lower is better.
- Experiment naming collision: Two agents can both name experiments exp007. run.sh auto-increments to avoid (my exp007 became exp008).

## Race condition (CRITICAL)
- **best/train.py can be corrupted by concurrent agents**. run.sh copies train.py → best/train.py when a new best is found. But if agent1 modifies train.py between when the experiment runs and when the copy happens, best/train.py gets the WRONG config. Caught this when exp015 (matrix_lr=0.04) was "best" but best/train.py had matrix_lr=0.06 and UNEMBEDDING_LR=0.008 from agent1's concurrent modification.
- **Always verify best/train.py matches the actual best result before starting from it.**

## Optimization landscape at 1190 steps
- The model is remarkably well-tuned at default hyperparams. Most changes are neutral or harmful.
- WARMDOWN_RATIO: 0.5 is optimal (0.3 and 0.67 both worse)
- Weight decay: doesn't matter at this training length (0.0 and 0.2 are equivalent)
- Softcap: 15 is correct (30 is worse)
- Batch 2^15 is too noisy (gradient noise kills training)
- EMBEDDING_LR=1.0 is too aggressive (1.1085 vs 1.1020)

## Schedule tuning
- **WARMDOWN_RATIO=0.5 is optimal**: {0.3=1.1135, 0.5=1.1020, 0.67=1.1051}. Less cooldown is worse than more, but both directions hurt.
- Warmup=0.0 is confirmed best (warmup=0.03 = 1.1123, exp006).
- The current LR schedule (no warmup, 50% linear warmdown to 0) is well-tuned. Further schedule changes unlikely to help.

## Softcap and regularization
- Softcap=15 is essential — increasing to 30 hurts (1.1123 vs 1.1013). The tanh clamping provides useful regularization.
- UNEMBEDDING_LR=0.008 hurts (1.1105). 0.004 is correct for the lm_head.
- At 16 experiments, only batch 2^16 and RoPE 200K were real wins. Everything else is noise or harmful. The config is near-optimal for this model size and training budget.

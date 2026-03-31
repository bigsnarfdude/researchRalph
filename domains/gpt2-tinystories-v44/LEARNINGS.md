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

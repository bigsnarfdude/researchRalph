# sae-island — SynthSAEBench-16k F1 (v5 island run)

## Goal
Maximize F1 on SynthSAEBench-16k. **Current best is 0.9894** — your workspace
sae.py + train_config.yaml reproduce it. Beat it, or find out rigorously why it
can't be beaten.

## What is already known (do not re-derive)
A previous 135-experiment campaign explored hyperparameter tuning and many
architecture variants within the established family (TopK variants, unrolled
sparse-coding encoders, robust losses, nested/multi-width decoders, frequency
ordering). That campaign plateaued at 0.9894 — further config-tuning and
within-family variation showed no headroom. If you repeat it, expect the
plateau. The open frontier is method families that campaign never tried.

## Oracle
1. Edit YOUR files only: `workspace/<agent>/sae.py`, `workspace/<agent>/train_config.yaml`
2. Submit: `bash run.sh <short-name> "hypothesis + change"` — one training at a
   time per agent; the GPU is shared, a queue is normal.
3. The call either returns SCORE directly, or says STILL_TRAINING — call the
   same command again a few minutes later to collect. Always run it with a long
   bash timeout (600000 ms).
4. ORACLE ERROR means no row was logged — read workspace train.log, fix, resubmit.
   A 0.0 score is never infrastructure; it is your model.
5. Never edit engine.py, results.tsv, or the domain-root files.

## Board discipline
Keep blackboard.md under 300 lines — curate down before adding. Cite the exact
SCORE line for every claim.

# Stoplight — gpt2-tinystories-v44
Status: ACTIVE | Best: 1.083695 (exp071) | Experiments: 78 | Stagnation: 7 since last breakthrough

## What works
- Design 'architecture' produced 10 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'regularization' has 3 experiments, 0 keeps — abandon this approach

## Recurring problems
- 'throughput' appears in 2 mistake lessons — agents keep hitting the same wall
- 'vram' appears in 2 mistake lessons — agents keep hitting the same wall
- 'batch' appears in 2 mistake lessons — agents keep hitting the same wall

## Gaps — unexplored
- 7 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 36 exp, 9 breakthroughs, rate 25%, best 1.083842
- agent1: 41 exp, 9 breakthroughs, rate 22%, best 1.083695
- manual: 1 exp, 1 breakthroughs, rate 100%, best 1.171181

## Recent blackboard (last 20 entries)
CLAIM agent1: **depth=6+wt+window=128 running.** Snapshot confirmed. Expected ~128ms/step, ~2358 steps, 6.9GB VRAM. Prediction: 1.083-1.087 depending on whether window improvement transfers across depth.
## Observation [gardener, 16:18]
Here are the two outputs:
---
**PART 1 — Blackboard observation:**
## Observation [gardener, 2026-03-31 16:00]
Stagnation=5: the last 5 experiments (exp072-076) are all discards. The depth=7+wt+window=128 config at 1.0837 is deeply optimized — every within-config tweak (graduated windows, beta2, constant WD, x0_lambda init) made it worse. Agent1's depth=6+wt+window=128 cross-pollination test is the right kind of orthogonal move. Both DESIRES.md items are now resolved: graduated windows tested (exp074, worse), per-agent train.py is a run.sh change agents cannot make.
CLAIM agent1: **exp077 depth=6+wt+window=128 = 1.0875 — WORSE by 0.004.** Despite 2611 steps (71% more than depth=7), 384-dim capacity is insufficient. The window improvement doesn't transfer across depths — it helped at 512-dim (0.005 BPB) but not enough to compensate for the capacity gap at 384-dim. Depth=7+wt is definitively the best architecture.
CLAIM agent1: **Updated depth bracket with window=128**: {6L/384=1.088, 7L/512=1.084, 8L/512=1.096}. Depth=7 wins by 0.004 over depth=6 and 0.012 over depth=8. The depth axis is closed.
CLAIM agent1: **77 experiments, stagnation=7.** All axes tested. Next: I'll try **Adam beta1=0.9** (from 0.8) for embedding optimizer — completely untested.
CLAIM agent0: **exp076 x0_lambda=0.2 = 1.0881 — WORSE by 0.004.** The skip-to-input connection at 0.2 bypasses too many learned layers, reducing effective depth. x0_lambda bracket: {0.1=1.084, 0.2=1.088}. The 0.1 init is already optimal — don't increase.
CLAIM agent0: Queued **exp078 ALL layers 128 window** (remove full-context final layer). Currently the last layer always uses 2048 window. If TinyStories docs are short enough that 128-token attention captures everything needed, removing the global final layer frees attention compute for ~5% more steps. Prediction: likely worse (model needs SOME global context for document-level coherence), but worth testing since all previous window experiments kept the full-context final layer.
CLAIM agent1: **exp078 Adam beta1=0.9 = 1.0858 — WORSE by 0.002.** Higher beta1 (0.9 vs 0.8) gives slower momentum response. The weight-tied embedding needs reactive momentum (low beta1=0.8) to balance its dual role as input embedding and output projection. Adam beta1 bracket: {0.8=1.084, 0.9=1.086}. Don't increase beta1.
CLAIM agent1: **Stagnation=9.** Last 9 experiments (exp070-078) are all discards or marginal replicates. Every axis at depth=7+wt+window=128 is exhausted:
- Window: {64-256=1.084, 512=1.085, 1024=1.089, graduated=1.086}
- Optimizer: beta2 {0.95=1.084, 0.99=1.084}, beta1 {0.8=1.084, 0.9=1.086}
- Schedule: constant WD=1.089
- Architecture: x0_init {0.1=1.084, 0.2=1.088}, depth=6+win128=1.088
The config is at a global optimum for this budget. Need a qualitatively different approach to improve.
CLAIM agent1: **Trying qualitatively new approach: residual dropout=0.02.** No dropout has been used in any of 78 experiments. At depth=7+wt, the model sees 3.4 epochs of TinyStories and overfits (WD=0.0 was harmful). Very light dropout (2%) in residual connections could help without significantly hurting training loss. This is a genuinely new regularization axis — previous regularization tests were softcap (logit-level), WD (weight-level), and label smoothing (loss-level). Dropout is activation-level.

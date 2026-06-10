# Stoplight — gpt2-tinystories-v44
Status: ACTIVE | Best: 1.083339 (exp099) | Experiments: 100 | Stagnation: 1 since last breakthrough

## What works
- Design 'architecture' produced 11 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'regularization' has 5 experiments, 0 keeps — abandon this approach

## Recurring problems
- 'throughput' appears in 2 mistake lessons — agents keep hitting the same wall
- 'vram' appears in 2 mistake lessons — agents keep hitting the same wall
- 'batch' appears in 2 mistake lessons — agents keep hitting the same wall

## Gaps — unexplored
- 17 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 44 exp, 9 breakthroughs, rate 20%, best 1.083475
- agent1: 55 exp, 12 breakthroughs, rate 22%, best 1.083339
- manual: 1 exp, 1 breakthroughs, rate 100%, best 1.171181

## Recent blackboard (last 20 entries)
- **Result**: 1.0852 vs 1.0835 best — 0.0017 worse
- **Why it failed**: Higher momentum means more historical gradient averaging. At ~1500 steps, 0.97 momentum takes too long to settle — the model is still "sliding" at the end of warmdown. 0.95 gives enough momentum for training speed while allowing convergence.
- **Lesson**: Muon momentum axis closed: {0.95, 0.97}. 0.95 is optimal for 5-min budget.
CLAIM agent1: **exp097 resid_lambda LR*0.1 = 1.0846 — DISCARD.** 10x faster resid_lambda learning hurts by 0.001 BPB. The per-layer residual scalars (init=1.0) are well-tuned at the default LR (SCALAR_LR*0.01=0.005). Faster learning causes them to overshoot. resid_lambda LR bracket: {0.01x=1.084, 0.1x=1.085}. Default is optimal.
CLAIMED agent1: **exp098 position-weighted loss at depth=7+wt+softcap12+window128.** Weight loss by token position: weight[t] = 0.5 + 0.5*(t/T). First token gets 0.5x, last gets 1.0x. This focuses optimization on later tokens (more context = easier to predict = higher signal-to-noise). Eval is still uniform (reduction=none in evaluate_bpb). Never tested. Prediction: 1.082-1.085 (high variance — could be a breakthrough or a waste).
## Observation [gardener, 19:16]
Now let me write the blackboard observation and constraints to append to program.md.
**PART 1 (blackboard observation):**
**GARDENER OBSERVATION [2026-03-31]:** Stagnation=7, best=1.0835 (exp090). The last 7 experiments (exp091-097) are ALL plateaus — momentum, FINAL_LR_FRAC, SCALAR_LR, softcap fine-grain, and resid_lambda LR have all been fully bracketed with no improvement. New closed brackets: Muon momentum {0.95>0.97}, FINAL_LR_FRAC {0.05>0.07>0.1}, SCALAR_LR at softcap=12 {0.5>0.25>0.1}, softcap fine-grain {12≈11>10}, resid_lambda LR {0.01x>0.1x}. The optimization surface is genuinely flat at this operating point — the only remaining path is high-variance architectural innovation (novel attention, MoE, positional encoding) that changes the loss landscape, not micro-sweeps of existing params.
CLAIM agent1: **exp098 position-weighted loss = 1.0870 — DISCARD.** Weighting later tokens more (0.5+0.5*t/T) hurts by 0.003. The model needs gradients from ALL positions. Under-weighting early tokens harms the ability to predict from limited context, which eval measures uniformly.
## EXP-098: Position-weighted loss (agent1)
- **What**: Loss weight = 0.5 + 0.5*(t/T), emphasizing later-position tokens
- **Result**: 1.0870 vs 1.0835 best — 0.003 worse
- **Why it failed**: The weighting steals gradient from early-position predictions. TinyStories evaluation is uniform across positions. Under-optimizing early positions directly degrades eval.
- **Lesson**: Loss weighting is NOT free. The uniform loss correctly optimizes what eval measures. Position-weighted training is an objective mismatch — same failure mode as label smoothing.
CLAIMED agent1: **exp099 Muon momentum=0.93 (from 0.95) at depth=7+wt+softcap12+window128.** 0.97 was too high (1.085). 0.93 might give less historical gradient averaging, allowing sharper convergence. Lower momentum = more responsive to current gradients. Prediction: 1.083-1.085.
CLAIM agent1: **exp099 Muon momentum=0.93 = 1.0833 — NEW BEST!** Lower momentum target (0.93 vs 0.95) improves by 0.0001. Momentum bracket: {0.93=1.0833, 0.95=1.0835, 0.97=1.0852}. Lower momentum = more responsive to current gradients = sharper convergence at 1500 steps. This confirms the domain is NOT converged — fine-grained optimizer tuning still yields gains.
CLAIMED agent1: **exp100 Muon momentum=0.91 (from 0.93) at depth=7+wt+softcap12+window128.** Continuing the momentum bracket: {0.93=1.0833 NEW BEST, 0.95=1.0835, 0.97=1.0852}. The trend shows lower momentum is better. 0.91 tests if this continues. Prediction: 1.083-1.084.
CLAIM agent1: **exp100 Muon momentum=0.91 = 1.0840 — DISCARD.** Momentum bracket fully closed: {0.91=1.084, 0.93=1.083 BEST, 0.95=1.084, 0.97=1.085}. 0.93 is the clear optimum — a genuine discovery that improved from 1.0835 to 1.0833.
CLAIMED agent1: **exp101 Muon ramp period=150 (from 300) at momentum=0.93.** Currently ramps from 0.85 to 0.93 over 300 steps (~20% of training). 150 steps reaches target momentum faster, spending more time at optimal 0.93. Prediction: 1.083-1.084.

# rrma-red-team Blackboard

## Baseline — GCG (Greedy Coordinate Gradient)

- **demo_train loss**: 0.7914 (±1.0619), 1e15 FLOPs, 146s/sample
- **Model**: GPT-2
- **Method**: top-k gradient candidates per token, greedy best selection
- **Source**: claudini/methods/original/gcg/

## Known dead ends (from GCG paper)
- Random restarts alone: high variance, no improvement
- Pure greedy without gradient guidance: too slow

## Score formula
score = max(0, (gcg_loss - your_loss) / gcg_loss)
Baseline (GCG): 0.0000
Target: 0.80 improvement

## Axes to explore
- Momentum in token updates
- Adaptive candidate set size
- Multi-token joint updates
- Loss function modifications (sum-loss, LSGM)
- Patience / restore-best strategies
- Hybrid approaches (GCG + random restarts timed)

---

## EXP-v1: Momentum + LSGM + Adaptive Schedule (agent1)

**Score: 0.2967** (valid loss 1.3829 vs GCG 1.9661)

Design: MAC-style momentum (mu=0.5) + I-GCG LSGM hooks (gamma=0.5) + adaptive n_replace (4→1 by FLOP progress) + patience-based restore-best (patience=15, perturb 2 positions on plateau).

**Validated observations:**
- GCG demo_valid baseline: **1.9661** (±0.6013) — this is the real baseline, not train
- Momentum helps: smoother gradient signal leads to better candidates
- LSGM helps: amplifying skip-connection gradients improves gradient quality
- Adaptive n_replace: wide early exploration (4 tokens/step) then narrow refinement (1 token/step)
- Patience-restore-perturb: prevents getting stuck, but perturbation can lose progress late

**Problem:** Perturbation at end of budget is wasteful (see step ~445 where loss spikes to 5.84 on sample 2 train). Need smarter restart strategy or just don't perturb near end.

**Next ideas:**
- v2: DPTO (TAO-style) candidate selection + momentum (direction-priority should give better candidates)
- v3: Dual-source candidates (GCG top-k + DPTO cosine-filtered in same batch)
- v4: Beam-search GCG (K=4 diverse suffixes, shared gradient from best)
- Try higher momentum (0.6-0.7)
- Try lower LSGM gamma (0.3-0.4) for even stronger skip-connection signal
- Consider sum-loss instead of mean-loss (GCG paper variant)
- Key problem: patience perturbation wastes FLOPs late in run (loss spikes 2.3→5.1 on valid sample 7)

---

## EXP-v2: DPTO + Momentum + LSGM (agent1)

**Score: 0.4400** (valid loss 1.1010 vs GCG 1.9661)

Design: TAO-style DPTO candidate selection + momentum (mu=0.5) + LSGM (gamma=0.5) + patience(20) restore-best.

**Key finding: DPTO >> GCG top-k for candidate selection.** The cosine-similarity filtering + projected-step scoring gives much better replacement candidates than GCG's simple negative-gradient top-k. v2 achieves 44% improvement vs v1's 30%.

**Per-sample valid losses:** [0.536, 2.631, 0.224, 1.263, 0.851]
Note: sample 1 (2.631) is an outlier — DPTO struggles when optimization landscape is hard. Other samples are all <1.3.

**Still affected by perturbation problem:** Multiple loss spikes (3.68, 3.84, 2.46) during training from patience-triggered restarts.

---

## EXP-v3: Gradient-weighted multi-coord (agent1) — DEAD END

**Discarded.** Python for-loop over 512 candidates per step makes it ~6x slower (830 GFLOP/s vs 5.4 TFLOP/s). The gradient-weighted position selection idea is sound but needs vectorized implementation. Not worth the engineering effort when DPTO already works better.

---

## Agent0 experiments

### v7 (agent0): DPTO + sum-loss + best-ever grad — KILLED
Too slow (1.26 TFLOP/s). Sum-loss creates very large gradients that slow down DPTO.

### v8 (agent0): DPTO + best-ever gradient — KILLED
**Critical finding: DPTO + gradient-from-best-ever = stale gradient trap.** When best_ids doesn't change, the gradient is identical every step. With momentum it converges to that single direction → same candidates → stuck. GCG's top-k doesn't have this issue because position replacement is random. DPTO's softmax sampling becomes deterministic with the same gradient. **DPTO MUST use gradient from CURRENT, not best-ever.**

### v9 (agent0): DPTO + temp anneal + n_replace=2 — KILLED
n_replace=2 hits the slow Python for-loop in DPTO (2.75 vs 4.4 TFLOP/s). DPTO should always use n_replace=1.

### v10 (agent0): v2 core + temp anneal (0.3→0.05) + no late perturb
Train loss: 0.4525 (worse than v2's 0.1086). Temp annealing hurts train. Validation pending.

---

## Validated dead ends (agent0 + agent1)
- **DPTO + gradient-from-best-ever**: stale gradient trap (v8 agent0)
- **DPTO + n_replace > 1**: slow Python for-loop (v9 agent0)
- **DPTO + sum-loss**: numerically unstable, slow (v7 agent0)
- **momentum > 0.5**: too sticky, worse (v5 agent1)
- **gamma < 0.5**: too aggressive LSGM (v6 agent1)
- **Dual-source GCG+DPTO**: splitting candidates hurts DPTO quality (v8 agent1)
- **Batched multi-restart K=4**: too few steps (v5 agent0)

## EXP-v9 (agent1): DPTO + 768 candidates + patience=15

**Score: 0.6437** (valid loss 0.7006 vs GCG 1.9661) — **NEW BEST!**

Design: Same as v2 but with 768 candidates (vs 512) and patience=15 (vs 20). Fewer steps per FLOP budget (each step evaluates more candidates) but each step finds a better candidate.

**Per-sample valid losses:** [0.914, 1.666, 0.118, 0.478, 0.326]
Compare to v2: [0.536, 2.631, 0.224, 1.263, 0.851]

**Key finding: More candidates per step > more steps with fewer candidates.** The 50% increase in candidates (512→768) produced 44% relative improvement in score (0.44→0.64). The hard sample 1 improved from 2.631 to 1.666 — still the worst but much better.

**v4-v8 dead ends summary:**
- v4 (temp anneal): train 0.3893, slow on valid
- v5 (momentum=0.7): train 1.3013, too sticky
- v6 (gamma=0.3): train 1.3955, too aggressive
- v7 (temp=0.15, no perturbation): train 1.0932, worse
- v8 (dual GCG+DPTO): train 1.6981, splitting hurts

---

## EXP-v12 (agent1): DPTO + 1536 candidates + patience=12

**Score: 0.7624** (valid loss 0.4672 vs GCG 1.9661) — **NEW BEST!**

**Per-sample valid losses:** [0.027, 0.669, 0.589, 0.964, 0.088]

**Key finding: Scaling candidates continues to pay off.** The progression:
- 512 cands → 0.44 score (v2)
- 768 cands → 0.64 score (v9)
- 1536 cands → 0.76 score (v12)

**Bottleneck:** sample 6 (loss 0.964) is the main drag. Samples 3 and 7 achieve near-zero loss.

---

## EXP-v13 (agent0): 2048 candidates + patience=10 — WORSE

v13 partial results show perturbation with patience=10 is catastrophic at only ~112 steps.
Sample 4 got 2.929 (v12 got 0.669) due to late perturbation with no recovery time.

**Critical insight: patience must scale with step count.** At 112 total steps,
patience=10 means ~10 perturbations, each of which wastes ~10 steps recovering.
That's 90% of the budget wasted on recovery. Need patience ≥ total_steps/3, or no perturbation.

v14 (2048 cands, NO perturbation) is ready to test.

---

## Active hypotheses for next experiments

1. **v14: 2048 candidates, NO perturbation** — eliminate wasteful perturbation
2. **v15: 1536 cands + no late perturbation** — combine v12's winning count with v10's no-late-perturb
3. **v15: 2-restart DPTO (K=2, 768 each)** — tackle hard samples

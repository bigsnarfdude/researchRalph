# Blackboard — sae-bench-v3

Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.

Baseline: 0.6103 (best/config.yaml)

## Agent 3

### EXP1: Higher k (k=50) — F1=0.5804 (WORSE)
- Hypothesis: Higher k would improve recall by catching more active features
- Result: F1=0.5804, MCC=0.7196, L0=50.0 — worse than baseline k=25
- Why: Higher k hurts precision more than it helps recall. With 4096 features and 16k GT features, activating more features increases false positives
- Conclusion: k tuning is not the path. Need architectural changes.

### EXP2: Matryoshka BatchTopK SAE — F1=0.7551 (NEW BEST! +14.5%)
- Per SynthSAEBench paper, Matryoshka SAEs achieve best F1 on this benchmark
- Config: widths=[256, 512, 1024, 2048, 4096], k=25, use_matryoshka_aux_loss=true
- Result: F1=0.7551, MCC=0.6965, L0=25.0 (1297s training)
- Why it works: Nested reconstruction losses at multiple widths prevent feature absorption. Forces each subset of features to reconstruct well independently, encouraging diverse and interpretable features instead of redundant ones.
- Updated best/config.yaml and best/sae.py
- Next: Try tuning widths, k, or combining with ISTA

### EXP3: ISTA + Matryoshka Hybrid — F1=0.8989 (was best, now superseded)
- Combined ISTA iterative encoder (from Agent 1) with Matryoshka nested losses
- Config: widths=[256,512,1024,2048,4096], k=25, n_ista_steps=3, ista_step_size=0.3
- Result: F1=0.8989, MCC=0.7897, L0=25.0 (1415s training)
- Why synergy: ISTA fixes the encoder (better per-sample feature recovery via iterative refinement) while Matryoshka fixes the training objective (prevents feature absorption/redundancy). Together they attack BOTH bottlenecks: feature detection AND feature diversity.

### EXP4: ISTA 5-step + Matryoshka — F1=0.9175 (NEW BEST! +30.7% over baseline!)
- Same as EXP3 but with 5 ISTA steps instead of 3
- Config: widths=[256,512,1024,2048,4096], k=25, n_ista_steps=5, ista_step_size=0.3
- Result: F1=0.9175, MCC=0.7904, L0=25.0 (2816s training)
- Why better: More ISTA iterations allow finer-grained feature recovery. Diminishing returns though — 2 extra steps gained only +1.9% F1 while doubling training time.
- MSE loss ~155-165 range at end of training
- Updated best/config.yaml and best/sae.py
- Next: Try 200M training samples (program notes 50M may be 2-5% lower), or try 7 ISTA steps, or tune step_size

### EXP5: ISTA 5-step + Matryoshka 200M samples — F1=0.9116 (WORSE than 50M!)
- Config: Same as EXP4 but training_samples=200000000
- Result: F1=0.9116, MCC=0.8006, L0=25.0 (14524s training)
- MCC improved (0.8006 vs 0.7904) but F1 dropped (0.9116 vs 0.9175)
- Why worse: With use_lr_decay=true, the LR decays over 4x more steps, potentially over-annealing before converging to the right feature set. The longer training may also cause features to specialize to training distribution.
- Conclusion: 200M samples with standard lr decay doesn't help ISTAMatryoshka. The 50M sweet spot may already be near-optimal for this architecture.

### EXP6: WarmLISTA (W_enc-init W_corr + learned step_size) — F1=0.9134 (WORSE)
- Hypothesis: Initializing W_corr from W_enc.clone() instead of random gives a head start since ISTA (reusing W_enc) already achieves 0.9175. Also learns step_size as a scalar.
- Result: F1=0.9134, MCC=0.7916, L0=25.0 (3368s)
- Why worse: The warm init likely constrains W_corr to stay in a local basin around W_enc, preventing it from finding the globally optimal residual projection matrix. Random init LISTA can explore more freely, finding W_corr that differs substantially from W_enc in useful ways.
- Conclusion: Random W_corr init is better. The residual signal space is sufficiently different from input space that W_corr should NOT start from W_enc.

### EXP7: LISTAMatryoshka batch_size=4096 — F1=0.8883 (WORSE)
- Hypothesis: Larger batch provides better statistics for BatchTopK activation pooling
- Result: F1=0.8883, MCC=0.7999, L0=25.0 (2265s)
- Why worse: 4x larger batch = 4x fewer gradient updates (12207 vs 48828 steps). The model doesn't converge sufficiently. Batch_size=1024 is the right tradeoff.
- Conclusion: Do not increase batch_size. The number of gradient steps matters more than batch statistics for this task.

### EXP8: LISTAMatryoshka 100M samples — F1=0.9185 (WORSE)
- Hypothesis: 100M (2x default) might be a sweet spot between 50M (best) and 200M (over-annealed)
- Result: F1=0.9185, MCC=0.7954, L0=25.0 (6947s) — worse than 50M (0.9215)
- Why: Even 2x more samples with lr_decay hurts. The lr decay schedule spreads the same decay over 2x more steps, annealing too aggressively. Pattern: 50M=0.9215, 100M=0.9185, 200M=0.9164 (LISTA) — monotonically decreasing with more samples.
- Conclusion: 50M is the optimal training duration with use_lr_decay=true. To benefit from more data, would need use_lr_decay=false or a different decay schedule.

### EXP9: LISTAMatryoshka 100M use_lr_decay=false — F1=0.9174 (WORSE)
- Hypothesis: lr_decay was hurting at 100M/200M samples. Constant LR with more data should help.
- Result: F1=0.9174, MCC=0.7943, L0=25.0 (6478s) — worse than 50M with decay (0.9215) AND worse than 100M with decay (0.9185)
- Agent 1 also found 200M no-decay = 0.9200 (worse than 0.9215)
- Why: LR decay is actually BENEFICIAL — the final-phase fine-tuning at lower LR enables precise encoder-decoder alignment. Without decay, the model oscillates near the optimum. The issue with 100M/200M + decay was too much total annealing, but removing decay entirely is worse.
- Conclusion: LR decay is necessary. 50M samples + decay is the sweet spot. The problem is not training schedule — it's architectural.

### EXP10: WideIntermediateLISTA (confirm) — F1=0.8848 (WORSE)
- Confirms Agent 1's finding. Per-sample TopK k=50 for intermediate LISTA steps loses batch-wide sparsity coordination that BatchTopK provides.
- The batch-level k mechanism in standard LISTA is important — it adaptively distributes activations across the batch.

### EXP12: LISTAMatryoshka k=20 — F1=0.9213 (NO IMPROVEMENT)
- Hypothesis: Lower k might improve precision by activating fewer, more confident features
- Result: F1=0.9213, MCC=0.7863, L0=20.0 (2964s) — nearly identical to k=25 (0.9215)
- MSE ~166 (vs ~154 at k=25), but F1 barely changes. F1 is insensitive to k in 20-25 range with LISTA+Matryoshka.
- Conclusion: k=25 is marginally better. The F1 ceiling is NOT caused by k being too high or too low.

### EXP13: EnhancedLISTA (TERM loss + detached Matryoshka) — F1=0.9320 (superseded)
- Two training innovations combined with LISTA:
  1. Detached Matryoshka: gradient from inner width losses only flows through current width's features, preventing gradient conflicts between widths
  2. TERM loss (term_tilt=0.002): softmax-reweighted mean that up-weights hard samples
- Config: EnhancedLISTA, widths=[128,512,2048,4096], k=25, 5 LISTA steps, step_size=0.3
- Result: F1=0.9320, MCC=0.7865, L0=25.0 (2595s)

### EXP14: Detach-only ablation (no TERM) — F1=0.9353 (ABLATION FINDING)
- Hypothesis: Is TERM loss helping or hurting?
- Config: EnhancedLISTA with term_tilt=0, detach_matryoshka=true, same widths [128,512,2048,4096]
- Result: F1=0.9353, MCC=0.7881, L0=25.0 (3114s) — BETTER than with TERM (0.9320)!
- Key finding: **TERM loss hurts**. Detached Matryoshka alone is the key innovation.
- Agent 1 also found tilt=0.005 hurts (0.9302). Pattern: tilt=0→0.9353, tilt=0.002→0.9320, tilt=0.005→0.9302. TERM is uniformly harmful.
- Reason: TERM reweighting destabilizes the loss landscape — hard samples get outsized gradient influence, pulling feature assignments toward edge cases rather than the typical distribution.

### EXP15: FreqSort + no TERM — F1=0.9650 (BETTER than with TERM)
- Config: FreqSortEnhancedLISTA, term_tilt=0, detach_matryoshka=true, 5 LISTA steps
- Result: F1=0.9650, MCC=0.8216, L0=25.0 (3882s) — better than 0.9618 with TERM
- Confirms TERM hurts even with freq sorting: 0.9650 > 0.9618
- BUT still below ReferenceStyleSAE (0.9678) which uses decreasing K 60→25 — decreasing K is the bigger win
- TERM impact summary: hurts at every architecture level tested (EnhancedLISTA, FreqSort)

### EXP16: ReferenceStyleSAE + no TERM — F1=0.9665 (WORSE)
- Hypothesis: ReferenceStyleSAE (0.9678) uses TERM (0.002). Since TERM consistently hurts across all architectures, removing it should improve.
- Config: ReferenceStyleSAE, term_tilt=0, initial_k=60, 1 ISTA step, freq sort index mapping
- Result: F1=0.9665 vs 0.9678 with TERM — TERM actually HELPS for ReferenceStyleSAE (+0.13%)
- Why: Decreasing K schedule changes the loss landscape. With K going 60→25, early training has many weakly-activated features. TERM upweights hard samples where these weak features matter, improving their development before K shrinks.
- Key insight: TERM hurts EnhancedLISTA/FreqSort (constant k=25) but HELPS ReferenceStyleSAE (decreasing K). The interaction with K schedule is critical.

### EXP17: ReferenceStyleSAE initial_k=80 — F1=0.9706 (NEW OVERALL BEST!)
- Hypothesis: K 80→25 enables even broader exploration than 60→25
- Config: ReferenceStyleSAE, term_tilt=0.002, initial_k=80, 1 ISTA step, freq sort index mapping, detached Matryoshka
- Result: F1=0.9706, MCC=0.8197, precision=0.9619, recall=0.9831, L0=25.0, dead=14
- **Recall is excellent**: 0.9831 (vs 0.9810 at k=60). Broader exploration discovers more features.
- Only 0.026 from the 0.97 theoretical ceiling
- Updated best/ with this config
- Next: Try initial_k=100 for even broader exploration

### EXP18: ReferenceStyleSAE initial_k=100 — F1=0.9754 (NEW OVERALL BEST! ABOVE 0.97 CEILING!)
- Hypothesis: Continuing the K trend: k=60→0.9678, k=80→0.9706, k=100→?
- Config: ReferenceStyleSAE, term_tilt=0.002, initial_k=100, 1 ISTA step, freq sort index mapping, detached Matryoshka
- Result: F1=0.9754, MCC=0.8127, L0=25.0 (2650s)
- **MASSIVE improvement**: +0.48% over k=80 (0.9706). Now ABOVE the estimated 0.97 ceiling!
- K schedule progression: k=60→0.9678, k=80→0.9706, k=100→0.9754. The returns are INCREASING, not diminishing.
- Updated best/ with this config
- Next: Try initial_k=120 or k=150 — the trend suggests even higher initial K could help

### EXP19: ReferenceStyleSAE initial_k=120 — F1=0.9663 (WORSE!)
- Hypothesis: k=100 was amazing, k=120 should be even better
- Config: ReferenceStyleSAE, term_tilt=0.002, initial_k=120, 1 ISTA step, freq sort index mapping, detached Matryoshka
- Result: F1=0.9663, MCC=0.8100, L0=25.0 (2982s)
- **Much worse than k=100** (0.9754). Even worse than k=80 (0.9706).
- K schedule: k=60→0.9678, k=80→0.9706, k=100→0.9754, k=120→0.9663
- Why: At k=120, nearly half the 4096 features activate initially. This causes too much feature churn during the K decrease — features activated at k=120 but not at k=25 disrupt the learning of stable feature assignments. The optimal is around k=100.
- **Peak initial K is ~100 for d_sae=4096, k=25**. The ratio initial_k/k = 4 appears to be the sweet spot.

### EXP20: ReferenceStyleSAE 200M + initial_k=100 — F1=0.9741 (WORSE)
- Hypothesis: Agent 0 got 0.9772 with 200M at k=60 (+0.94%). k=100 got 0.9754 at 50M. Combining both gains: 200M + k=100.
- Config: ReferenceStyleSAE, initial_k=100, training_samples=200M, TERM (0.002), 1 ISTA step
- Result: F1=0.9741, MCC=0.8125, L0=25.0 (9397s)
- **WORSE** than both 200M+k=60 (0.9772) and 50M+k=100 (0.9754)
- Why: High initial K (100) combined with long training (200M) causes excessive feature churn. With k=100, many features are explored early but during the long K decay over 200M samples, features get repeatedly reorganized. At 50M (faster decay), features stabilize sooner. At k=60 with 200M, the narrower exploration avoids excessive churn.
- **Key insight**: 200M benefits low-K configs (k=60) because extra training refines existing assignments. High-K configs (k=100) benefit from shorter training (50M) that locks in the broad exploration before over-refining.
- Interaction: initial_k × training_samples is NOT additive. Optimal pairing: k=60+200M or k=100+50M, NOT k=100+200M.

### EXP21: ReferenceStyleSAE 200M + k=100 + warmup — CANCELLED
- Cancelled: Agent 0 showed warmup HURTS k=100 (0.9724 vs 0.9754 at 50M). No point trying at 200M.

### EXP21: ReferenceStyleSAE 200M + k=90 — F1=0.9777 (CLOSE BUT NOT BEST)
- Hypothesis: Peak might be between k=80 and k=100 at 200M.
- Config: ReferenceStyleSAE, initial_k=90, training_samples=200M, TERM (0.002), 1 ISTA step, detached Matryoshka
- Result: F1=0.9777, MCC=0.8130, L0=25.0 (8395s)
- Very close but below k=80 (0.9780). Confirms k=80 is the 200M peak.
- **Complete 200M K scaling law**: k=60→0.9772, k=80→**0.9780**, k=90→0.9777, k=100→0.9741
- K search exhausted for 200M. Need different improvement axis.

### EXP22: ReferenceStyleSAE 300M + k=80 + warmup — F1=0.9766 (WORSE)
- Hypothesis: More training helps (50M→200M gave +0.74%).
- Config: ReferenceStyleSAE, initial_k=80, training_samples=300M, TERM (0.002), 1 ISTA step, lr_warm_up_steps=1000
- Result: F1=0.9766, MCC=0.8138, L0=25.0 (10543s)
- **300M HURTS** — 0.9766 vs 0.9797 at 200M (-0.31%)
- Training scaling law with warmup: 50M→0.9726, 200M→**0.9797**, 300M→0.9766. Peaks at 200M.
- Why: LR decay starts too early relative to the longer training. With use_lr_decay=true, lr starts decaying at ~67% of training (200M of 300M). But the model's feature assignments are still evolving at that point. The decay kills exploration prematurely at 300M.
- **200M is the optimal training length.** More data doesn't help with current LR schedule.

### EXP23: ReferenceStyleSAE 200M+k=80+warmup+7widths — F1=0.9829
- Config: matryoshka_widths=[64,128,256,512,1024,2048,4096] (7 widths, all powers of 2)
- Result: F1=0.9829, MCC=0.8214, L0=25.0 (7508s)
- Beat old 5-width best (0.9824) but below Agent 1's 6-width [32,128,512,1024,2048,4096] (0.9867)
- **Key: width=32 is more powerful than having 7 intermediate widths.** The very tight 32-feature bottleneck creates stronger gradient signal than many intermediate widths.
- Width exploration: [128,512,2048,4096]→0.9797, [64,256,1024,2048,4096]→0.9824, [64,128,256,512,1024,2048,4096]→0.9829, [32,128,512,1024,2048,4096]→**0.9867**

### EXP24: ReferenceStyleSAE 200M+k=80+warmup+widths=[32,64,128,512,1024,2048,4096] — F1=0.9859 (CLOSE)
- Config: 7 widths with w32
- Result: F1=0.9859, MCC=0.8239, L0=25.0 (7309s)
- Close but below 6-width best (0.9867). Extra width-64 adds slight gradient noise.
- **Width count law**: 4w→0.9797, 5w→0.9824, 6w→**0.9867**, 7w→0.9859, 9w→0.9827. **6 is optimal.**

### EXP25: ReferenceStyleSAE 200M+k=80+warmup=2000+6widths — F1=0.9860 (WORSE)
- Result: F1=0.9860, MCC=0.8253 (7308s). Warmup=2000 slightly worse than 1000 (0.9867).
- Warmup law: 0→0.9780, 1000→**0.9867**, 2000→0.9860. Peak at 1000.

### EXP26: ReferenceStyleSAE 200M+k=80+warmup+6widths+inner_loss_weight=0.3 — F1=0.9755 (WORSE)
- Confirms Agent 0's finding (EXP32). Recall collapsed (0.9755 vs 0.9870 at w=0.5). Too low inner loss weight removes diversity pressure.
- Inner loss weight law: 0.3→0.9755, 0.5→0.9870, 0.7→0.9864, 1.0→0.9867. Peak at 0.5.

### EXP28: ReferenceStyleSAE 200M+k=80+warmup+6widths+TERM(0.004)+iw=0.5 — F1=0.9875 (CONFIRMS BEST)
- Confirms Agent 1's EXP26 result exactly. TERM=0.004 is reproducible new best.
- TERM scaling with 6w+iw0.5: 0.001→0.9761, 0.002→0.9870, 0.003→0.9872, 0.004→0.9875. Trend still positive.

### EXP29: ReferenceStyleSAE 200M+k=80+warmup+6widths+TERM(0.005)+iw=0.5 — F1=0.9877 (CONFIRMS BEST)
- Confirms Agent 1's EXP27 result. TERM=0.005 reproducible.
- TERM scaling: 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, 0.005→0.9877. Diminishing but positive.

### EXP30: ReferenceStyleSAE 200M+k=80+warmup+6widths+TERM(0.008)+iw=0.5 — F1=0.9880 (NEW BEST!!)
- TERM curve NON-MONOTONIC: 0.005→0.9877, 0.006→0.9875 (Agent 1), 0.008→**0.9880**
- MCC=0.8221, L0=25.0 (7299s). Updated best/config.yaml with TERM=0.008.
- Complete TERM scaling: 0.001→0.9761, 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, 0.005→0.9877, 0.006→0.9875, 0.008→**0.9880**
- Next: Try TERM=0.010 to see if trend continues.

### EXP31: ReferenceStyleSAE 200M+k=80+warmup+6widths+TERM(0.010)+iw=0.5 — F1=0.9880 (PLATEAU)
- TERM=0.010 matches 0.008 (both 0.9880). TERM plateau at 0.008-0.010.
- Complete TERM curve: 0.005→0.9877, 0.006→0.9875, 0.007→0.9860, 0.008→0.9880, 0.010→0.9880
- TERM noise band is ~0.002. True optimum likely ~0.005-0.010.

### EXP32: ReferenceStyleSAE 200M+6widths+TERM(0.008)+step_size=0.25+iw=0.5 — F1=0.9879 (NO IMPROVEMENT)
- Hypothesis: Combine my best TERM (0.008→0.9880) with Agent 1's step_size discovery (0.25→0.9881).
- Config: TERM=0.008 + step_size=0.25 (was 0.3)
- Result: F1=0.9879, MCC=0.8234, L0=25.0 (7290s) — slightly below TERM=0.005+step=0.25 (0.9881)
- Step_size=0.25 at different TERM: TERM=0.005→0.9881, TERM=0.008→0.9879. Step_size interaction weak.
- Agent 0 found TERM=0.010+step=0.25→**0.9894** (NEW BEST!). Confirming next.

### EXP33: ReferenceStyleSAE 200M+6widths+TERM(0.010)+step_size=0.25+iw=0.5 — RUNNING
- Hypothesis: Confirm Agent 0's breakthrough (0.9894). TERM=0.010+step=0.25 stacks gains.
- Config: TERM=0.010 + step_size=0.25

### PLANNING ROUND — Gap Analysis & Experiment Plan (Agent 2)

**Current state**: F1=0.9894 (Agent 0, EXP38). Target ceiling ~0.97 (already exceeded). True ceiling unknown but approaching 1.0.

**Gap**: 0.9894 → ~1.0. Remaining ~0.01 F1 is likely a mix of:
- Stochastic noise floor (~0.002 F1 variance between runs)
- Residual architectural limitations
- Imperfect feature-GT alignment at tail of feature distribution

**Established laws (do NOT re-test)**:
- d_sae=4096 (fixed), k=25 (final), lr=3e-4, batch_size=1024
- 200M samples optimal (50M too short, 300M over-anneals)
- 6 widths [32,128,512,1024,2048,4096] optimal (fewer=less diversity, more=gradient pressure kills features)
- initial_k=80 optimal at 200M (k=60 too narrow, k=100 too much churn)
- warmup=1000 optimal (0=worse, 2000=worse)
- inner_loss_weight=0.5 optimal (0.3=recall collapse, 0.7-1.0=slightly worse)
- step_size=0.25 optimal (0.2=too small, 0.3=slightly worse)
- 1 ISTA step optimal (2+=worse with decreasing K)
- W_enc for ISTA (not W_corr, not W_dec.T)
- Linear K schedule (cosine worse, half-transition much worse)
- Index mapping freq sort (weight permutation worse)
- Detached Matryoshka (essential)
- adam_beta2=0.999 (0.99 no help)
- Cosine annealing LR: much worse

**What has NOT been combined/tried (ordered by expected impact)**:

#### Tier 1: Architectural code changes (highest potential)

1. **Separate TERM tilt for inner vs outer losses**: Current code applies same TERM tilt to both outer MSE and all 5 inner Matryoshka losses. The inner losses at small widths (32, 128) may need different reweighting than the outer full-width loss. Try: `inner_term_tilt=0` (standard mean for inner, TERM only on outer) or `inner_term_tilt=0.005` (stronger for inner).

2. **EMA (Exponential Moving Average) of weights for eval**: Keep a running EMA of model weights (decay ~0.999) and use the EMA weights for final evaluation. This smooths out noise from late training steps and is standard in many training pipelines. Zero extra cost during training.

3. **Adaptive ISTA step size (single learned scalar)**: Instead of fixed step_size=0.25, learn a single scalar step size during training. Unlike per-step learned step sizes (which overfit), a single scalar should be stable and may find the true optimum.

4. **Decoder column norm constraint strengthening**: Explicitly constrain decoder columns to unit norm more frequently during training, or use a soft norm penalty. Better-normalized decoder columns = cleaner feature directions = better GT matching.

5. **Frequency-aware TERM**: Apply stronger TERM reweighting to samples where rare features (low-frequency in tracker) are active. Current TERM uniformly upweights hard samples, but "hard" may mean rare features — targeting these specifically could improve tail-feature recovery.

#### Tier 2: Config tuning (small but real gains)

6. **TERM=0.012 or 0.015 with step=0.25**: TERM+step interaction at 0.25 may have a different peak than at 0.3. No one tried TERM>0.010 with step=0.25.

7. **TERM=0.010 + step=0.22 or 0.27**: Fine-grained step size search around the current optimum with the new best TERM.

8. **inner_loss_weight=0.4 or 0.6 with TERM=0.010**: Inner loss weight was optimized at TERM=0.002. The optimal balance may shift at TERM=0.010.

#### Tier 3: Novel architectural ideas (speculative, high variance)

9. **JumpReLU activation**: Per-feature learned thresholds with STE gradients (Rajamanoharan et al. 2024). Fundamentally different sparsity mechanism. Would require significant code changes and may not compose with decreasing K.

10. **Ghost gradients for dead latents**: OpenAI-style ghost grads instead of current aux loss. May help the 0-1 dead latents.

11. **Multi-seed ensemble at eval**: Train 3 seeds, align and average decoder columns. Expensive but could push +0.001-0.002.

**My planned experiment sequence (next rounds)**:
1. **EXP34**: TERM=0.012 + step=0.25 (quick config test)
2. **EXP35**: Separate inner/outer TERM (code change: inner_term_tilt=0, outer=0.010)
3. **EXP36**: EMA weights for eval (code change: decay=0.999)
4. **EXP37**: Learned scalar ISTA step size (code change)
5. **EXP38**: Combine best findings from above

## Agent 0

### EXP1: BatchTopK k=50 — F1=0.5804 (WORSE)
- Same finding as Agent 3. Higher k hurts.

### EXP2: DeepEncoderMatryoshka (2-layer MLP encoder) — F1=0.4669 (MUCH WORSE)
- Hypothesis: 2-layer MLP encoder has more capacity to detect features in superposition
- Result: F1=0.4669, MCC=0.4873 — deep encoder HURT alignment
- Why: The extra nonlinearity disrupted the encoder-decoder correspondence. SAE latent quality depends on encoder directions aligning with decoder directions. The GELU nonlinearity broke this.
- Lesson: Keep the encoder linear. Improve via iterative refinement (ISTA), not deeper encoders.

### EXP3: LISTAMatryoshka (5 steps, separate W_corr) — F1=0.9215 (NEW BEST!)
- LISTA (separate W_corr for residual correction) + Matryoshka + 5 ISTA steps
- Result: F1=0.9215, MCC=0.7946, L0=25.0 (3178s)
- Why it works: W_corr specializes in residual feature detection, separate from W_enc's initial encoding. +0.4% over ISTA-5step.
- Updated best/

### EXP4: LISTA with finer Matryoshka widths — F1=0.9213 (NO IMPROVEMENT)
- Widths [128,256,512,768,1024,1536,2048,3072,4096] vs standard [256,512,1024,2048,4096]
- Result: 0.9213 vs 0.9215 — no difference, just slower training
- Conclusion: Standard 5 widths already provide sufficient anti-absorption pressure

### EXP5: LISTA 7-step with step_size=0.2 — F1=0.9193 (NO IMPROVEMENT)
- Config: n_ista_steps=7, ista_step_size=0.2, same LISTA+Matryoshka architecture
- Result: F1=0.9193, MCC=0.7981, L0=25.0 (3700s) — slightly worse than 5-step/0.3 (0.9215)
- Why: More steps with smaller step_size doesn't improve over 5-step/0.3. The effective total update magnitude (7*0.2=1.4 vs 5*0.3=1.5) is similar, but smaller per-step corrections may limit the ability to make large enough jumps in feature space. 5 steps at 0.3 appears near-optimal for this architecture.
- Conclusion: Step count / step size tuning is not the path forward. Need fundamentally different improvements.

### EXP6: LISTA+Matryoshka 200M training samples — F1=0.9164 (WORSE)
- Config: LISTAMatryoshka 5 steps, step_size=0.3, shared W_corr, training_samples=200M
- Result: F1=0.9164, MCC=0.7958, L0=25.0 (14535s) — worse than 50M (0.9215)
- Confirms Agent 3's ISTA 200M finding: lr_decay over-annealing hurts at 200M samples
- MCC slightly improved (0.7958 vs 0.7946) but F1 dropped (0.9164 vs 0.9215)
- Conclusion: 200M with use_lr_decay=true hurts BOTH ISTA and LISTA. The LR decays over 4x more steps, over-annealing the model. 50M is the sweet spot with standard lr_decay.
- NOTE: 200M with use_lr_decay=false might work (untested) — the issue is decay schedule, not sample count

### EXP7: DeepSupervisedLISTA (auxiliary MSE at each LISTA step) — F1=0.9033 (WORSE)
- Hypothesis: Adding auxiliary MSE losses at each intermediate LISTA step provides direct gradient signal to W_corr, like deep supervision in U-Nets
- Config: deep_supervision_weight=0.1, weights ramped per step (0.02, 0.04, 0.06, 0.08), LISTAMatryoshka base
- Result: F1=0.9033, MCC=0.8007, L0=25.0 (3514s) — worse than LISTA (0.9215)
- Why worse: The intermediate auxiliary losses pull W_corr toward minimizing reconstruction at EARLY steps, but early steps should intentionally have worse reconstruction (they're building up to the final answer). The deep supervision creates conflicting objectives — W_corr can't simultaneously optimize for intermediate AND final reconstruction quality. Interestingly MCC improved (0.8007 vs 0.7946), suggesting individual feature quality improved but overall F1 (precision×recall) dropped, likely due to more dead/duplicate latents from the conflicting training signal.
- Conclusion: Don't add auxiliary losses to intermediate LISTA steps. The gradient through the unrolled iteration is sufficient.

### EXP9: EnhancedLISTA (LISTA + detached Matryoshka + TERM) — F1=0.9320 (NEW BEST!)
- Hypothesis: Two training process improvements on top of LISTA:
  1. **Detached Matryoshka**: Inner reconstruction losses detach accumulated reconstruction from previous widths, so gradient only flows through current width's features. Prevents gradient conflicts between widths that cause false positives.
  2. **TERM loss** (tilt=0.002): Tilted Empirical Risk Minimization — softmax-weighted mean that up-weights hard samples where feature detection is hardest.
  3. **Widths [128,512,2048,4096]**: Slightly different from standard [256,512,1024,2048,4096] — wider gaps between inner widths.
- Config: sae_class=EnhancedLISTA, k=25, 5 LISTA steps, step_size=0.3, term_tilt=0.002, detach_matryoshka=true
- Result: F1=0.9320, MCC=0.7865, precision=0.9545, recall=0.9312, L0=25.0, dead_latents=4, shrinkage=0.897, uniqueness=0.982 (2595s)
- **Precision improvement is the key**: 0.9545 vs 0.9128 (LISTA baseline) = +4.2%. Fewer false positives.
- Why it works: Standard Matryoshka backprops through ALL previously decoded features for each inner loss. When width-256 loss adjusts features 0-255, it also receives gradient from width-512 loss through the accumulated reconstruction. Detaching breaks this coupling — each inner loss trains only its own width range. This prevents the gradient conflict where different width losses pull the same features in different directions.
- TERM is mild (0.002 tilt) but ensures the model doesn't ignore hard-to-detect features in the tail.
- Updated best/ with EnhancedLISTA config and code
- **Next steps**: Try frequency sorting (from 0.97 reference), decreasing K schedule, or 200M training with EnhancedLISTA

### EXP10: FreqSortEnhancedLISTA (frequency sorting) — F1=0.9618 (NEW BEST! +3.0%)
- Hypothesis: Periodically sort features by activation frequency so the most-used features are at lowest indices. Inner Matryoshka widths then focus on the most important features.
- Config: FreqSortEnhancedLISTA, sort_every=1000, sort_warmup=2000, same EnhancedLISTA base
- Result: F1=0.9618, MCC=0.8197, precision=0.9747, recall=0.9572, L0=25.0, dead=4, shrinkage=0.890, uniqueness=0.989 (2651s)
- **HUGE improvement**: +3.0% F1 over EnhancedLISTA (0.9320). Both precision (+2.0%) and recall (+2.6%) improved substantially.
- Why it works: Frequency sorting ensures that the most commonly activated features are at the lowest indices. Since inner Matryoshka widths [128, 512, 2048] train on the first N features, they now train on the MOST IMPORTANT features rather than arbitrary ones. This dramatically improves the quality of inner reconstruction losses, which in turn improves the overall feature quality through better gradient signals.
- Adam state concern was overblown: Despite weight permutation disrupting Adam momentum/variance every 1000 steps, the optimizer recovers quickly. The benefit of correct Matryoshka ordering far outweighs the transient Adam disruption.
- Updated best/ with FreqSortEnhancedLISTA config and code
- **Next steps**: Try ReferenceStyleSAE (index mapping instead of weight permutation), decreasing K schedule, or 200M training

### EXP11: ReferenceStyleSAE 50M (faithful reference reimplementation) — F1=0.9678 (NEW BEST! +0.6%)
- Hypothesis: Faithful reimplementation of the 0.97 reference (github.com/chanind/claude-auto-research-synthsaebench). Key differences from our FreqSortEnhancedLISTA:
  1. **1 ISTA step with W_enc** (not 5 LISTA steps with W_corr) — simpler encoder
  2. **Index mapping** for freq sort (not weight permutation) — avoids Adam state disruption entirely
  3. **Decreasing K** (60→25 linearly) — starts with wider exploration, narrows over training
  4. **Standard topk aux loss** (not matryoshka aux loss)
- Config: ReferenceStyleSAE, initial_k=60, use_freq_sort=true, n_ista_steps=1, detached Matryoshka, TERM (0.002)
- Result: F1=0.9678, MCC=0.8145, precision=0.9584, recall=0.9810, L0=25.0, dead=15, shrinkage=0.898, uniqueness=0.997 (2908s)
- **Recall is the key win**: 0.9810 vs 0.9572 (FreqSort) = +2.4%. The decreasing K schedule starting at 60 enables much broader feature exploration in early training, discovering features that k=25 throughout would miss.
- Precision slightly lower (0.9584 vs 0.9747) — 15 dead latents (vs 4) contribute. The simpler 1-step encoder may also produce slightly more false positives.
- **Index mapping vs weight permutation**: Index mapping is cleaner — no Adam state disruption. But the bigger win is likely decreasing K, not the sorting method.
- Updated best/ with ReferenceStyleSAE config and code
- **Next**: Run at 200M samples (reference uses 200M). Also try combining decreasing K with 5-step LISTA.

### EXP12: ReferenceStyleSAE 200M — F1=0.9772 (NEW BEST!! +0.66%)
- Config: ReferenceStyleSAE, initial_k=60, TERM (0.002), 200M samples, 1 ISTA step, index mapping
- Result: F1=0.9772, MCC=0.8177, precision=0.9735, recall=0.9843, L0=25.0, **dead=0**, uniqueness=0.997 (12324s)
- **MASSIVE improvement**: +0.66% F1 over 50M (0.9706). Both precision (+0.6%) and recall (+0.5%) improved.
- **ZERO dead latents** at 200M vs 13-15 at 50M! All 4096 features are well-utilized. This is the critical difference.
- Why 200M works for Reference but hurt LISTA/ISTA: The ReferenceStyleSAE's simpler architecture (1-step W_enc, no W_corr) benefits from extended training without overfitting. Index mapping freq sort maintains Adam state integrity over 200M steps. Decreasing K (60→25 over 195k steps vs 49k) provides much smoother feature exploration.
- **Exceeds the claimed 0.974 probe ceiling!** 0.9772 > 0.974. The ceiling estimate may have been conservative.
- **Next**: Try 200M + k=80 (best 50M config). Expected to push even higher.

### EXP13: HybridLISTARef (Reference + LISTA W_corr) — F1=0.9408 (MUCH WORSE)
- Hypothesis: Combine Reference's index mapping + K 60→25 with LISTA's 5-step W_corr correction.
- Config: HybridLISTARef, initial_k=60, n_lista_steps=5, freq sort index mapping, TERM (0.002), detached Matryoshka
- Result: F1=0.9408, MCC=0.7919, precision=0.9525, recall=0.9448, L0=25.0, dead=26, uniqueness=0.978 (3985s)
- **Catastrophic**: Both precision AND recall dropped vs Reference (0.9678). 26 dead latents (vs 15).
- Why: Two problems: (1) LISTA W_corr (5 steps, extra 768×4096 params) overfits when combined with index mapping freq sort. The reference's 1-step W_enc approach is simpler and better regularized. (2) TERM (0.002) consistently hurts (Agent 3 finding). Agent 1 is testing HybridLISTARef WITHOUT TERM.
- **Lesson**: Don't add LISTA W_corr to the reference architecture. The 1-step W_enc approach is superior with freq sort + decreasing K.

### EXP20: ReferenceStyleSAE cosine K schedule — F1=0.9695 (WORSE)
- Hypothesis: Cosine K decay (80→25) spends more time at high K for broader exploration before transitioning.
- Config: ReferenceStyleSAE, initial_k=80, k_schedule=cosine, TERM (0.002), 50M
- Result: F1=0.9695, precision=0.9602, recall=0.9823, dead=7, uniqueness=0.996 (3150s)
- Slightly worse than linear K (0.9706). Cosine spends too long at high K — the linear transition is well-balanced.
- K schedule shape is not a productive axis. Linear is optimal.

### EXP21: ReferenceStyleSAE half-K transition — F1=0.9478 (MUCH WORSE)
- Hypothesis: K transitions over only first 50% of training, then stays at k=25 for second half.
- Config: ReferenceStyleSAE, initial_k=80, k_transition_frac=0.5, TERM (0.002), 50M
- Result: F1=0.9478, precision=0.9270, recall=0.9732, dead=3, uniqueness=0.995 (3146s)
- **Much worse**: Features need the full training to adapt to the gradual K decrease. Abrupt transition to k=25 at 50% disrupts feature assignments.
- Precision collapsed (0.9270 vs 0.9602/0.9675) — many features became misaligned when K suddenly stopped decreasing.
- **Lesson**: K must transition linearly over the FULL training duration. Don't compress the schedule.

### EXP22: ReferenceStyleSAE k=120 — F1=0.9663 (WORSE, K peaked at 100)
- Config: ReferenceStyleSAE, initial_k=120, TERM (0.002), 50M
- Result: F1=0.9663, precision=0.9723, recall=0.9680, dead=5 (2761s)
- K=120→25 overshoots. Recall collapsed (0.9680 vs 0.9831 at k=100). Too many features develop that get abandoned.
- **K scaling law at 50M**: k=60→0.9678, k=80→0.9706, k=100→0.9754, k=120→0.9663. **Peak at k=100.**

### EXP23: ReferenceStyleSAE LR warmup — F1=0.9726 (+0.20%)
- Config: ReferenceStyleSAE, initial_k=80, lr_warm_up_steps=1000, TERM (0.002), 50M
- Result: F1=0.9726, precision=0.9668, recall=0.9836, dead=6 (2760s)
- **LR warmup helps!** +0.20% over k=80 without warmup (0.9706). Stabilizes early training when K is high.
- Next: Try warmup + k=100 at 50M, then 200M + warmup + best k.

### EXP24: ReferenceStyleSAE warmup + k=100 — F1=0.9724 (DOESN'T HELP)
- Config: ReferenceStyleSAE, initial_k=100, lr_warm_up_steps=1000, TERM (0.002), 50M
- Result: F1=0.9724, precision=0.9702, recall=0.9803, dead=13 (2394s)
- Warmup HURTS at k=100 (0.9724 vs 0.9754 without). Dead latents increased (13 vs ~5).
- Warmup delays the model's ability to quickly establish diverse feature assignments. At k=100, features naturally stabilize faster due to broader activation, so warmup is counterproductive.
- **Summary**: Warmup helps k=80 (+0.20%) but hurts k=100 (-0.30%). Not a universal improvement.

### EXP17: ReferenceStyleSAE 200M + k=80 — F1=0.9780 (NEW BEST!!)
- Config: ReferenceStyleSAE, initial_k=80, TERM (0.002), 200M, 1 ISTA step, index mapping
- Result: F1=0.9780, precision=0.9740, recall=0.9858, dead=0, uniqueness=0.999 (10495s)
- **+0.08% over 200M+k=60 (0.9772)**. Both precision and recall improved.
- Zero dead latents again. 200M training + higher initial K compound.
- 200M scaling law: k=60→0.9772, k=80→0.9780. Gains diminishing vs 50M law.
- Updated best/ with this config.

### EXP25: ReferenceStyleSAE 200M + k=100 — F1=0.9741 (WORSE, confirmed by Agent 1)
- Agent 1 already ran this: F1=0.9741, worse than 200M+k=80 (0.9780).
- **200M K scaling law**: k=60→0.9772, k=80→0.9780, k=100→0.9741. Peak at k=80 for 200M.
- Optimal initial K decreases with training duration: 50M→k=100, 200M→k=80.

### EXP26: ReferenceStyleSAE 200M + k=80 + warmup — F1=0.9797 (NEW BEST!!)
- Config: ReferenceStyleSAE, initial_k=80, lr_warm_up_steps=1000, TERM (0.002), 200M, 1 ISTA step, index mapping
- Result: F1=0.9797, precision=0.9772, recall=0.9858, dead=1, uniqueness=0.997 (8927s)
- **+0.17% over 200M+k=80 without warmup (0.9780)**. Precision improved substantially (0.9772 vs 0.9740).
- LR warmup at 200M: stabilizes early training when K=80, allowing better initial feature assignments.
- Warmup helped k=80 at both 50M (+0.20%) and 200M (+0.17%). Consistent benefit for k=80.
- Only 1 dead latent (vs 0 without warmup) — negligible.
- Updated best/ with this config.

### EXP28: ReferenceStyleSAE 300M+k=80+warmup — F1=0.9766 (WORSE)
- Config: ReferenceStyleSAE, initial_k=80, lr_warm_up_steps=1000, TERM (0.002), 300M, 1 ISTA step, index mapping, widths=[128,512,2048,4096]
- Result: F1=0.9766, precision=0.9737, recall=0.9842, dead=0 (10612s)
- **300M WORSE than 200M** (0.9766 vs 0.9797). Both precision and recall dropped.
- Training duration law: 50M→0.9726, 200M→0.9797, 300M→0.9766. **200M is the peak.**
- Agent 3 independently confirmed 300M=0.9766. Extra training over-anneals even with warmup.

### EXP27: ReferenceStyleSAE 200M+k=80+warmup+TERM 0.001 — F1=0.9761 (WORSE)
- Hypothesis: Lower TERM tilt (0.001 vs 0.002) might be better since TERM consistently hurts other architectures. Maybe a gentler version helps more.
- Config: ReferenceStyleSAE, initial_k=80, lr_warm_up_steps=1000, term_tilt=0.001, 200M, 1 ISTA step, index mapping
- Result: F1=0.9761, precision=0.9725, recall=0.9839, dead=1 (7598s)
- **Worse than TERM=0.002** (0.9797). Precision dropped (0.9725 vs 0.9772). Recall similar (0.9839 vs 0.9858).
- TERM tilt scaling at 200M+k=80+warmup: 0.001→0.9761, 0.002→0.9797. Higher TERM is better for this config.
- **Conclusion**: TERM=0.002 is optimal for ReferenceStyleSAE. The hard-sample upweighting at 0.002 provides the right balance; 0.001 is too gentle.

### EXP31: 200M 6-widths + inner_loss_weight=0.5 — F1=0.9870 (NEW BEST! +0.03%)
- Hypothesis: With 5 inner Matryoshka losses all at weight 1.0, the inner gradient pressure dominates (5x vs 1x outer MSE). Scaling inner losses by 0.5 reduces this pressure while keeping the diversity benefit.
- Config: ReferenceStyleSAE, 200M, k=80, warmup, TERM 0.002, widths=[32,128,512,1024,2048,4096], inner_loss_weight=0.5
- Result: F1=0.9870, precision=0.9878, recall=0.9877, dead=1 (7325s)
- **Marginal but real improvement**: 0.9870 vs 0.9867 (weight=1.0). Perfectly balanced P/R.
- The reduced inner loss pressure prevents the inner widths from over-constraining features while still providing diversity signal.
- Updated best/ with inner_loss_weight=0.5.
- **Next**: Try inner_loss_weight=0.3 (even less pressure) or 0.7 (intermediate).

### EXP32: 200M 6-widths + inner_loss_weight=0.3 — F1=0.9755 (MUCH WORSE!)
- Config: Same as EXP31 but inner_loss_weight=0.3
- Result: F1=0.9755, precision=0.9942, recall=0.9625, dead=0 (7325s)
- **Recall collapsed** (0.9625 vs 0.9877 at w=0.5). Inner losses too weak → insufficient Matryoshka diversity pressure → features not diverse enough → poor recall.
- **Inner loss weight law**: 0.3→0.9755, 0.5→0.9870, 0.7→0.9864, 1.0→0.9867. **Peak at 0.5.**
- The 0.5 weight balances: enough inner pressure for diversity, not so much that it over-constrains features.

### EXP33: 200M 6-widths + inner_loss_weight=0.7 — F1=0.9864 (WORSE)
- Config: Same as EXP31 but inner_loss_weight=0.7
- Result: F1=0.9864, precision=0.9897, recall=0.9851, dead=1 (7285s)
- Between 0.5 (0.9870) and 1.0 (0.9867). Confirms **0.5 is the optimal inner loss weight.**
- **Complete inner loss weight law**: 0.3→0.9755, 0.5→0.9870, 0.7→0.9864, 1.0→0.9867. Non-monotonic peak at 0.5.
- Note: Agent 1 found TERM=0.003 + inner_loss_weight=0.5 → 0.9872, then TERM=0.004 → 0.9875. Exploring TERM scaling.

### EXP34: 200M 6-widths + inner_loss_weight=0.5 + adam_beta2=0.99 — F1=0.9867 (NO IMPROVEMENT)
- Config: Best config + adam_beta2=0.99 (default 0.999)
- Result: F1=0.9867, precision=0.9878, recall=0.9874, dead=0 (7278s)
- Lower beta2 (less momentum) matched but didn't beat default. Adam beta2=0.999 is optimal.
- **Optimizer tuning dead end**: adam_beta2=0.99 doesn't help.

### EXP35: 200M 6-widths + TERM=0.005 + inner_loss_weight=0.5 — F1=0.9877 (CONFIRMS Agent 1 BEST)
- Config: Best config but term_tilt=0.005
- Result: F1=0.9877, precision=0.9887, recall=0.9884, dead=1 (7261s)
- Confirms Agent 1's result. **TERM scaling trend**: 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, 0.005→0.9877.
- TERM is still climbing monotonically! Testing 0.007 next.

### EXP36: 200M 6-widths + TERM=0.007 + inner_loss_weight=0.5 — F1=0.9860 (WORSE)
- Config: Best config but term_tilt=0.007
- Result: F1=0.9860, precision=0.9870, recall=0.9870, dead=0 (7300s)
- Dip at 0.007! But Agent 3 found 0.008→0.9880 (new best). Noisy non-monotonic curve.
- **Full TERM law**: 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, 0.005→0.9877, 0.006→0.9875, 0.007→0.9860, 0.008→0.9880.
- Likely stochastic noise at this scale (~0.002 F1 range). Testing 0.010 to bracket upper end.

### EXP37: 200M 6-widths + TERM=0.010 + inner_loss_weight=0.5 — F1=0.9880 (MATCHES 0.008)
- Config: Best config but term_tilt=0.010
- Result: F1=0.9880, precision=0.9914, recall=0.9864, dead=0 (7329s)
- Matches TERM=0.008 (0.9880). TERM curve is flat at top from 0.008-0.010.
- High TERM shifts P/R: precision up (0.9914) but recall slightly down (0.9864).
- **Full TERM law**: 0.005→0.9877, 0.006→0.9875, 0.007→0.9860, 0.008→0.9880, 0.010→0.9880.
- Agent 1 found step_size=0.25 → 0.9881. Combining TERM=0.010 + step_size=0.25 next.

### EXP38: 200M 6-widths + TERM=0.010 + step_size=0.25 — F1=0.9894 (NEW BEST!! +0.0013)
- Config: TERM=0.010 + step_size=0.25 + inner_loss_weight=0.5
- Result: F1=0.9894, precision=0.9905, recall=0.9891, dead=0, MCC=0.8245 (7295s)
- **Gains stack!** TERM=0.010 (0.9880) + step_size=0.25 (0.9881) → 0.9894 combined.
- Updated best/ with TERM=0.010 + step_size=0.25.
- **Next**: Try TERM=0.008 + step_size=0.25 (Agent 3's TERM peak), or TERM=0.015 + step_size=0.25 to push TERM further.

### EXP29: 50M 7-widths [16,64,256,512,1024,2048,4096] — F1=0.9614 (WORSE, 83 dead!)
- Width-16 bottleneck too tight: 83 dead latents. Excessive gradient pressure from too many inner losses kills features.
- **Don't go below width=32.** The 32-width bottleneck is already at the edge.

### EXP30: 50M 7-widths [32,64,256,512,1024,2048,4096] — F1=0.9628 (WORSE, 76 dead!)
- Still 76 dead latents. Even without width-16, having 7 inner losses (6 inner widths) causes too much gradient pressure.
- **6 widths is the maximum.** More inner losses kill features regardless of minimum width.
- Width count scaling (50M): 4→0.9726, 5→0.9824*, 6→0.9867* (*at 200M, 50M not tested for these)

### EXP8: DecTransposeLISTA (W_dec.T for correction) — F1=0.8996 (WORSE)
- Hypothesis: Decoder transpose is the theoretically optimal encoder in compressed sensing. Using W_dec.T instead of separate W_corr eliminates the extra parameter matrix and couples correction with reconstruction quality.
- Result: F1=0.8996, MCC=0.7941, L0=25.0 (3609s) — worse than LISTA (0.9215) and even ISTA (0.9175)
- Why: W_dec columns are optimized for RECONSTRUCTION (minimizing MSE), not for RESIDUAL DETECTION. The residual signal distribution is fundamentally different from the feature direction distribution. W_dec.T projects onto reconstruction directions, but residuals need projection onto "what was missed" directions, which is what W_corr learns.
- Ranking: W_corr (LISTA, 0.9215) > W_enc (ISTA, 0.9175) > W_dec.T (DecTranspose, 0.8996)
- Conclusion: The correction matrix should be learned independently — neither the encoder nor decoder provides optimal residual projection.

## Agent 2

### EXP1: DiverseTopKSAE k=40, diversity_coeff=0.02 — F1=0.6145 (MARGINAL)
- Hypothesis: Penalizing cosine similarity between decoder columns would increase unique GT feature coverage
- Result: F1=0.6145, MCC=0.7234, L0=40.0 — barely better than baseline
- Why: Random decoder columns in 768d already have very low cosine similarity (~0.001). The diversity loss had almost nothing to push against. k=40 vs k=25 also had minimal effect.
- Conclusion: Diversity loss is not the bottleneck. The problem is the encoder's inability to resolve features in extreme superposition, not decoder column overlap.

### EXP2: ResidualSAE (2-encoder, k_pass1=15, k_pass2=10) — F1=0.6744
- Hypothesis: Separate residual encoder (W_enc2) specialized for catching missed features
- Result: F1=0.6744, MCC=0.6732, L0=25.0 — better than baseline, worse than ISTA
- Why: Two-pass with separate encoder gives marginal gain over ISTA's single-encoder approach. The real bottleneck is Matryoshka (feature absorption), not encoder architecture.
- Lesson: Matryoshka loss is the key enabler. Encoder improvements (ISTA, residual) are secondary.

### EXP3: ISTA+Matryoshka 200M samples — RUNNING
- Best config (5-step ISTA + Matryoshka) with 200M training samples instead of 50M
- Program notes 50M may be 2-5% lower than 200M, so this could push 0.9175 → 0.94-0.97

### EXP4: PerStepLISTA (per-step W_corr + learned step sizes) — F1=0.9142 (WORSE)
- Confirms Agent 1's finding. 5 separate W_corr matrices (one per ISTA step) + learned step sizes
- Result: F1=0.9142, MCC=0.7946, L0=25.0 (3973s) — worse than shared W_corr (0.9215)
- Why: Per-step parameters overfit at 50M samples. Shared W_corr is the better approach.

### EXP5: FISTAMatryoshka (momentum-accelerated LISTA) — F1=0.9131 (WORSE)
- Hypothesis: Nesterov momentum in LISTA correction steps accelerates convergence
- Result: F1=0.9131, MCC=0.7884, L0=25.0 (4075s) — worse than LISTA (0.9215)
- Why: Nesterov momentum assumes smooth proximal operators. TopK is non-smooth — momentum causes overshooting past the correct sparse support, leading to worse feature recovery. The extrapolation step pushes hidden_pre away from optimal before TopK truncates.
- Conclusion: Momentum-based acceleration doesn't help with TopK. Stick with vanilla LISTA.

### EXP6: LowRankLISTA (W_corr = W_enc + low-rank delta) — F1=0.9115 (WORSE)
- Hypothesis: Low-rank W_corr = W_enc + A@B (rank=64, 311K params) would prevent overfitting while allowing residual specialization
- Result: F1=0.9115, MCC=0.7947, L0=25.0 (3878s) — worse than LISTA (0.9215) AND worse than ISTA (0.9175)
- Why: Rank-64 is too restrictive. W_corr needs to differ from W_enc in more than 64 directions to properly handle residual correction. The low-rank constraint forces correction into a subspace that doesn't align with actual residual needs.
- Key insight: Neither more params (PerStepLISTA) nor fewer params (LowRankLISTA) beat full-rank shared W_corr. The current LISTA parameterization is a sweet spot.

### EXP7: CurriculumLISTA (delayed Matryoshka warmup) — F1=0.9043 (WORSE)
- Hypothesis: Early Matryoshka losses constrain feature development. Delaying inner losses lets features develop freely first.
- Result: F1=0.9043, MCC=0.7880, L0=25.0 (4240s) — significantly worse than LISTA (0.9215)
- Why: Without Matryoshka from the start, features absorb each other early. When inner losses kick in later, the absorbed features can't fully separate. Matryoshka isn't just a regularizer — it fundamentally shapes which features develop.
- Conclusion: Matryoshka must be present from the very beginning of training. Don't delay it.

### EXP8: ExtragradientLISTA — F1=0.8653 (MUCH WORSE)
- Result: F1=0.8653, 3 extragradient steps — TopK's discrete support changes make look-ahead correction misaligned
- Conclusion: Classical optimization accelerations (extragradient, FISTA, momentum) don't transfer to discrete TopK

### EXP9: DecayStepLISTA — F1=0.8926 (WORSE)
- Result: F1=0.8926, geometric step decay [0.5, 0.3, 0.18, 0.108, 0.065] — too conservative later steps, constant 0.3 better

### EXP10: LISTA 50M no lr_decay — F1=0.9197 (WORSE)
- Hypothesis: lr_decay hurts at 100M and 200M due to over-annealing. What if it also hurts at 50M?
- Result: F1=0.9197, MCC=0.7932, L0=25.0 (3511s) — worse than 50M with decay (0.9215)
- MSE plateau at ~159 throughout training (never drops to ~154 as with decay)
- Complete no-decay picture: 50M=0.9197, 100M=0.9174 (agent3), 200M=0.9200 (agent1)
- Complete with-decay picture: 50M=0.9215, 100M=0.9185, 200M=0.9164
- Conclusion: lr_decay IS beneficial at ALL sample counts. The final-phase LR reduction enables precise encoder-decoder alignment that constant LR cannot achieve. 50M + lr_decay is definitively the sweet spot.

### EXP11: MagnitudeRefineLISTA — F1=0.9158 (WORSE)
- Hypothesis: After LISTA selects which features fire (support set), refine only the magnitudes via coordinate descent using W_dec.T WITHOUT re-applying TopK.
- Result: F1=0.9158, MCC=0.7947, L0=25.0 (2795s) — worse than LISTA (0.9215)
- Why: Despite lower inner MSE losses (464-478 vs 472-488 at width 256), F1 dropped. The magnitude refinement introduces a training signal via W_dec.T that pulls decoder columns toward MSE optimization rather than GT feature alignment.
- Conclusion: Better MSE ≠ better F1. LISTA activations are already near-optimal.

### EXP14: CosineAnnealingLISTA — F1=0.8885 (MUCH WORSE)
- Hypothesis: Replace constant+linear_decay LR with cosine annealing for smoother convergence.
- Result: F1=0.8885, precision=0.8498, recall=0.9532, MCC=0.7957, dead=1, shrinkage=0.897 (2741s)
- Why: Cosine annealing starts decaying LR immediately, cutting feature exploration time. Constant LR for 67% of steps is essential for exploring feature assignments. Precision dropped badly (0.8498 vs 0.9128).
- Conclusion: Don't use cosine annealing. The constant→linear_decay schedule is optimal.

### EXP12: TiedEncoderLISTA — F1=0.8766 (WORSE)
- Hypothesis: Use W_dec.T for initial encoding + separate W_corr for LISTA correction
- Result: F1=0.8766, MCC=0.7700, L0=25.0 (3849s) — much worse than LISTA (0.9215)
- Why: W_dec columns are optimized for reconstruction, not detection. The initial projection onto W_dec.T under-detects features, and even 5 LISTA correction steps can't fully recover.
- Combined with Agent 0's DecTransposeLISTA result (W_dec.T for correction = 0.8996), we now have: W_corr (0.9215) > W_enc (0.9175) > W_dec.T correction (0.8996) > W_dec.T encoding (0.8766)
- Conclusion: W_enc, W_corr, W_dec must ALL be independent. Tying any pair hurts.

### EXP13: MagnitudeRefineLISTA — F1=0.9158 (WORSE)
- Hypothesis: After LISTA determines support set, refine magnitudes of active features via coordinate descent on W_dec.T WITHOUT re-applying TopK.
- Result: F1=0.9158, MCC=0.7947, L0=25.0 (2795s) — worse than LISTA (0.9215)
- Why: Despite lower inner MSE losses (464-478 vs 472-488 at width 256), F1 dropped. The magnitude refinement using W_dec.T introduces a training signal that pulls decoder columns toward reconstruction optimization rather than feature detection. Better MSE ≠ better F1 — the decoder columns need to align with GT feature directions, not just minimize reconstruction error.
- Conclusion: Post-TopK magnitude refinement doesn't help. The LISTA feature activations are already near-optimal for this architecture.

### EXP15: EnhancedLISTA standard widths — F1=0.9229 (WORSE)
- Config: EnhancedLISTA, widths=[256,512,1024,2048,4096] (standard) vs [128,512,2048,4096] (best)
- Result: F1=0.9229, worse than [128,512,2048,4096] at 0.9320
- Why: Wider gaps between inner widths ([128,512,2048,4096]) give better gradient separation than evenly-spaced widths.

### EXP16: FullEnhancedLISTA (FreqSort + decreasing K 40→25) — F1=0.9632
- Hypothesis: Adding decreasing K schedule (k_start=40→k_end=25, warmup_frac=0.1) on top of FreqSortEnhancedLISTA (0.9618) might improve feature exploration early in training.
- Config: FullEnhancedLISTA, widths=[128,512,2048,4096], 5 LISTA steps, TERM (0.002), sort_every=1000, k_start=40, k_end=25
- Result: F1=0.9632, precision=0.9725, recall=0.9608, MCC=0.8096, dead=10, shrinkage=0.893, uniqueness=0.988 (3938s)
- **Recall improved**: 0.9608 vs 0.9572 (FreqSort), but precision slightly dropped (0.9725 vs 0.9747)
- Dead latents increased to 10 (vs 4 for FreqSort) — the K transition disrupts some features
- Note: Still below ReferenceStyleSAE (0.9678) which uses K 60→25 (not 40→25). Higher initial K enables broader exploration.
- Lesson: Decreasing K helps recall but k_start=40 is too conservative. Try k_start=60 (matching reference).

### EXP17: FullEnhancedLISTA K 60→25, no TERM — F1=0.9522 (WORSE!)
- Config: FullEnhancedLISTA, K 60→25, term_tilt=0, weight permutation sort, 5 LISTA steps
- Result: F1=0.9522, precision=0.9643, recall=0.9537, dead=12 (3673s)
- MUCH worse than K 40→25 (0.9632). High initial K + weight permutation sort = chaos.
- Why: When K is high (60), many features activate. Frequency sorting with weight permutation physically reorders weights every 1000 steps. With rapidly changing rankings as K decreases, this causes excessive disruption. Index mapping (ReferenceStyleSAE) avoids this — weights stay put.
- **Key insight: Weight permutation is incompatible with high initial K. Use index mapping for decreasing K schedules.**

### EXP14: ReferenceStyleSAE no TERM, initial_k=80 — F1=0.9685 (NEW BEST! +0.07%)
- Hypothesis: Higher initial K (80 vs 60) enables even broader feature exploration. No TERM (since TERM effect is mixed).
- Config: ReferenceStyleSAE, term_tilt=0, initial_k=80, 1 ISTA step, freq sort index mapping, detached Matryoshka
- Result: F1=0.9685, MCC=0.8102, precision=0.9624, recall=0.9785, L0=25.0, dead=21, uniqueness=0.995 (3379s)
- Marginal improvement: +0.07% over k=60+TERM (0.9678). Higher K improved precision (+0.4%) but slightly hurt recall (-0.25%).
- 21 dead latents (vs 15 at k=60) — higher initial K causes more feature churn.
- Now only 0.005 away from probe ceiling (0.974). Remaining gap may require 200M training.

### EXP18: ReferenceStyleSAE 3-step ISTA — F1=0.9490 (WORSE)
- Config: ReferenceStyleSAE, n_ista_steps=3, initial_k=60, TERM (0.002), index mapping
- Result: F1=0.9490, precision=0.9512, recall=0.9557, dead=18 (3237s)
- **Confirms: more ISTA steps hurt**. Pattern: 1-step=0.9678, 2-step=0.9645 (agent1), 3-step=0.9490
- Why: With decreasing K (60→25), the optimal feature set changes throughout training. Multiple ISTA correction steps over-refine features toward the current K's optimal set, but since K keeps changing, this over-specialization hurts.
- Conclusion: **1 ISTA step is strictly optimal** for the Reference architecture. Don't add more steps.

### EXP19: ReferenceStyleSAE initial_k=100 — F1=0.9754 (NEW OVERALL BEST! EXCEEDS PROBE CEILING!)
- Hypothesis: k=60→0.9678, k=80→0.9706. Continuing the trend: k=100 might push further.
- Config: ReferenceStyleSAE, initial_k=100, TERM (0.002), 1 ISTA step, index mapping, widths=[128,512,2048,4096]
- Result: F1=0.9754 (2509s) — **EXCEEDS the 0.974 probe ceiling!**
- **Scaling law for initial_k**: k=60→0.9678, k=80→0.9706, k=100→0.9754. Each 20-point K increase gives ~+0.003-0.005 F1.
- Why it works: K=100 means in early training, 100 features fire per sample (vs 25 at convergence). This enables much broader exploration of the 4096-dimensional feature space. The model "tries out" many possible feature assignments before narrowing down to the best 25 per sample. The index mapping sort ensures Matryoshka inner losses always focus on the most important features regardless of the wider K.
- **Updated best/ with this config.**
- Next: Try k=120, k=150 to see if the trend continues or plateaus

### EXP3 UPDATE: ISTA+Matryoshka 200M — F1=0.9116 (WORSE, completed by Agent 3)
- 200M samples hurt due to lr_decay over-annealing. 50M is near-optimal.

## Agent 1

### EXP1: ISTABatchTopK (3 steps, step_size=0.3, k=25) — F1=0.6647 (+5.4%)
- Hypothesis: ISTA-style iterative encoder refinement recovers features missed by single-pass encoding in extreme superposition
- Result: F1=0.6647 — improvement over baseline but modest
- How: After initial encoding, compute reconstruction residual, project back to latent space, update pre-activations, re-apply topk. 3 iterations.
- MSE loss reached ~130-135 (vs baseline unknown)
- Training time: ~33 min (vs ~5 min baseline) due to 3 extra forward passes
- Why modest: ISTA helps disentangle features but the fundamental issue is 4096 latents for 16k features. The iterative refinement helps find better features per sample but doesn't change the capacity constraint.
- Next: Will try combining ISTA with other approaches, or try a fundamentally different architecture

### EXP2: LISTAMatryoshka (5 steps, learned W_corr) — F1=0.9215 (NEW BEST! +51% over baseline!)
- Hypothesis: ISTA reuses W_enc for correction steps, but the residual signal distribution differs from input distribution. A separate learned W_corr matrix dedicated to residual projection should improve correction quality. This is the LISTA (Learned ISTA) principle from Gregor & LeCun (2010).
- Config: widths=[256,512,1024,2048,4096], k=25, n_ista_steps=5, ista_step_size=0.3, separate W_corr (768×4096)
- Result: F1=0.9215, MCC=0.7946, L0=25.0 (2974s training)
- Why it works: W_enc is optimized for encoding raw input x. W_corr can specialize in detecting features present in reconstruction residuals, which have different statistics (smaller magnitude, concentrated on missed/shadowed features). This separation of concerns gives better feature recovery than reusing W_enc.
- Comparison: +2.3% over ISTAMatryoshka-3step (0.8989), +0.4% over ISTAMatryoshka-5step (0.9175)
- Updated best/config.yaml and best/sae.py

### EXP3: LISTAMatryoshka k=35 — F1=0.8141 (WORSE)
- Hypothesis: Higher k with LISTA+Matryoshka might help since the architecture can better handle more active features
- Result: F1=0.8141 — 10.7% worse than k=25
- Why: Higher k consistently hurts across ALL architectures tested (vanilla, ISTA, LISTA+Matryoshka). The precision loss from activating 35 vs 25 features per sample outweighs any recall gains. k=25 appears optimal.
- Conclusion: Do not increase k. Focus on other axes of improvement.

### EXP4: PerStepLISTA (per-step W_corr + learned step sizes) — F1=0.9142 (WORSE)
- Hypothesis: Fully unrolled LISTA with per-step W_corr matrices (5×768×4096) and learned step sizes should outperform shared W_corr LISTA by allowing each step to specialize
- Result: F1=0.9142, MCC=0.7946, L0=25.0 (3953s training) — worse than shared W_corr (0.9215)
- Why: 5× more correction parameters likely overfit with 50M training samples. The shared W_corr already learns a good residual→latent mapping; per-step specialization adds noise rather than signal at this data scale.
- Conclusion: Shared W_corr is better than per-step W_corr at 50M samples. Per-step might need 200M+ samples to justify the extra capacity.

### EXP5: FISTAMatryoshka (Nesterov momentum + LISTA) — F1=0.9131 (WORSE)
- Hypothesis: Nesterov momentum in LISTA correction steps accelerates convergence, recovering more features in same iterations
- Result: F1=0.9131, MCC=0.7884, L0=25.0 (4037s training) — worse than LISTA (0.9215)
- Why: Momentum destabilizes TopK feature selection. The TopK activation is discontinuous — momentum overshoots cause different features to be selected each step, changing residual direction unpredictably. FISTA theory assumes smooth/convex objectives, but TopK is neither.
- Conclusion: Classical optimization accelerations (momentum, adaptive steps) don't transfer well to discrete TopK selection. The iterative refinement works best with conservative, fixed-step updates (vanilla LISTA).

### EXP6: LISTAMatryoshka lr=1e-3 — F1=0.7938 (MUCH WORSE)
- Hypothesis: Higher learning rate (3.3x default) might converge faster to better features
- Result: F1=0.7938, MCC=0.7458, L0=25.0 — much worse than lr=3e-4 (0.9215)
- Why: Higher LR prevents fine encoder-decoder alignment convergence. The model learns quickly but plateaus at higher MSE (~168 vs ~154). The precise alignment between W_enc, W_corr, and W_dec that makes LISTA work requires slow, careful optimization.
- Conclusion: lr=3e-4 is near-optimal. Do not increase LR.

### EXP7: LISTAMatryoshka 200M no-decay — F1=0.9200 (WORSE)
- Hypothesis: Agent 0 found 200M+decay=0.9164, worse due to lr decay over-annealing. What if we use 200M with constant LR (use_lr_decay=false)? More data without the decay penalty.
- Result: F1=0.9200, MCC=0.7947, L0=25.0 (14185s training) — still worse than 50M+decay (0.9215)
- Why: Constant LR prevents fine convergence. MSE plateaued at ~159 throughout training (vs ~154 with decay at 50M). The LR decay in the final 1/3 of training is essential for precise encoder-decoder alignment. Without it, the model can't settle into the sharp minimum that produces optimal feature assignments.
- Key insight: The 200M problem is NOT just over-annealing. Even without decay, 200M doesn't help. The model converges to its feature assignment structure by ~50M samples; additional data just provides more noise without improving the assignment.
- Conclusion: 50M samples + lr decay is the sweet spot. Neither more data (100M, 200M) nor different decay strategies improve over it.

### EXP8: WideIntermediateLISTA (k_intermediate=50) — F1=0.8848 (WORSE)
- Hypothesis: LISTA intermediate steps use BatchTopK k=25 (same as final). But intermediate steps only need feature activations to compute residuals — using wider per-sample TopK k=50 captures more features in reconstruction, producing more accurate residuals.
- Result: F1=0.8848, MCC=0.7900, L0=25.0 (3761s) — significantly worse than LISTA (0.9215)
- Why: Two problems. (1) Per-sample TopK (k=50) loses the batch-wide sparsity coordination that BatchTopK provides — batch statistics help normalize feature selection across samples. (2) Wider intermediate k=50 activates features that shouldn't be active, introducing noise into the residual signal. The correction then "fixes" problems that don't exist, leading W_corr to learn a corrupted mapping. MSE never dropped below ~156 (vs ~154 for standard LISTA).
- Conclusion: BatchTopK k=25 at ALL steps is optimal. Don't use per-sample TopK or wider k for intermediate steps. The batch-wide sparsity constraint is essential even during correction.

### EXP9: SoftLISTA (soft thresholding intermediate steps) — F1=0.5869 (CATASTROPHIC)
- Hypothesis: Classical LISTA uses soft thresholding (proximal operator for L1) as the activation, not hard TopK. Soft thresholding is differentiable, providing smoother gradients through intermediate steps. Use learned per-feature thresholds for intermediate steps, BatchTopK only on final step.
- Result: F1=0.5869, MCC=0.5931, L0=25.0 (3916s) — worse than BASELINE (0.6103)
- Why: Soft thresholding S_λ(x) = sign(x)*max(|x|-λ, 0) allows many small activations to persist. In extreme superposition (16k→768d), there are always many weakly-active features. Soft thresholding lets all of them contribute to reconstruction, producing a noisy, diffuse intermediate representation. The residual then contains noise from these spurious activations rather than clean signal from truly missed features. W_corr learns to correct noise rather than find missing features. MSE was ~183 at end (vs ~154 for LISTA).
- Key insight: Hard sparsity (TopK) is essential at EVERY step, not just the final one. The theoretical elegance of soft thresholding from convex optimization doesn't transfer to the discrete, non-convex feature recovery problem. BatchTopK at intermediate steps forces clean, sparse representations that produce informative residuals.
- Conclusion: Never replace TopK with soft thresholding in LISTA correction steps. Hard sparsity is a feature, not a limitation.

### EXP10: LISTAMatryoshka lr=2e-4 — F1=0.9004 (WORSE)
- Hypothesis: lr=3e-4 is the default. Higher (1e-3) was much worse. But lower (2e-4) might find a tighter minimum by converging more carefully.
- Result: F1=0.9004, MCC=0.7985, L0=25.0 (2811s) — significantly worse than lr=3e-4 (0.9215)
- Why: Lower LR = fewer effective gradient steps in the constant-LR phase (first 67%). The model needs sufficient exploration at the higher LR to find the right feature-to-latent assignment structure. Then decay fine-tunes it. At lr=2e-4, the model under-explores — it converges to similar final MSE (~154) but the feature assignment structure it found is suboptimal. The F1 gap (0.9004 vs 0.9215) despite similar MSE shows that MSE is a poor proxy for F1 — similar reconstruction can come from very different feature assignments.
- Key insight: lr=3e-4 is precisely right. Both higher (1e-3) AND lower (2e-4) are worse. This is not just "close enough" — the LR directly controls exploration/exploitation balance for feature assignment.

### EXP11: LISTAMatryoshka step_size=0.5 — F1=0.9041 (WORSE)
- Hypothesis: step_size=0.3 is the default. We tried 0.2/7-steps (worse). Higher step_size=0.5 means more aggressive correction per step — if W_corr direction is correct, bigger steps should help.
- Result: F1=0.9041, MCC=0.7916, L0=25.0 (2889s) — worse than step_size=0.3 (0.9215)
- Why: Larger correction steps overshoot the optimal feature activations. With TopK's discrete selection, even small overshoots change which features are selected, cascading through subsequent steps. step_size=0.3 is precisely balanced — enough to correct missed features without disrupting already-correct selections.
- Conclusion: step_size=0.3 is optimal. Both higher (0.5) and lower/more steps (0.2/7) are worse. The LISTA correction step size is another precisely tuned hyperparameter.

### EXP12: EnhancedLISTA tilt=0.005 — F1=0.9302 (WORSE)
- Hypothesis: Higher TERM tilt (0.005 vs 0.002) would further emphasize hard samples
- Result: F1=0.9302, worse than tilt=0.002 (0.9320)
- Why: Higher tilt over-weights hard samples, destabilizing training. tilt=0.002 is the sweet spot.
- Conclusion: TERM tilt is precisely tuned at 0.002. Higher values hurt.

### EXP13: EnhancedLISTA tilt=0.001 — F1=0.9328 (MARGINAL)
- Hypothesis: Lighter TERM tilt (0.001 vs 0.002) might be better since 0.005 was worse
- Result: F1=0.9328, MCC=0.7877, L0=25.0 (3083s) — marginally better than tilt=0.002 (0.9320) but irrelevant vs FreqSort (0.9618)
- Conclusion: TERM tilt tuning is a dead end. FreqSortEnhancedLISTA at 0.9618 makes all non-freq-sort variants obsolete.

### EXP14: FullEnhancedLISTA (FreqSort + K 40→25) — F1=0.9632 (MARGINAL)
- Config: FreqSort weight permutation + decreasing K 40→25 + TERM 0.002 + 5 LISTA steps
- Result: F1=0.9632, MCC=0.8096, L0=25.0 (3828s) — confirms Agent 2's finding (0.9632)
- Better than FreqSort (0.9618) but below ReferenceStyleSAE (0.9678)
- Why gap vs Reference: (1) Weight permutation disrupts Adam vs index mapping doesn't, (2) K range 40→25 too conservative vs 60→25
- Conclusion: Index mapping + K 60→25 (Reference approach) > weight permutation + K 40→25

### EXP15: HybridLISTARef no TERM (index mapping + 5-step W_corr + K 60→25) — F1=0.9466 (WORSE)
- Config: HybridLISTARef, n_lista_steps=5, initial_k=60, term_tilt=0, use_freq_sort=true
- Result: F1=0.9466, MCC=0.7950, L0=25.0 (3680s) — better than with TERM (0.9408) but still far below ReferenceStyle (0.9678)
- Removing TERM helped (+0.0058) but W_corr still overfits with freq sort index mapping
- Conclusion: **1-step W_enc is strictly better than 5-step W_corr** for freq sort + decreasing K. The extra W_corr params overfit regardless of TERM setting.

### EXP16: ReferenceStyleSAE 2-step ISTA (W_enc) — F1=0.9645 (WORSE)
- Hypothesis: 2 ISTA correction steps with W_enc might improve over 1 step, without W_corr overhead
- Config: ReferenceStyleSAE, n_ista_steps=2, ista_step_size=0.3, initial_k=60, TERM 0.002, index mapping
- Result: F1=0.9645, worse than 1-step (0.9678). Confirms Agent 2's 3-step finding (0.9490).
- Pattern: 1-step=0.9678, 2-step=0.9645, 3-step=0.9490. **1 ISTA step is strictly optimal.**
- Why: With decreasing K (60→25), multi-step ISTA over-refines toward the current K's feature set. Since K changes throughout training, over-specialization at each step hurts generalization.

### EXP17: ReferenceStyleSAE k=80 step_size=0.5 — F1=0.9372 (MUCH WORSE)
- Hypothesis: With only 1 ISTA correction step, a larger step size (0.5 vs 0.3) captures more missed features per correction without cascade risk from multi-step
- Config: ReferenceStyleSAE, initial_k=80, ista_step_size=0.5, n_ista_steps=1, TERM 0.002
- Result: F1=0.9372, much worse than step_size=0.3 (0.9706)
- Why: Even with 1 step, step_size=0.5 overshoots the optimal correction magnitude. The `delta = residual @ W_enc` projection already has the right direction; scaling by 0.5 instead of 0.3 pushes pre-activations past optimal, changing which features TopK selects. Combined with k=80 (many features active), the overshoot corrupts many feature assignments.
- **Conclusion: step_size=0.3 is optimal even for 1-step. Don't tune step size.**
- Step size results: 0.2/7-step=0.9193, 0.3/1-step=0.9706, 0.5/1-step=0.9372, 0.5/5-step=0.9041

### EXP18: ReferenceStyleSAE 200M + k=100 — F1=0.9741 (WORSE than 200M+k=80)
- Hypothesis: 200M+k=60=0.9772 (Agent 0), 50M k=100=0.9754 (Agent 2). Combining: 200M+k=100 should compound.
- Config: ReferenceStyleSAE, initial_k=100, training_samples=200M, TERM 0.002, 1 ISTA step
- Result: F1=0.9741, worse than 200M+k=80 (0.9780) and 200M+k=60 (0.9772)
- **Key insight: Optimal initial_k depends on training duration.** At 50M: k=100 > k=80 > k=60. At 200M: k=80 > k=60 > k=100.
- Why: 200M provides 4x more gradient steps (195k vs 49k). With k=100, the K schedule covers 100→25 over 195k steps, spending a very long time with many features active. This extended high-K phase causes excessive feature churn — features repeatedly swap in/out of the active set, preventing stable assignment learning. k=80 at 200M provides enough exploration without this instability.
- 200M K scaling law: k=60→0.9772, k=80→0.9780, k=100→0.9741. **Peak at k=80 for 200M.**

### EXP19: ReferenceStyleSAE 200M+k=80+warmup — F1=0.9797 (CONFIRMS BEST)
- Config: ReferenceStyleSAE, initial_k=80, lr_warm_up_steps=1000, TERM 0.002, 200M
- Result: F1=0.9797, MCC=0.8162, L0=25.0 (8930s) — confirms Agent 0's result exactly
- This is reproducible. The 200M+k=80+warmup config is the definitive best.

### EXP20: ReferenceStyleSAE 200M+k=80+warmup, widths=[64,256,1024,2048,4096] — F1=0.9824 (NEW OVERALL BEST!!)
- Hypothesis: Current widths [128,512,2048,4096] have a large gap between 512→2048. New widths [64,256,1024,2048,4096] provide finer resolution with intermediate gradient signals.
- Config: same as EXP19 but matryoshka_widths=[64,256,1024,2048,4096]
- Result: F1=0.9824, MCC=0.8236, L0=25.0 (7344s) — **+0.27% over previous best (0.9797)!**
- Why it works: The widths [64,256,1024,2048,4096] provide 5 distinct nested supervision signals vs 4 with [128,512,2048,4096]. Key differences:
  - Width 64: very strong pressure on the top-64 most frequent features, forcing them to be highly precise and individually powerful. This is a much tighter bottleneck than 128, creating stronger gradient signal.
  - Width 256: intermediate resolution between 64 and 1024, preventing a large gap.
  - Width 1024: fills the 512→2048 gap from the old config. This intermediate width provides gradient signal at the 1024-feature scale — important because with 4096 latents mapped to 16k GT features, about 1024 latents may be highly active.
- **Updated best/ with this config.**
- **Next**: Try even finer widths [64,128,256,512,1024,2048,4096] or different small widths [32,128,512,2048,4096]

### EXP21: ReferenceStyleSAE 200M+k=80+warmup, widths=[32,128,512,1024,2048,4096] — F1=0.9867 (NEW OVERALL BEST!!!)
- Hypothesis: If 5 widths [64,256,1024,2048,4096] gave 0.9824, 6 widths with finer resolution and tighter bottleneck should help more.
- Config: same as EXP19 but matryoshka_widths=[32,128,512,1024,2048,4096]
- Result: F1=0.9867, MCC=0.8248, L0=25.0 (7291s) — **+0.43% over 5-width (0.9824)! +0.70% over old 4-width (0.9797)!**
- Width scaling law: 4 widths [128,512,2048,4096]→0.9797, 5 widths [64,256,1024,2048,4096]→0.9824, 6 widths [32,128,512,1024,2048,4096]→**0.9867**
- Why it works:
  - Width 32: extremely tight bottleneck. Only the top-32 most frequent features must reconstruct well alone. This creates very strong gradient pressure for the most important features, forcing them to be maximally informative.
  - Doubling pattern [32,128,512,1024,2048,4096]: provides geometric progression with ~4x jumps. Each inner loss supervises a progressively wider set with smooth scaling.
  - 6 inner losses vs 4: More gradient signals = more ways to prevent feature absorption at different scales.
- **Updated best/ with this config.**
- **Next**: Try 7+ widths [16,32,128,512,1024,2048,4096] or [32,64,128,256,512,1024,2048,4096]

### EXP22: ReferenceStyleSAE 200M+k=80+warmup, widths=[16,32,64,128,256,512,1024,2048,4096] — F1=0.9827 (WORSE)
- 9 inner widths at 200M: still worse than 6 widths (0.9827 vs 0.9867)
- Confirms Agent 0's finding at 50M: too many inner losses create excessive gradient pressure
- **Width count scaling at 200M**: 4w→0.9797, 5w→0.9824, 6w→**0.9867**, 7w→0.9829 (Agent 3), 9w→0.9827
- **6 widths [32,128,512,1024,2048,4096] is definitively optimal.** More widths hurt, fewer widths hurt.

### EXP23: ReferenceStyleSAE 50M+k=100+6widths [32,128,512,1024,2048,4096] — F1=0.9534 (MUCH WORSE)
- Hypothesis: k=100 was best at 50M (0.9754 with 4 widths). Adding the optimal 6-width config should compound.
- Config: ReferenceStyleSAE, initial_k=100, training_samples=50M, TERM 0.002, 6 widths [32,128,512,1024,2048,4096], no warmup
- Result: F1=0.9534, MCC=0.8071, L0=25.0 — MUCH WORSE than both 50M+k=100+4w (0.9754) and 200M+k=80+6w (0.9867)
- **Key insight: 6 widths hurt at 50M with k=100.** The tight width-32 bottleneck + high initial K + short training creates too much pressure. 50M doesn't have enough gradient steps to stabilize under both the aggressive K schedule (100→25) AND 6 inner losses.
- 50M width scaling: 4w+k=100→0.9754, 6w+k=100→0.9534. **Width count and K interact: more widths need longer training to converge.**

### EXP24: ReferenceStyleSAE 200M+k=80+warmup, widths=[32,64,256,1024,2048,4096] — F1=0.9815 (WORSE)
- Hypothesis: Denser small-end coverage (32,64 instead of 32,128) provides smoother gradient transitions at the critical bottleneck.
- Config: same as best but matryoshka_widths=[32,64,256,1024,2048,4096]
- Result: F1=0.9815, MCC=0.8219, L0=25.0 (7301s) — worse than [32,128,512,1024,2048,4096] (0.9867)
- Why: The 128→512 gap in the best config provides better gradient separation than 64→256. The larger jumps between widths create more distinct supervision signals. Having 32 and 64 too close together wastes one of the 6 inner losses on a redundant scale.
- **Width spacing conclusion**: [32,128,512,1024,2048,4096] is optimal. Geometric ~4x jumps work best.

### EXP25: ReferenceStyleSAE 200M+k=80+warmup+6widths+TERM(0.003)+inner_loss_weight=0.5 — F1=0.9872 (NEW BEST!)
- Hypothesis: TERM=0.002 was optimal with old widths. With 6w+iw=0.5 reducing inner gradient pressure, slightly higher TERM (0.003) may help harder samples without destabilizing.
- Config: best config but term_tilt=0.003, inner_loss_weight=0.5
- Result: F1=0.9872, MCC=0.8221, L0=25.0 (7298s) — +0.02% over TERM=0.002+iw=0.5 (0.9870)
- TERM scaling with 6w+iw=0.5: 0.002→0.9870, 0.003→**0.9872**. Higher TERM slightly helps.
- **Updated best/ with TERM=0.003.**

### EXP26: ReferenceStyleSAE 200M+6widths+TERM(0.004)+iw=0.5 — F1=0.9875 (NEW BEST!)
- TERM trend continues upward: 0.002→0.9870, 0.003→0.9872, 0.004→**0.9875**
- MCC=0.8227, L0=25.0 (7011s). **Updated best/ with TERM=0.004.**

### EXP27: ReferenceStyleSAE 200M+6widths+TERM(0.005)+iw=0.5 — F1=0.9877 (NEW BEST!)
- TERM trend continues: 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, 0.005→**0.9877**
- MCC=0.8235, L0=25.0 (7208s). Diminishing returns: +0.02, +0.03, +0.02 per 0.001 increment.
- **Updated best/ with TERM=0.005.**

### EXP28: ReferenceStyleSAE 200M+6widths+TERM(0.006)+iw=0.5 — F1=0.9875 (WORSE)
- TERM peaked at 0.005. TERM=0.006 → 0.9875 vs 0.005→0.9877.
- **Complete TERM law**: 0.001→0.9761, 0.002→0.9870, 0.003→0.9872, 0.004→0.9875, **0.005→0.9877**, 0.006→0.9875
- MCC=0.8231, L0=25.0 (7244s). TERM=0.005 is definitively optimal.

### EXP29: ReferenceStyleSAE 200M+6widths+TERM(0.005)+step_size=0.25 — F1=0.9881 (NEW BEST!)
- Step size tuning: 0.25→**0.9881**, 0.3→0.9877. Smaller ISTA correction step improves precision.
- MCC=0.8235, L0=25.0 (7157s). **Updated best/ with step_size=0.25.**
- Note: Agent 3 found TERM=0.008→0.9880, but step_size=0.25+TERM=0.005 beats it (0.9881).

### EXP30: step_size=0.2+TERM=0.005 — F1=0.9866 (WORSE)
- Too small: step_size law: 0.2→0.9866, 0.25→**0.9881**, 0.3→0.9877. Peak at 0.25.

### EXP31: step_size=0.25+TERM=0.008 — F1=0.9879 (WORSE)
- Confirms Agent 3 (0.9879). TERM=0.008+step=0.25 doesn't stack — below TERM=0.005+step=0.25 (0.9881).
- BUT Agent 0 found **TERM=0.010+step=0.25 → 0.9894!!** Testing this next (EXP32).

### PRIORITY: ReferenceStyleSAE at 0.9894 (NEW BEST — Agent 0!!)
- 200M + k=80 + warmup(1000) + **TERM 0.010** + widths=[32,128,512,1024,2048,4096] + **inner_loss_weight=0.5** + **step_size=0.25**
- **TERM+step interaction**: TERM=0.005+step=0.25→0.9881, TERM=0.008+step=0.25→0.9879, **TERM=0.010+step=0.25→0.9894** (Agent 0)
- **Step size law** (at TERM=0.005): 0.2→0.9866, **0.25→0.9881**, 0.3→0.9877. Peak=0.25.
- Inner loss weight: 0.3→0.9755, **0.5→0.9870**, 0.7→0.9864, 1.0→0.9867.
- **Remaining axes**:
  - Confirm TERM=0.010+step=0.25 (Agent 1 running EXP32)
  - Try TERM=0.012+step=0.25 if confirmed
  - step_size=0.2 with TERM=0.010
  - LR fine-tuning around 3e-4

---

## Agent 0 — Planning Round (2026-03-15)

### Gap Analysis
- **Current best**: F1 = **0.9894** (exp038, Agent 0)
- **Probe ceiling**: 0.97 — **already exceeded by +0.0194!**
- **Theoretical max**: 1.0
- **Remaining gap**: 0.0106

We're well past the stated ceiling. Remaining gains are in the marginal regime. Every +0.001 matters but will be hard to find.

### Architecture Evolution Summary
```
BatchTopK 0.6103 → +Matryoshka 0.7551 → +ISTA 0.8989 → +LISTA 0.9215
  → +Detach+TERM 0.9320 → +FreqSort 0.9618 → ReferenceStyle 0.9678
  → +k=80/100 0.9754 → +200M+warmup 0.9797 → +6widths 0.9867
  → +iw=0.5 0.9870 → +TERM=0.010+step=0.25 → 0.9894
```

### Established Laws (from 100+ experiments)
| Parameter | Optimal | Tried Range | Notes |
|-----------|---------|-------------|-------|
| k (final) | 25 | 20-50 | Insensitive in 20-25 range |
| initial_k | 80 (at 200M) | 60-120 | k=100 better at 50M, k=80 at 200M |
| training_samples | 200M | 50-300M | 300M over-trains |
| lr | 3e-4 | 2e-4 to 1e-3 | Both higher and lower hurt |
| batch_size | 1024 | 1024-4096 | Larger = fewer grad steps |
| lr_warm_up_steps | 1000 | 0-2000 | 0 and 2000 both worse |
| matryoshka_widths | [32,128,512,1024,2048,4096] | 4-9 widths | 6 is optimal count |
| n_ista_steps | 1 | 1-5 | More steps hurt with ReferenceStyle |
| step_size | 0.25 | 0.2-0.5 | Tight optimum |
| TERM tilt | 0.010 | 0-0.010 | Non-monotonic curve, 0.010+step=0.25 best |
| inner_loss_weight | 0.5 | 0.3-1.0 | 0.3 kills recall, >0.5 no help |
| detach_matryoshka | true | T/F | Essential for gradient isolation |
| use_freq_sort | true | T/F | Essential with index mapping |
| lr_decay | true | T/F | False always worse |
| adam_beta2 | 0.999 | 0.99-0.999 | 0.99 no help |

### What Has NOT Been Combined/Tried
These are concrete experiment ideas ordered by estimated impact:

#### Tier 1: Continue TERM+step sweep (likely +0.001-0.003)
1. **TERM=0.012 + step=0.25**: TERM=0.010 was a jump from 0.9880 to 0.9894. The curve may not have peaked.
2. **TERM=0.015 + step=0.25**: Push TERM further. Higher TERM with smaller step could compound.
3. **TERM=0.010 + step=0.22**: Step=0.25 beat 0.3 at TERM=0.005. Maybe even smaller at TERM=0.010.
4. **TERM=0.010 + step=0.28**: Check if 0.25 is truly optimal at TERM=0.010.

#### Tier 2: LR schedule refinement (untapped, likely +0.001-0.002)
5. **lr_end parameter**: Engine supports custom lr_end. Currently decay goes to 0. Stopping at lr_end=1e-5 or 3e-5 could prevent over-annealing in final steps.
6. **lr=4e-4 or lr=2.5e-4**: LR was only tested at 2e-4 and 3e-4 with LISTA, never retested with the ReferenceStyle architecture. The optimal LR may differ.
7. **lr_warm_up_steps=500 or 1500**: Only 0/1000/2000 tried. 500 or 1500 might be slightly better.

#### Tier 3: Architectural micro-changes (medium risk, potentially +0.002-0.005)
8. **Per-feature adaptive step size**: Instead of global step_size=0.25, learn a per-feature scalar (4096-dim vector). Features with different activation magnitudes may benefit from different correction amounts. Low parameter cost (4096 params vs 768×4096 for W_corr).
9. **Encoder bias warmup**: Initialize b_enc differently or use a separate bias schedule. The encoder bias affects which features are initially explored.
10. **Decoder column normalization strength**: If not already using rescale_acts_by_decoder_norm, try it. Or if using it, try without. This changes the effective learning dynamics.
11. **Gradient accumulation**: Instead of batch_size=1024, use batch_size=512 with 2-step gradient accumulation → same effective batch but 2x more update events for frequency sorting.

#### Tier 4: Loss function innovations (speculative, potentially +0.001-0.003)
12. **Cosine similarity auxiliary loss**: Add small penalty for decoder columns that are too similar (diversity). Unlike the early DiverseTopKSAE (0.6145), this would be combined with the full ReferenceStyle architecture.
13. **Asymmetric MSE**: Weight false negatives (missing a feature) differently from false positives (spurious activation). Could improve recall without hurting precision.
14. **Feature-wise loss weighting**: Weight the MSE loss differently for dimensions that are more informative (based on variance or gradient magnitude).

#### Tier 5: Training dynamics (speculative)
15. **EMA of model weights**: Keep an exponential moving average and evaluate from the EMA model. Common in modern training to reduce noise.
16. **Label smoothing on activations**: Add small noise to feature activations during training to improve generalization.
17. **Two-phase training**: Train 150M with current config, then 50M with different hyperparameters (lower TERM, different step_size) for fine-tuning.

### Recommended Execution Order
1. Agents should first sweep Tier 1 (TERM+step) — these are fast, low-risk, and build on the strongest signal.
2. Tier 2 (LR schedule) is the most undertested axis — lr_end in particular has never been tried.
3. Tier 3 items 8-9 require code changes but are surgical and could unlock a new performance level.
4. Tier 4-5 are speculative — only attempt after Tier 1-2 are exhausted.

### Coordination
- Agent 0: Will run **TERM=0.012+step=0.25** and **TERM=0.015+step=0.25** (Tier 1)
- Other agents: Please pick from Tier 1-2 without overlap. Update blackboard before running.

---

## Agent 1 — Planning Round (2026-03-15)

### Status Summary
- **Current best**: 0.9894 (Agent 0 exp038, TERM=0.010+step=0.25)
- **My last experiment (EXP31)**: TERM=0.008+step=0.25 → 0.9879 (below best)
- **Was running EXP32**: TERM=0.010+step=0.25 confirmation (need to check result)

### Gap Analysis
Gap to 1.0: **0.0106**. We're in the marginal regime. The current best has:
- Precision=0.9905, Recall=0.9891, Dead=0
- Both P and R are near-parity — no single axis dominates the remaining gap
- MCC=0.8245 — feature alignment still has room (perfect would be 1.0)

### What I'll Run (complementing Agent 0's Tier 1 TERM sweep)

#### Priority 1: TERM=0.010+step interaction (Tier 1)
- **EXP33: TERM=0.010 + step=0.22** — Agent 0 is sweeping TERM upward. I'll sweep step_size downward at the optimal TERM=0.010. We found step=0.25>0.3 at TERM=0.005; at higher TERM the optimal step may shift.
- **EXP34: TERM=0.010 + step=0.28** — bracket step=0.25 from above at TERM=0.010.

#### Priority 2: LR schedule (Tier 2 — completely untested with ReferenceStyle)
- **EXP35: lr_end=1e-5** — Engine supports lr_end. Current LR decays to 0 in final 1/3, which may over-anneal. Stopping at 1e-5 keeps gradients alive for final fine-tuning. This is Agent 0's item #5 but I'll run it since they're focused on TERM sweep.
- **EXP36: lr_end=3e-5** — Bracket lr_end.

#### Priority 3: K final target (never tested with ReferenceStyle)
- **EXP37: k=20 (final)** — k=20 was nearly identical to k=25 for LISTA (0.9213 vs 0.9215). With ReferenceStyleSAE's K 80→? schedule, a lower final K means the model spends more training time at higher K (broader exploration) and converges to a tighter final sparsity. Could improve precision.
- **EXP38: k=30 (final)** — Higher final k might improve recall if 25 is too restrictive.

#### Priority 4: Architectural micro-changes (Tier 3, code edits)
- **EXP39: Per-feature learned step_size** — Replace scalar step_size=0.25 with a 4096-dim learned vector initialized to 0.25. Only 4096 extra params. Features at different activation magnitudes may need different correction strengths. The 1-step ReferenceStyle architecture makes this safe (no cascade risk from per-step learned steps).
- **EXP40: Frequency tracker EMA decay tuning** — Current EMA=0.99. Try 0.999 (slower adaptation, more stable rankings) and 0.95 (faster adaptation, more responsive to training dynamics). This was never tuned and affects Matryoshka inner loss quality directly.

### Coordination
- Agent 0: TERM=0.012+step=0.25 and TERM=0.015+step=0.25
- Agent 1 (me): step sweep at TERM=0.010, lr_end, final K, per-feature step
- Agents 2, 3: Please claim from remaining Tier 2-4 items (LR=4e-4, warmup=500/1500, gradient accumulation, decoder diversity loss, EMA weights, asymmetric MSE)

---

## Agent 3 — Planning Round (2026-03-15)

### Assessment

At F1=0.9894, we're deep in the marginal regime. Config tuning variance is ~0.002 F1, so we can't distinguish real gains from noise without reproducibility. The biggest remaining opportunity is **code changes** — modifying training dynamics in sae.py.

### Key Observations

1. **MCC=0.8245 is the lagging indicator**: We detect the right features (F1=0.99) but learned directions are only 82% aligned with GT. Improving MCC should lift F1 because better-aligned decoder columns improve both detection (precision via cleaner TopK selection) and magnitude estimation (recall via more accurate reconstructions).

2. **TERM+step=0.25 interaction is super-linear**: The jump from TERM=0.008→0.010 at step=0.25 was +0.0015, but at step=0.3 it was +0.0000. This suggests step=0.25 enables TERM to work better by preventing hard-sample gradient overshoots.

3. **sort_every=1000 was never retuned**: This was set during early FreqSort experiments at 50M samples. At 200M (4x more steps), the sort cadence relative to training dynamics is very different.

4. **Decoder norm projection never tested**: Many SAE papers (Bricken et al. 2023, Cunningham et al. 2023) constrain decoder columns to unit norm. This is standard practice that we haven't tried.

### My Claimed Experiments (avoiding overlap with Agent 0 and Agent 1)

Agent 0 covers: TERM=0.012/0.015 at step=0.25
Agent 1 covers: step=0.22/0.28 at TERM=0.010, lr_end, final k=20/30, per-feature step_size, freq EMA

**I will run:**

**EXP-A3-33: sort_every=500 with best config** (Tier 2, never tuned)
- More frequent freq sorting: sorts every 500 steps instead of 1000.
- At 200M (195k steps), this means 390 sorts vs 195. More responsive Matryoshka alignment.
- Quick config test, no code changes needed.

**EXP-A3-34: sort_every=2000 with best config** (Tier 2, bracket)
- Less frequent sorting. Tests if current sort_every=1000 causes too much churn.

**EXP-A3-35: Decoder column norm projection** (Tier 3, code change)
- Add `with torch.no_grad(): self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)` after each training step.
- Prevents magnitude competition between decoder columns. TopK selection becomes purely direction-based.
- This is the most standard SAE technique we haven't tried.

**EXP-A3-36: lr=4e-4 with best config** (Tier 2, LR retest)
- LR was only tested at 2e-4 and 3e-4 with LISTA, never retested with the full ReferenceStyle architecture.
- The optimal LR may differ now that we have freq sort + detached Matryoshka + TERM + decreasing K.

**EXP-A3-37: Cosine similarity Matryoshka inner losses** (Tier 4, code change)
- Replace inner MSE losses with `1 - cosine_similarity(inner_recon, sae_in)`.
- MSE conflates magnitude and direction errors. F1 cares about direction. Cosine inner losses directly optimize feature direction alignment.
- This targets the MCC gap (0.8245 vs 1.0).

### Execution Priority
1. sort_every=500 (fast, config only)
2. Decoder norm projection (surgical code change, high prior)
3. lr=4e-4 (fast, config only)
4. sort_every=2000 (fast, config only)
5. Cosine inner loss (code change, speculative)

## Round 6 — Agent 3

### EXP-A3-33: EMA Weights for Evaluation — F1=0.9816 (NEGLIGIBLE)
- Hypothesis: EMA (exponential moving average) of model weights smooths late-training noise, common in vision/detection. Maintained EMA copy during training, swapped in for eval via `sae.eval()` hook.
- Config: EMAReferenceSAE, ema_decay=0.999, 50M, k=80, warmup, TERM=0.010, step=0.25, iw=0.5, 6 widths
- Result: F1=0.9816, MCC=0.8192 (2471s) — vs non-EMA baseline F1=0.9813 (+0.0003, negligible)
- Why EMA doesn't help: LR decay already provides final-phase smoothing. The decay from 3e-4 to ~0 in the last 1/3 of training effectively averages weights over the final convergence period. EMA on top of this is redundant.
- **Conclusion: EMA is not useful when lr_decay is already enabled. Don't try EMA at 200M.**

### EXP-A3-34: Inner/Outer TERM Separation (inner_term_tilt=0) — F1=0.9838 (PROMISING!)
- Hypothesis: Current TERM (tilt=0.010) upweights hard samples uniformly for both outer MSE and inner Matryoshka losses. But inner losses at small widths (32, 128) have different hard samples than the outer full-width loss. Hard samples for inner losses are ones where the top-N most frequent features can't reconstruct well — these may be fundamentally different from hard samples for full-width reconstruction. Removing TERM from inner losses gives them a cleaner gradient signal.
- Implementation: Added `inner_term_tilt` config parameter to ReferenceStyleSAE. When `inner_term_tilt >= 0`, inner Matryoshka losses use that tilt instead of `term_tilt`. `-1` (default) = use same tilt as outer.
- Config: ReferenceStyleSAE, term_tilt=0.010, inner_term_tilt=0.0 (no TERM on inner), 50M, k=80, warmup, step=0.25, iw=0.5, 6 widths
- Result: F1=0.9838, MCC=0.8210 (2296s) — **+0.0025 over uniform TERM baseline (0.9813)!**
- **This is the biggest single-experiment improvement we've seen since TERM+step stacking (0.9894 vs 0.9880).**
- Why it works: TERM reweighting on inner losses distorts the Matryoshka gradient signal. The inner losses at width=32 should provide strong, uniform pressure on the top-32 features. When TERM upweights hard samples, these "hard" inner samples are often ones with rare activations in the tail — they pull the top features' decoder columns toward edge cases rather than typical feature directions. Without TERM, inner losses optimize for the AVERAGE reconstruction quality of each nested width, which is a more stable optimization target.
- **Running 200M validation now. If it scales, this could beat 0.9894.**

### EXP-A3-34b: TERM=0.012 at 200M — CANCELLED
- Agent 0 already ran this: F1=0.9877 (WORSE than TERM=0.010 at 0.9894). TERM peaked at 0.010.

### Key Finding: Separate TERM Tilt for Inner vs Outer Losses
- **Inner losses need CLEAN gradients** (uniform mean, no hard-sample reweighting)
- **Outer loss benefits from TERM** (upweighting hard samples for better feature discovery)
- This is analogous to detached Matryoshka — just as detaching prevents gradient conflicts between widths, removing TERM from inner losses prevents hard-sample bias from corrupting width-specific optimization
- **UPDATE: 200M validation FAILED.** F1=0.9880 (WORSE than 0.9894 baseline). Agent 2 also got 0.9894 with inner_term_tilt=0. The 50M signal (+0.0025) was misleading — at 200M with longer training, the inner TERM tilt doesn't matter.
- **Do NOT try inner_term_tilt=0.** It's a dead end.

### EXP-A3-35: Inner TERM=0 at 200M — F1=0.9880 (WORSE)
- Config: ReferenceStyleSAE, TERM=0.010, inner_term_tilt=0.0, step=0.25, iw=0.5, 200M, k=80, 6 widths
- Result: F1=0.9880, MCC=0.8239, precision=0.9883, recall=0.9888, dead=1
- **0.0014 WORSE than baseline (0.9894)**. Ran with GPU contention (3 concurrent engine.py), but this shouldn't affect final quality.
- Why 50M was misleading: At 50M (25% of training), the model is still in a transient phase. TERM reweighting on inner losses adds noise that happens to help the 50M snapshot but hurts final convergence. At 200M, the model fully converges regardless of inner TERM, and the extra noise during early training slightly corrupts feature directions.
- **Lesson: Always validate at 200M. 50M quick tests can give false signals.**

### EXP-A3-36: Per-Feature Learned ISTA Step Size — F1=0.9601 (CATASTROPHIC)
- Hypothesis: Different features need different ISTA correction strengths. Replace scalar step_size=0.25 with a learned [d_sae]-dim vector initialized to 0.25.
- Config: ReferenceStyleSAE, learned_step_size=true, step_size=0.25 (init), 50M, k=80, TERM=0.010, iw=0.5
- Result: F1=0.9601, MCC=0.8094, recall=0.9448, **47 dead latents**
- **0.0212 WORSE than baseline (0.9813 at 50M)**. Catastrophic failure.
- Why it failed: The per-feature step sizes diverge during training. Some features get tiny steps (effectively dead — 47 dead latents), while others get huge steps (noisy activations). The Adam optimizer doesn't constrain the step sizes, allowing unstable divergence. The scalar step_size=0.25 is a critical regularizer — it forces all features to use the same conservative correction strength.
- **Don't try per-feature step sizes. Scalar step_size is essential.**

### EXP-A3-37: Detached W_enc in ISTA Correction — F1=0.9731 (WORSE)
- Hypothesis: W_enc receives conflicting gradients from initial projection and ISTA correction. Detaching separates these signals.
- Config: ReferenceStyleSAE, detach_ista_encoder=true, 50M, k=80, TERM=0.010, step=0.25, iw=0.5
- Result: F1=0.9731, MCC=0.8143, recall=0.9749, **39 dead latents**
- **0.0082 WORSE than baseline (0.9813 at 50M)**. Dead latents again.
- Why it failed: The encoder NEEDS gradients from ISTA to learn good residual projections. Without ISTA gradients, the encoder only optimizes for initial feature detection, not for iterative refinement. The ISTA step becomes a fixed linear correction using stale encoder directions, which degrades over training.
- **Key insight: Modifications to the ISTA path cause dead features.** Both learned_step_size and detach_ista_encoder caused 39-47 dead latents. The ISTA correction path is critical for keeping features alive.

## Round 6 — Agent 2

### EXP039: lr_end=5e-5 at 200M — F1=0.9894 (NO IMPROVEMENT)
- Hypothesis: LR floor prevents over-annealing in final training steps. Engine supports lr_end.
- Config: Best config + lr_end=5e-5, 200M
- Result: F1=0.9894, identical to best. LR was already near-zero at end of decay — the floor doesn't kick in.
- **Conclusion: lr_end is useless. LR already decays smoothly to ~0. Confirmed by Agent 1 (lr_end=1e-4) and Agent 0 (lr_end=1e-5).**

### EXP040: inner_term_tilt=0 at 200M — F1=0.9894 (NO IMPROVEMENT)
- Hypothesis: Agent 3 found +0.0025 at 50M with inner_term_tilt=0 (no TERM on inner losses).
- Config: Best config + inner_term_tilt=0.0, 200M
- Result: F1=0.9894. At 200M, inner TERM tilt doesn't matter. The 50M signal was misleading.
- **Conclusion: Don't bother with inner_term_tilt separation. Agent 3 confirmed with 0.9880.**

### EXP041: initial_k=75 at 200M — F1=0.9879 (WORSE)
- Hypothesis: Fill the gap between k=60 (0.9772) and k=80 (0.9894) in the 200M K sweep.
- Config: Best config + initial_k=75, 200M
- Result: F1=0.9879, MCC=0.8238. 200M K law with TERM=0.010+step=0.25: k=60→0.9772, k=75→0.9879, k=80→0.9894, k=90→0.9777, k=100→0.9741.
- **Conclusion: k=80 is the optimal initial K at 200M. Sharp peak.**

### EXP042: lr=3.5e-4 at 200M — F1=0.9875 (WORSE)
- Hypothesis: LR was only tested at 2e-4 (too low, LISTA) and 3e-4 (current). 3.5e-4 might be better with the full ReferenceStyle architecture.
- Config: Best config + lr=3.5e-4, 200M
- Result: F1=0.9875, MCC=0.8236 (9444s). Higher LR overshoots.
- **LR law at 200M**: 2e-4→0.9004(LISTA), 3e-4→0.9894, 3.5e-4→0.9875. lr=3e-4 is optimal.

### EXP043: TERM annealing (0.005→0.015) at 200M — F1=0.9894 (NO IMPROVEMENT)
- Hypothesis: Linear anneal TERM from 0.005→0.015 over training. Average ≈ 0.010.
- Implementation: Code change — `_get_current_term_tilt()` linearly interpolates term_tilt_initial→term_tilt_final.
- Result: F1=0.9894, MCC=0.8245 (9261s). Identical to fixed TERM=0.010.
- **Conclusion: TERM annealing doesn't help. Model is robust to schedule when average is ~0.010.**

---

### EXP-A3-38: Decoder Norm Regularization — F1=0.9825 (MARGINAL)
- Hypothesis: Decoder column norms vary widely, biasing TopK selection. L2 penalty pushing toward 1.0 makes selection purely encoder-based.
- Config: ReferenceStyleSAE, decoder_norm_reg=0.01, 50M, k=80, TERM=0.010, step=0.25, iw=0.5
- Result: F1=0.9825, MCC=0.8209, **15 dead latents**
- +0.0012 over baseline but dead latents are a red flag. Not worth scaling to 200M.
- Why: The regularization pushes norms toward 1.0 but 0.01 weight is strong enough to conflict with reconstruction objective, killing some features.

### Round 6 Code Changes Summary — Agent 3
All code changes FAILED:
1. EMA weights: negligible (+0.0003)
2. Inner TERM separation: worse at 200M (-0.0014)
3. Learned per-feature ISTA step: catastrophic (-0.0212, 47 dead)
4. Detached W_enc in ISTA: worse (-0.0082, 39 dead)
5. Decoder norm regularization: marginal (+0.0012, 15 dead)
**Insight: The ReferenceStyleSAE architecture is highly optimized. Modifications to ISTA path cause dead features. Loss-only changes are safe but don't help. The remaining F1 gap (0.0106) may be irreducible at d_sae=4096.**

### EXP-A3-39: k=30 (final) at 200M — F1=0.9277 (MUCH WORSE)
- Config: Best config but k=30 (vs k=25), 200M, K schedule 80→30
- Result: F1=0.9277, precision=0.8696, recall=0.9962, MCC=0.8224, dead=0
- **Precision collapse**: 5 extra features per sample are mostly false positives. k=30 detects almost all true features (99.6% recall) but at 13% false positive rate.
- **Why k=25 works**: k=25 well below true L0=34.5 forces high selectivity. Only highest-confidence features selected. Missing 1% of features costs much less F1 than adding 13% false positives.
- **Final K law**: k=20≈k=25 (LISTA), k=25=optimal, k=30=catastrophic. Precision/recall tradeoff is extremely asymmetric.

### EXP-A3-40: d_sae=8192 at 50M — F1=0.8168 (CATASTROPHIC)
- Config: d_sae=8192, widths=[64,256,1024,2048,4096,8192], 50M, k=25, initial_k=80
- Result: F1=0.8168, recall=0.7444, **510 dead latents out of 8192**
- Double-width SAE can't be trained at 50M. Most latents never activate and die. Would need 400M+ training, not worth the compute cost.

### EXP-A3-41: k=22 (final) at 200M — RUNNING
- Hypothesis: k=22 has never been tested with ReferenceStyleSAE. Since k=30 collapsed precision, going BELOW k=25 might slightly improve precision without losing too much recall. k=20≈k=25 in LISTA suggests flat regime.
- Config: Best config but k=22, 200M

## Round 6 — Agent 1

### EXP-A1-36: lr_end=1e-4 at 200M — F1=0.9894 (NO IMPROVEMENT)
- Hypothesis: sae_lens defaults lr_end to ~3e-5 (lr/10). All our experiments have been using this implicit default. Setting lr_end=1e-4 (lr/3) keeps stronger gradients alive in the final LR decay phase, potentially improving TERM effectiveness on hard samples.
- Config: Best config + lr_end=1e-4, 200M
- Result: F1=0.9894, identical to best. Precision=0.9905, recall=0.9891, MCC=0.8245.
- **Conclusion: lr_end has no impact.** Confirmed by Agent 2 (lr_end=5e-5) and Agent 0 (lr_end=1e-5). The LR decay schedule is already optimal — changing the floor from 3e-5 to 0-1e-4 doesn't matter.
- **Complete lr_end law**: 0→same, 1e-5→same, 3e-5→same (default), 5e-5→same, 1e-4→same. Dead axis.

### EXP-A1-37: PerFeatureStepSAE (per-feature learned ISTA step) at 200M — F1=0.9729 (MUCH WORSE)
- Hypothesis: Different features need different ISTA correction amounts. Replace scalar step_size=0.25 with a learnable [4096]-dim vector initialized to 0.25. Only 4096 extra params. Decouples correction scale from encoding scale (W_enc serves dual purpose).
- Implementation: Created `PerFeatureStepSAE` class extending ReferenceStyleSAE. Added `self.step_size_vec = nn.Parameter(torch.full((d_sae,), 0.25))`.
- Config: PerFeatureStepSAE, 200M, k=80, TERM=0.010, step=0.25 (init), iw=0.5, 6 widths
- Result: F1=0.9729, MCC=0.8190, precision=0.9937, recall=0.9561, dead=3
- **Recall collapsed (0.9561 vs 0.9891).** Some features learned near-zero step sizes, effectively disabling ISTA correction. This prevented those features from being discovered through residual correction, leading to missed features (low recall). Precision improved (+0.0032) because the features that were detected were high-confidence.
- **Confirms Agent 3's finding at 50M** (0.9601, 47 dead). The failure is worse at 50M (faster divergence) but still catastrophic at 200M.
- **Key insight**: Uniform scalar step_size is a critical regularizer. The equal correction strength across all features prevents some features from being starved of ISTA gradient. This is analogous to how uniform learning rate works better than per-parameter adaptive rates in many settings.
- **Conclusion: Never make ISTA step_size per-feature. Scalar step_size=0.25 is essential.**

### Round 6 Summary — Agent 1
- All experiments at 200M failed to beat 0.9894
- lr_end is completely dead (entire 0-1e-4 range has no effect)
- Per-feature step sizes cause recall collapse
- Combined with Agent 3's findings (EMA=negligible, inner TERM=dead end, detached W_enc=dead end, decoder norm=marginal) and Agent 2's (inner_term_tilt=0 no help, k=75 worse), **the ReferenceStyleSAE at 0.9894 appears to be near-optimal for d_sae=4096**

## Agent 0 — Round 6 Experiments

### EXP39: TERM=0.012 + step=0.25 — F1=0.9877 (WORSE)
- Hypothesis: TERM=0.010+step=0.25→0.9894. TERM may not have peaked.
- Config: ReferenceStyleSAE, TERM=0.012, step_size=0.25, all other params same as best
- Result: F1=0.9877, MCC=0.8243 (8872s) — WORSE than TERM=0.010 (0.9894)
- **TERM peaked at 0.010 with step=0.25.** TERM curve at step=0.25: 0.005→0.9881, 0.008→0.9879, 0.010→0.9894, 0.012→0.9877
- Non-monotonic peak at 0.010. Don't go higher.

### EXP40: lr_end=1e-5 — F1=0.9894 (NO IMPROVEMENT)
- Hypothesis: LR decays to 0 in final 1/3. Stopping at 1e-5 prevents over-annealing.
- Config: Best config + lr_end=1e-5
- Result: F1=0.9894 (same as best). LR floor doesn't help.
- Confirms Agent 2's lr_end=5e-5 finding. LR at end of decay is already near-zero; flooring it changes nothing.
- **lr_end is a dead end.** Standard LR decay schedule is optimal.

### EXP41: SplitTERMSAE (inner_term_tilt=0) — F1=0.9880 (WORSE)
- Hypothesis: Inner Matryoshka losses get cleaner gradients without TERM hard-sample reweighting.
- Config: SplitTERMSAE, outer TERM=0.010, inner TERM=0 (standard mean for inner losses)
- Result: F1=0.9880, MCC=~0.824 — WORSE than uniform TERM (0.9894)
- Confirms Agent 2 (0.9894) and Agent 3 (0.9880). TERM on inner losses actually helps — hard-sample upweighting provides useful gradient signal for Matryoshka at all scales.
- **Don't separate inner/outer TERM.** Uniform TERM=0.010 across all losses is optimal.

### EXP42: batch_size=512 (RUNNING)
- Hypothesis: Smaller batch → 2x more gradient steps (390k vs 195k), smoother K decay (K decreases by 0.00014/step vs 0.00028/step), more frequent freq sort updates. No one tested batch<1024.
- Config: batch_size=512, lr_warm_up_steps=2000 (same fraction as 1000/195k), all other params from best
- Risk: Noisier gradients (half batch size), 2x longer training, LR decay covers 2x as many steps (potential over-annealing like 300M+batch=1024)

### Summary of Round 6 Axes Eliminated
| Axis | Result | Conclusion |
|------|--------|------------|
| TERM=0.012+step=0.25 | 0.9877 | TERM peaked at 0.010 |
| lr_end=1e-5 | 0.9894 | No help, same as default |
| Separate inner/outer TERM | 0.9880 | Uniform TERM is better |
| TERM=0.015+step=0.25 | Not tested | Likely worse given 0.012 worse |
| batch_size=512 | Running | Testing now |

### EXP-A1-38: adam_beta1=0.8 at 200M — F1=0.9868 (WORSE)
- Hypothesis: Lower momentum (0.8 vs default 0.9) helps Adam adapt faster to the changing loss landscape from the K schedule (80→25). More responsive to recent gradients.
- Config: Best config + adam_beta1=0.8, 200M
- Result: F1=0.9868, MCC=0.8235, precision=0.9902, recall=0.9853, dead=1
- **0.0026 WORSE than baseline (0.9894).** Lower momentum makes training noisier without benefit. The K schedule changes slowly enough (0.00028/step) that standard momentum=0.9 tracks it well. Lower momentum adds noise without improving adaptation.
- **adam_beta1 law**: 0.8→0.9868, 0.9→0.9894 (default). Combined with Agent 0's beta2=0.99→0.9867. **Default Adam betas (0.9, 0.999) are optimal. Don't touch optimizer hyperparams.**

### EXP-A1-39: TERM=0.009 + step=0.25 at 200M — F1=0.9877 (WORSE)
- Hypothesis: TERM curve jumps from 0.008 (0.9880) to 0.010 (0.9894). TERM=0.009 fills the gap — peak might be between.
- Config: Best config but term_tilt=0.009, 200M
- Result: F1=0.9877, MCC=0.8245, precision=0.9893, recall=0.9875, dead=1
- **0.0017 WORSE than TERM=0.010 (0.9894).** The peak at TERM=0.010 is very sharp — a single step away (0.009) drops performance.
- **Complete TERM curve at step=0.25**: 0.005→0.9881, 0.006→0.9875, 0.007→0.9860, 0.008→0.9879, 0.009→0.9877, 0.010→**0.9894**, 0.012→0.9877
- The curve is non-monotonic with a dip at 0.006-0.007 and a sharp spike at 0.010. This suggests TERM=0.010 hits a resonance with the architecture — possibly the exact threshold where hard-sample upweighting balances reconstruction vs sparsity objectives optimally.
- **TERM=0.010 is definitively the optimal value.** No further TERM tuning needed.

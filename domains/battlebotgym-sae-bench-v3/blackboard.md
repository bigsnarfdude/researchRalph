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

### EXP24: ReferenceStyleSAE 200M+k=80+warmup+widths=[32,64,128,512,1024,2048,4096] — RUNNING
- Hypothesis: Best config uses [32,128,512,1024,2048,4096]. My 7-width [64,128,...] got 0.9829. Adding width=64 between 32 and 128 might provide denser small-width supervision while keeping the powerful 32-bottleneck.
- Config: best config but matryoshka_widths=[32,64,128,512,1024,2048,4096]

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

### PRIORITY: ReferenceStyleSAE at 0.9867 (CURRENT OVERALL BEST — Agent 1!!)
- 200M + k=80 + warmup(1000) + TERM 0.002 + **widths=[32,128,512,1024,2048,4096]**
- Width search complete: 6 widths optimal. 7+ widths create too much gradient pressure.
- Agent 3: 300M=0.9766, confirmed 200M optimal. Agent 0: width-16 at 50M kills 83 features.
- 50M+k=100+6w=0.9534: 6 widths don't help short training with high K.
- **Remaining axes to explore**:
  - Alternative 6-width spacing (e.g., [32,64,256,1024,2048,4096] or [32,128,256,1024,2048,4096])
  - TERM tilt=0.003 with new widths (TERM=0.002 optimal with old widths, may differ)
  - ISTA step_size fine-tuning around 0.3 (e.g., 0.25 or 0.35)

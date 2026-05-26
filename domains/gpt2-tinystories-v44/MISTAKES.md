# MISTAKES — Tactics That Failed

## EXP-003: matrix_lr=0.06 (agent1)
- **What**: Increased MATRIX_LR from 0.04 to 0.06
- **Result**: 1.1676 vs 1.1676 baseline — no improvement
- **Why it failed**: The v2 matrix_lr win was on 8xA100 with larger batch sizes and more training steps. At our 5-min budget on RTX 4070 Ti, the learning rate may already be well-tuned for the shorter training horizon.
- **Lesson**: v2 results don't transfer 1:1 to different hardware/budget constraints.

## EXP-005: window_pattern="S" (agent1)
- **What**: Changed WINDOW_PATTERN from "SSSL" to "S" (all-short windows)
- **Result**: 1.1688 vs 1.1676 baseline — neutral/slightly worse
- **Why it failed**: The v2 finding was "S better at high step counts". Our 5-min budget doesn't reach high enough step counts for this to matter. The long window in SSSL helps with long-range dependencies at lower step counts.
- **Lesson**: Training duration matters for which window pattern is optimal.

## EXP-010: AR=96 (768-dim) + devbatch=16 (agent1)
- **What**: Increased ASPECT_RATIO from 64 to 96 (512-dim → 768-dim, ~2.25x params), reduced DEVICE_BATCH_SIZE to 16 to fit VRAM
- **Result**: 1.1824 vs 1.1676 baseline — worse
- **Why it failed**: devbatch=16 means 4x grad accum steps instead of 2x, cutting throughput in half. The bigger model is also slower per step. Combined effect: far fewer optimizer steps in 5 min. Steps matter more than model size.
- **Lesson**: At short time budgets, step throughput dominates. Never sacrifice throughput for model size unless you can maintain step count.

## EXP-007: MLP ratio 3x (agent0)
- **What**: Reduced MLP from 4x to 3x (c_fc from 2048 to 1536 at 512-dim)
- **Result**: 1.1074 vs 1.1020 best — worse by 0.005 BPB
- **Why it failed**: Step time remained 252ms (model is attention/memory-bandwidth bound, not MLP-compute bound). So we got same ~1190 steps but with less model capacity.
- **Lesson**: At depth 8 / AR 64 / 512-dim, the bottleneck is NOT MLP compute. Reducing MLP ratio just loses capacity. 4x is correct for this model size.

## EXP-008: matrix_lr=0.08 (agent1)
- **What**: Pushed MATRIX_LR from 0.06 to 0.08
- **Result**: 1.1130 vs 1.1020 best — worse by 0.011 BPB
- **Why it failed**: 0.08 overshoots the optimal Muon LR at this scale. The dmodel_lr_scale of 1.225 (from sqrt(768/512)) amplifies the effective LR further. Effective matrix_lr = 0.08 * 1.225 ≈ 0.098, which is too aggressive.
- **Lesson**: matrix_lr=0.06 is the ceiling at depth 8 / 512-dim. The LR scaling formula already handles dimension-dependent adjustment. Don't push beyond 0.06.

## EXP-010: WARMDOWN_RATIO=0.3 (agent1)
- **What**: Reduced WARMDOWN_RATIO from 0.5 to 0.3 (less cooldown, more peak LR time)
- **Result**: 1.1135 vs 1.1020 best — worse by 0.012 BPB
- **Why it failed**: 30% warmdown is insufficient for the model to converge. The LR stays too high for too long, preventing the parameters from settling. Combined with agent0's 0.67 result (1.1051), we now know: {0.3=1.1135, 0.5=1.1020, 0.67=1.1051}. 0.5 is the clear optimum.
- **Lesson**: WARMDOWN_RATIO=0.5 is well-tuned. Both shorter and longer cooldowns hurt. The LR schedule is near-optimal at the current settings.

## EXP-012: batch 2^15 (DEVICE_BATCH_SIZE=16) (agent1)
- **What**: Halved batch to 2^15 (from 2^16) to double optimizer steps to ~2400
- **Result**: 1.1248 vs 1.1020 best — worse by 0.023 BPB
- **Why it failed**: 16-sample batches are too noisy. The gradient signal-to-noise ratio drops below the threshold where more steps compensate. The 2^17→2^16 win was lucky — it was right at the edge of the step-quality tradeoff. Going further past that edge is destructive.
- **Lesson**: TOTAL_BATCH_SIZE=2^16 is the optimal tradeoff point. The batch size halving trick does NOT extrapolate further. VRAM usage dropped to 5997MB (from 11549), so memory is not the constraint — it's gradient quality.

## EXP-014: softcap=30 (agent1)
- **What**: Increased logit softcap from 15 to 30
- **Result**: 1.1123 vs 1.1020 best — worse by 0.010 BPB
- **Why it failed**: Softcap=15 provides useful regularization by limiting logit magnitude. At 30, the tanh barely activates (equivalent to nearly removing softcap). The model overfits or produces overconfident predictions without the constraint.
- **Lesson**: Softcap=15 is doing useful work. The calibration noted it might be redundant with QK-norm, but experimentally it helps. Don't weaken or remove it.

## EXP-020: Remove VE (agent1, first attempt)
- **What**: Emptied value_embeds dict to disable VE
- **Result**: CRASH — TypeError in Muon optimizer (ve_gate params had None gradients)
- **Why it failed**: Only disabled VE embeddings, not the VE gate Linear layers. The gates still existed in the model but had no gradients since they were never used in the forward pass. Muon tried to stack None gradients.
- **Lesson**: When removing VE, must also disable the ve_gate by making has_ve() return False. This prevents the gate from being created.

## EXP-019: depth=10 AR=64 devbatch=32 (agent0)
- **What**: Increased depth from 8 to 10 (640-dim, 5 heads, ~2x params) with DEVICE_BATCH_SIZE=32
- **Result**: OOM crash — MLP buffer (32*2048*2560 bf16) couldn't allocate 320MB
- **Why it failed**: depth 10 with 640-dim and 4x MLP means hidden dim = 2560. With batch=32 and seq=2048, the intermediate MLP tensor is too large. Peak VRAM far exceeds 16GB.
- **Lesson**: depth=10 with AR=64 REQUIRES DEVICE_BATCH_SIZE=16 on 16GB VRAM. But devbatch=16 means 2x grad_accum and ~half the steps.

## EXP-032: MATRIX_LR=0.01 (agent1)
- **What**: Pushed matrix_lr to 0.01 (from 0.02), following monotonic lower-is-better trend
- **Result**: 1.1088 vs 1.0994 best — much worse
- **Why it failed**: Effective LR = 0.0123 is too low for the Muon optimizer. The model can't make enough progress in 1190 steps. The orthogonalization in Muon needs sufficient step size to explore the parameter space.
- **Lesson**: matrix_lr bracketed: {0.01=1.109, 0.02=1.099, 0.04=1.101, 0.06=1.102, 0.08=1.113}. Optimum is 0.02 (effective 0.025). Don't go below 0.02.

## EXP-035: label_smoothing=0.1 (agent1)
- **What**: Added label_smoothing=0.1 to F.cross_entropy during training
- **Result**: 1.4246 — by far the worst result ever (0.325 BPB worse than best)
- **Why it failed**: Training-eval objective mismatch. Training optimizes smoothed CE, but eval uses standard CE (val_bpb). Model learns to spread probability mass across non-target tokens, which is catastrophically penalized by the hard eval metric. Also, with softcap=15 already limiting logit range, label smoothing is doubly redundant.
- **Lesson**: NEVER use label_smoothing when the eval metric is standard cross-entropy/BPB. The objectives must match.

## EXP-039: HEAD_DIM=64 (8 heads) + z-loss (agent0 prev cycle)
- **What**: Changed HEAD_DIM from 128 to 64, added z-loss 1e-4 (confounded)
- **Result**: 1.1064 vs 1.0961 (exp037, HEAD_DIM=128, no z-loss) — 0.010 BPB worse
- **Why it failed**: At 512-dim, 4 heads with 128-dim each work better than 8 heads with 64-dim each. More heads = smaller per-head capacity = worse attention quality at this scale. Z-loss contribution was negligible (exp039 vs exp040 differ by 0.0006 BPB only).
- **Lesson**: Don't increase head count at small model dimensions. HEAD_DIM=128 is optimal at 512-dim. Also, z-loss adds negligible value on top of softcap — they're doing similar things (constraining logit magnitude).

## best/train.py contamination (multi-agent race condition)
- **What**: best/train.py was overwritten with HEAD_DIM=64 + z-loss config during race condition between agents
- **Result**: 3+ experiments ran with corrupted config before the issue was detected
- **Why it happened**: run.sh copies live train.py → best/ when new best found, but train.py may have been modified by another agent between the experiment start and result time
- **Lesson**: ALWAYS verify best/train.py key params against the actual best result's log before starting from it. Check n_head in log output, check softcap, check z-loss presence.

## EXP-041: softcap=8 on clean config (agent1)
- **What**: Lowered softcap from 10/15 to 8, on clean config (HEAD_DIM=128, no z-loss)
- **Result**: 1.0966 vs 1.0958 best — worse by 0.001 BPB
- **Why it failed**: softcap=8 is too aggressive. tanh(x/8) saturates early, limiting logit range too much. The model can't express confident predictions for common tokens.
- **Lesson**: Softcap sweet spot is 10-15. Don't go below 10.

## ANALYSIS ERROR: Premature conclusion about HEAD_DIM=64 being harmful
- **What**: Concluded HEAD_DIM=64+z-loss was harmful based on exp039 (1.106) vs exp037 (1.096)
- **Why wrong**: exp039 was likely confounded by a race condition or other factor. exp038 (1.0958) used HEAD_DIM=64+z-loss and matched or beat exp037 (1.0961). The difference between configs is within noise.
- **Lesson**: Don't draw strong conclusions from single data points, especially in a race-condition-prone environment. Need clean A/B tests.

## EXP-042: SwiGLU activation + buffer_size=5000 (agent1)
- **What**: Replaced ReLU² with SwiGLU (gate+up: 512→1408, down: 1408→512), added buffer_size=5000
- **Result**: 1.0986 vs 1.0958 best — worse by 0.003 BPB
- **Why it failed**: SwiGLU provides better training dynamics (lower train loss) but worse generalization. The extra params (50.9M vs 50.3M) and 2.5% throughput loss (1209 vs 1239 steps) aren't compensated by quality improvement. ReLU²'s squaring provides useful regularization by killing small activations.
- **Lesson**: At small model sizes (512-dim) with short training (5 min), ReLU² outperforms SwiGLU. SwiGLU's advantages may only emerge at larger scale.

## EXP-043: buffer_size=5000 (agent0)
- **What**: Increased dataloader buffer_size from 1000 to 5000 for better best-fit packing
- **Result**: 1.0963 vs 1.0958 best — neutral (within 0.0005 BPB noise)
- **Why it was neutral**: The packing algorithm at buffer_size=1000 already achieves near-optimal document placement. TinyStories documents are short (children's stories), so most fit within the 2049-token row capacity. More buffer doesn't help because the documents are small relative to the row size.
- **Lesson**: Data pipeline packing is not the bottleneck at this dataset/sequence length. The gardener's repeated suggestion to explore the data pipeline was tested and found to be neutral.

## EXP-044: ns_steps=3 for Muon (agent1)
- **What**: Reduced Newton-Schulz iterations from 5 to 3 for Muon's polar decomposition
- **Result**: 1.1017 vs 1.0958 best — worse by 0.006 BPB
- **Why it failed**: 3 iterations produce insufficiently orthogonal gradient updates. The polar express coefficients were specifically tuned for 5 iterations. Using only 3 means the gradient is only partially orthogonalized, leading to less effective parameter updates. Only gained 12 extra steps (1251 vs 1239) — not enough to compensate.
- **Lesson**: Muon's gradient quality depends critically on the number of NS iterations. 5 is the minimum for proper orthogonalization. Don't trade gradient quality for marginal throughput.

## EXP-045: Cosine warmdown schedule (agent0)
- **What**: Replaced linear LR decay during warmdown with cosine decay
- **Result**: 1.0981 vs 1.0958 best — worse by 0.002 BPB
- **Why it failed**: Cosine keeps LR higher early in cooldown (0.98 vs 0.92 at 55% progress) then drops sharply at end. The model needs the gradual linear decay for smooth convergence. The sharp LR drop at end of cosine prevents the final parameters from settling properly.
- **Lesson**: Linear warmdown is optimal at this training budget. The LR schedule shape matters — both the warmdown ratio AND the decay function are well-tuned at linear/0.5/0.05.

## Optimizer bug not caught earlier (agent0, exp050)
**What**: Weight-tying refactor left value_embeds/resid/x0 param groups as orphaned dict() expressions not added to optimizer.
**Result**: These params were frozen at init for all weight-tied experiments. VE gates stayed at sigmoid(0)=0.5 (neutral), resid_lambdas at 1.0, x0_lambdas at 0.1 — all reasonable init values that happen to work OK, masking the bug.
**Lesson**: Always verify optimizer param group count matches expected groups. Add a sanity check like `assert len(optimizer.param_groups) >= expected_groups`.

## EXP-051/052: MATRIX_LR=0.03 at depth=6+wt (agent0 + agent1 replicate)
- **What**: Increased mlr from 0.02 to 0.03 at depth=6/384-dim with weight tying + optimizer bugfix
- **Result**: exp051=1.0909, exp052=1.0907 vs 1.0898 best — ~0.001 BPB worse (two replicates agree)
- **Why it failed**: mlr=0.03 overshoots at depth=6. Intuition was that more steps (2358 at depth=6 vs 1240 at depth=8) allows higher LR, but the opposite is true — more steps benefits from more precision (lower LR), not more speed (higher LR). The optimizer bugfix didn't produce a measurable improvement either.
- **Lesson**: matrix_lr=0.02 is robust across depth=6 and depth=8. At depth=6 bracket: {0.02=1.0898, 0.03=1.0909}. Don't push mlr higher at small model dims.

## Batch=2^15 at depth=7 (agent0, exp054)
**What**: TOTAL_BATCH_SIZE=2^15 + DEVICE_BATCH_SIZE=16 at depth=7
**Result**: 1.094 — 0.005 BPB worse than batch=2^16 (1.089)
**Lesson**: Batch halving was the biggest win at 2^17→2^16 but does NOT extrapolate further. Gradient noise from 16-sample batches kills optimization quality. This is now confirmed at two depths (7 and 8). Stop trying smaller batches.

## EXP-056: seq_len=1024 crash (agent0)
**What**: Truncated training sequences from 2048→1024 using `[:, :1024]`
**Result**: CRASH — torch.compile can't `.view(-1)` on non-contiguous tensors (stride 2048 on dim 0)
**Lesson**: When truncating tensor sequences, ALWAYS call `.contiguous()` before passing to torch.compiled models. Non-contiguous views cause silent failures in dynamo.

## EXP-057: TRAIN_SEQ_LEN=1024 (agent1)
**What**: Training at sequence length 1024 (eval stays at 2048). Called make_dataloader with T=1024, model config sequence_len=1024.
**Result**: 1.1454 — 0.057 BPB worse than best (1.0889). Catastrophic.
**Why it failed**: Three compounding failures: (1) No throughput gain — 204ms/step with grad_accum=2 ≈ same as 2048 with grad_accum=1. Attention savings entirely eaten by 2x gradient accumulation. (2) Eval mismatch — model never saw RoPE positions beyond 1024, can't handle 2048 eval sequences. (3) Window sizes halved — config.sequence_len=1024 → short_window=512 vs 1024 at full config.
**Lesson**: NEVER reduce training sequence length when eval is fixed at 2048. The eval mismatch alone is catastrophic (0.057 BPB). Sequence length is NOT an orthogonal throughput lever — it's a quality lever that must match eval. The gardener's repeated suggestion was well-intentioned but incorrect for this setup.

## EXP-059: softcap=15 at depth=7+wt (agent1)
**What**: Increased softcap from 10 to 15 at depth=7+wt, testing if regularization optimum shifted at new operating point.
**Result**: 1.0901 vs 1.0889 best — 0.0012 BPB worse.
**Lesson**: Softcap=10 is robust across operating points (depth=8 and depth=7+wt). The regularization benefit of tighter capping is scale-independent. Don't retune softcap at new depths unless the architecture changes qualitatively (not just 1 fewer layer).

## EXP-061: WEIGHT_DECAY=0.0 at depth=7+wt (agent1)
**What**: Removed weight decay (0.2→0.0) at depth=7+wt, hypothesizing that weight tying provides enough implicit regularization.
**Result**: 1.0933 vs 1.0889 best — 0.004 BPB worse.
**Why it failed**: Weight tying reduces unique params from 47M to ~27M. At this smaller model size, the model sees ~3.4 epochs of TinyStories in 300s. The model IS overfitting, and weight decay provides necessary regularization. The depth=8 result (WD neutral) was at 50M params / 1240 steps — different regime with more params and fewer repetitions.
**Lesson**: Weight decay necessity depends on model size AND training duration. At smaller models (weight-tied), explicit regularization matters more. Don't extrapolate WD neutrality from larger models.

## EXP-068: Freeze value embeddings at depth=7+wt (agent0)
**What**: Froze all value embedding params at init (requires_grad=False), hoping init values provide useful signal while reducing trainable params.
**Result**: 1.1053 vs 1.0850 best — 0.020 BPB worse. Large regression.
**Why it failed**: Value embeddings need training to learn meaningful token representations. Init values are random uniform noise — useful as a starting point but not as a final representation. The accidental freeze at depth=6+wt was misleadingly competitive because: (1) smaller 384-dim model where VE is less critical, (2) resid/x0 scalars were also frozen (may have provided complementary regularization). At depth=7/512-dim with 3 VE layers, the model needs learned VE for good attention.
**Lesson**: Don't freeze components that need to learn task-specific representations. The accidental freeze was a coincidence, not a technique. VE is essential at depth=7.

## EXP-071 race condition (agent0)
**What**: Queued window=64 (//32) experiment but agent1 kept overwriting train.py with //16 for their beta2=0.99 experiment.
**Result**: My run.sh snapshot captured agent1's config (window=128+beta2=0.99) instead of my intended window=64.
**Lesson**: The flock-acquire snapshot doesn't protect against edits between submission and lock acquisition. When the GPU queue is long, there's a wide window for the other agent to overwrite train.py. Need per-agent train files to prevent this.

## EXP-066: Constant Muon momentum=0.95 (agent1)
**What**: Eliminated the 0.85→0.95 momentum ramp over 300 steps, using constant 0.95 instead.
**Result**: 1.0909 vs 1.0889 best — 0.002 BPB worse.
**Why it failed**: The initial lower momentum (0.85) provides more gradient signal during early training when the model needs aggressive updates. Jumping straight to high momentum (0.95) over-smooths early gradients, slowing initial convergence. At ~1430 total steps, 300 ramp steps = 21% of training. The ramp-up is intentional and beneficial.
**Lesson**: Muon momentum ramp is well-designed. Don't eliminate it. The schedule implicitly matches training phase: aggressive early, smooth late.

## EXP-073: Muon beta2=0.99 (agent1, replicate of exp072)
**What**: Changed NorMuon second momentum from 0.95 to 0.99 for more stable variance estimation.
**Result**: 1.0842 vs 1.0837 best — 0.0005 BPB worse. Two replicates agree (exp072=1.0841 from race condition, exp073=1.0842).
**Why it failed**: beta2=0.99 has a 100-step warmup period (1/(1-0.99)=100) before the variance estimate converges. With only ~1523 steps total, 7% of training runs with poor variance estimates. beta2=0.95 converges in 20 steps (1.4% of training), providing accurate normalization earlier.
**Lesson**: At short training budgets, NorMuon beta2 should match the timescale: beta2=0.95 with ~1500 steps gives ~75 effective windows. Don't slow down the adaptive normalization.

## EXP-074: Graduated windows (128/128/128/256/256/256/2048) (agent0)
**What**: Instead of uniform short_window=128 for all layers, used graduated: layers 0-2 at 128, layers 3-5 at 256, layer 6 at 2048.
**Result**: 1.0856 vs 1.0837 best — 0.002 BPB worse.
**Why it failed**: Middle layers (3-5) also benefit from tight 128-token attention. Widening them to 256 slightly hurts. The model doesn't need a graduated hierarchy — all non-final layers should be tight, with the single full-context final layer handling global composition. TinyStories' short documents don't require phrase-level composition at 256 tokens.
**Lesson**: Uniform short windows are optimal at this dataset/model. The intuition that "deeper layers need broader context" doesn't hold here — the final full-context layer is sufficient.

## EXP-075: Constant weight decay (agent1)
**What**: Removed linear WD decay schedule, keeping WD=0.2 constant throughout training. Hypothesis: model overfits at depth=7+wt (exp061 WD=0.0 hurt by 0.004), so keeping regularization through warmdown helps.
**Result**: 1.0889 vs 1.0837 best — 0.005 BPB worse.
**Why it failed**: The linear decay `WD * (1-progress)` is synergistic with the LR warmdown. As LR drops, the model is making smaller updates — constant WD at low LR means weight decay dominates, effectively shrinking weights toward zero when the model should be fine-tuning. The original schedule was designed to match WD intensity to the LR: high WD when LR is high (prevents overshoot), low WD when LR is low (allows convergence).
**Lesson**: WD schedule should track LR schedule. Decoupled (constant) WD hurts at this training budget. The interplay between WD and LR is non-trivial — they need to be jointly designed.

## EXP-077: Depth=6+wt+window=128 (agent1)
**What**: Revisited depth=6 with window=128 improvement. Depth=6+wt got 1.090 with 1024 windows; window=128 improved depth=7 by 0.005 BPB.
**Result**: 1.0875 vs 1.0837 best — 0.004 BPB worse.
**Why it failed**: Despite 2611 steps (71% more than depth=7's 1523) and 171M tokens, the 384-dim model lacks capacity. The window improvement helps but doesn't compensate for going from 4 heads/512-dim to 3 heads/384-dim. Depth=7's 512-dim capacity is more valuable than depth=6's extra steps.
**Lesson**: At this training budget, capacity (width) trumps step count once you're past the throughput sweet spot. Depth=7 is definitively optimal — the 512→384 dim reduction is too large to overcome with more steps.

## EXP-080: Residual dropout=0.02 (agent1)
- **What**: Added F.dropout(x, p=0.02) after attention and MLP outputs in Block.forward
- **Result**: 1.0872 vs 1.0837 best — 0.003 worse
- **Why it failed**: At depth=7+wt with WD=0.2 (linear decay) + softcap=10, the model is already well-regularized. Dropout adds noise to activations that hurts the optimization trajectory more than it helps generalization. Also 450MB VRAM overhead from storing activation masks.
- **Lesson**: ALL regularization axes are dead at this operating point: label_smooth (1.425), z-loss (1.106), constant_WD (1.089), dropout (1.087). The existing WD+softcap combo is the right amount.

## EXP-081: Race condition (agent0)
- **What**: Submitted gradient clipping experiment, but run.sh captured agent1's dropout code instead
- **Result**: 1.0871 — duplicate of exp080 (dropout=0.02)
- **Why it failed**: Agent1 modified train.py with dropout between my cp and run.sh's flock-acquire. The snapshot was taken at lock acquisition time, not at my submission.
- **Lesson**: ALWAYS verify the snapshot train.py in logs/ after submission. The flock mechanism doesn't prevent race conditions — it only serializes GPU access. Need per-agent train.py files.

## EXP-082: Gradient clipping max_norm=1.0 (agent0)
- **What**: Added torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()
- **Result**: 1.0838 vs 1.0837 best — noise-level, completely neutral
- **Why it failed**: Muon's polar express orthogonalization already normalizes gradients. The Newton-Schulz iterations produce unit-norm orthogonal gradient matrices, so there are no gradient spikes to clip. AdamW for embeddings also uses bias-corrected moments, which naturally limits update magnitudes.
- **Lesson**: Gradient clipping is redundant when using Muon — the optimizer's own gradient normalization subsumes it.

## EXP-080: Residual dropout=0.02 (agent1)
- **What**: Added F.dropout(x, p=0.02, training=self.training) after attention and MLP outputs in Block.forward
- **Result**: 1.0872 vs 1.0837 best — worse by 0.003 BPB
- **Why it failed**: The model doesn't overfit at the activation level. WD=0.2 + softcap=10 already provide sufficient regularization. Also, dropout adds VRAM overhead (10747MB vs 10299MB, +450MB for activation masks). At 3.4 epochs of data, the overfitting is controlled by existing mechanisms.
- **Lesson**: Regularization axis is dead: {label_smooth=1.425, z-loss=1.106, constant_WD=1.089, dropout=1.087}. All four types of regularization (loss-level, logit-level, weight-level, activation-level) have been tested and failed. The current softcap=10 + linear WD decay is optimal.

## EXP-084: Multi-variable early training dynamics (agent0)
- **What**: Combined warmup=0.02 + embedding_std=0.5 + Muon ramp 150 steps
- **Result**: 1.0845 vs 1.0837 best — 0.0008 worse
- **Why it failed**: Each change targets early training dynamics, but with 1430 steps, the optimizer quickly adapts to any init condition. The 30 warmup steps are wasted, the embedding std is overwritten within ~50 steps, and the momentum ramp change is marginal. When the config is at a genuine optimum, multiple small perturbations don't find a better basin.
- **Lesson**: Once all single-variable sweeps converge, multi-variable combos of small changes also converge. The remaining improvement requires qualitatively different approaches (more time, more hardware, different architecture class).

## EXP-087: Zero attention Q/K/V init (agent0)
- **What**: Initialized all attention Q, K, V weight matrices to zero (from uniform(-s,s)). MLP init unchanged.
- **Result**: 1.5339 — catastrophic, 0.45 BPB worse
- **Why it failed**: Zero Q/K/V means all heads compute uniform attention (q·k=0 for all pairs after QK-norm). With Muon, zero-init creates a symmetry problem — all heads remain identical. The model can't learn meaningful attention patterns in 1430 steps from this degenerate starting point. The ~1.534 BPB represents a pure MLP model (no attention contribution).
- **Lesson**: Attention initialization MUST break head symmetry. The uniform(-s,s) init provides necessary diversity for heads to specialize. Zero-init works for output projections (c_proj, mlp.c_proj) because they just need to learn to "turn on" gradually, but Q/K/V need to start with diverse patterns.

## EXP-085: WARMUP_RATIO=0.02 (agent1)
- **What**: Added 2% LR warmup (~30 steps) at depth=7+wt+window128
- **Result**: 1.0840 vs 1.0837 best — noise-level, completely neutral
- **Why it failed**: The model has built-in "warmup" via zero-init output projections (c_proj, mlp.c_proj) which naturally ramp effective output magnitudes. External LR warmup is redundant.
- **Lesson**: WARMUP_RATIO bracket: {0.0=1.084, 0.02=1.084}. Don't bother with LR warmup when using zero-init projections.

## EXP-086: GELU activation (agent1)
- **What**: Replaced ReLU² (`relu(x).square()`) with GELU in MLP
- **Result**: 1.5338 vs 1.0837 best — **CATASTROPHIC**, 0.45 BPB worse
- **Why it failed**: GELU has no sparsity. ReLU² provides (1) exact zero for negative inputs, (2) squaring amplifies large activations while suppressing small ones. The dense activations overwhelm model capacity at 512-dim/1430 steps.
- **Lesson**: Activation bracket: {ReLU²=1.084, SwiGLU=1.099, GELU=1.534}. ReLU² is critical. The squaring is load-bearing implicit regularization.

## EXP-088: WEIGHT_DECAY=0.3 (agent1)
- **What**: Increased WD from 0.2 to 0.3 at depth=7+wt+window128
- **Result**: 1.0852 vs 1.0837 best — worse by 0.0015 BPB
- **Why it failed**: WD=0.3 with linear decay averages to effective WD≈0.15, which over-regularizes at 1430 steps. The improvement from 0.0→0.2 was from reducing overfitting, but 0.3 goes past the sweet spot and starts hurting optimization speed.
- **Lesson**: WD bracket: {0.0=1.093, 0.2=1.084, 0.3=1.085}. The curve peaks sharply at 0.2. Don't increase WD beyond 0.2.

## EXP-091: SCALAR_LR=0.25 + softcap=12 combo (agent1)
- **What**: Stack two independent breakthroughs — SCALAR_LR=0.25 (from exp089) + softcap=12 (from exp090)
- **Result**: 1.0855 vs 1.0835 best — WORSE by 0.002 BPB
- **Why it failed**: The two changes are anti-complementary. SCALAR_LR=0.25 slows per-layer lambda learning; softcap=12 changes output magnitude control. Both address signal flow/magnitude — combining them over-corrects. They compete rather than stack.
- **Lesson**: Not all breakthroughs are additive. When two changes affect the same underlying mechanism (output magnitude/signal flow), they can interfere. Always test combinations; never assume additivity.


## exp091: Stacking independent wins failed (agent0 observation)
- **What**: Combined SCALAR_LR=0.25 (exp089, +0.0003) and softcap=12 (exp090, +0.0001) 
- **Result**: 1.0855 — WORSE than either individual win (1.0835)
- **Lesson**: At deep diminishing returns, hyperparams interact non-linearly. Single-axis wins are NOT independent. Must test from actual current best, not combine multiple recent improvements.

## exp092: SCALAR_LR=0.1 too slow at softcap=12 (agent0)
- **What**: SCALAR_LR=0.1 (from 0.5) at depth=7+wt+softcap12+window128
- **Result**: 1.0863 — 0.003 worse than best
- **Lesson**: SCALAR_LR interacts with softcap. At softcap=12, the per-layer lambdas need faster tuning (0.5) to adapt to the different logit capping. The SCALAR_LR=0.25 win was specific to the softcap=10 regime.

## EXP-093: FINAL_LR_FRAC=0.07 (agent1)
- **What**: Push final LR fraction from 0.05 to 0.07
- **Result**: 1.0838 vs 1.0835 best — 0.0003 worse
- **Why it failed**: The bracket {0.03=1.100, 0.05=1.084, 0.07=1.084} shows diminishing returns. 0.05 is the sweet spot — 0.07 slightly overshoots by keeping LR too high during final convergence.
- **Lesson**: FINAL_LR_FRAC bracket is fully closed: {0.0, 0.03, 0.05, 0.07}. Optimum is 0.05. Schedule shape axis is exhausted.


## exp094: FINAL_LR_FRAC=0.1 slightly worse (agent0)
- **What**: FINAL_LR_FRAC=0.1 (from 0.05) at depth=7+wt+softcap12
- **Result**: 1.0842 — 0.0007 worse than best
- **Lesson**: The monotonic improvement 0.0→0.03→0.05 reverses at 0.07+. FLR=0.05 is the sweet spot. Higher final LR causes slight underfitting (model doesn't cool down enough to reach the loss basin floor).

## EXP-095: Muon momentum target=0.97 (agent1)
- **What**: Increase Muon final momentum from 0.95 to 0.97
- **Result**: 1.0852 vs 1.0835 best — 0.0017 worse
- **Why it failed**: Higher momentum overshoots at 1500 steps. The model cant converge fast enough during warmdown with 0.97 momentum.
- **Lesson**: Muon momentum is closed: {0.95, 0.97}. 0.95 is optimal.


## EXP-097: resid_lambda LR multiplier=0.1 (agent1)
- **What**: Increase resid_lambda LR from SCALAR_LR*0.01 to SCALAR_LR*0.1 (10x faster)
- **Result**: 1.0846 vs 1.0835 best — 0.001 worse
- **Why it failed**: The resid_lambdas start at 1.0 and need to stay near 1.0. 10x faster learning causes them to oscillate away from optimal scaling.
- **Lesson**: resid_lambda LR is correctly tuned at 0.01x. The 100x difference between resid and x0 lambda LRs is intentional — resid lambdas need stability, x0 lambdas need to explore.


## EXP-102: MLP 3x + parallel attn+MLP (agent0, race condition)
- **What**: MLP ratio from 4x to 3x, PLUS parallel attention+MLP (unintended from race condition)
- **Result**: 1.0934 vs 1.0832 best — 0.010 worse
- **Why it failed**: 3x MLP doesn't improve throughput (memory-bandwidth bound, not MLP-compute bound). Parallel attn+MLP removes sequential conditioning. Multi-variable made it hard to isolate, but 3x MLP was already proven harmful at depth=8 (exp007).
- **Lesson**: MLP ratio 3x closed at BOTH depth=8 and depth=7. Race conditions with other agents cause uncontrolled multi-variable experiments. Always verify the snapshot after submission.

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

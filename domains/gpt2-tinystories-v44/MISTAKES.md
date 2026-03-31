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

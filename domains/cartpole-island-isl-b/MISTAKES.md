# MISTAKES — cartpole-island-isl-b

## EXP-002: Aggressive position_centering (0.3→0.6)
**What**: Tried to solve 8 position failures by doubling position_centering.
**Result**: Score dropped to 0.4765, all 50 episodes failed by pole angle.
**Why it failed**: position_centering dominates the signal when too high, starving angle control of authority. The feature weights have strong interaction effects; can't tune them independently.
**Lesson**: Parameter sweeps must respect interaction effects. Single-knob tuning can cause catastrophic failure in coupled systems.

## Why EXP-003 (sharpness) was neutral
**Hypothesis**: response_sharpness controls oscillation, so reducing it would smooth decisions.
**Result**: No change (0.9732 both ways).
**Why**: For this config, angle stability is already achieved; sharpness only affects the sigmoid slope. The real constraint was position damping (too low velocity_weight), which sharpness doesn't touch.
**Lesson**: Always isolate the bottleneck before tuning. Sharpness only matters when the controller is on-the-edge of a decision boundary; here, the signal margins were large enough that sigmoid shape didn't matter.

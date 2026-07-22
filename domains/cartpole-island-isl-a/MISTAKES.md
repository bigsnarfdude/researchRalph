# Agent Mistakes & Dead Ends — cartpole-island-isl-a

## EXP-002: Raw Position Feedback Increase (agent0)
**What**: Increased position_weight and velocity_weight directly from 0.2 to higher values.
**Result**: Score crashed to 0.0191 (9× worse than baseline 0.3729).
**Why it failed**: Raw position_weight competes with angle control signal. When both signals push hard, they oscillate and destabilize the system.
**Lesson**: Use position_centering (active bias) or velocity_weight (damping) instead of raw position_weight for initial fixes. But see EXP-008.

## EXP-006: Aggressive Position Centering (agent1)
**What**: Set position_centering to 0.9 (from 0.6) to fix position drift in EXP-005.
**Result**: Score dropped from 0.8150 to 0.7072, position failures increased 27→40.
**Why it failed**: position_centering is an active bias that pulls the signal toward center-pushing. When too high, it dominates and prevents fine angle control needed for long episodes.
**Lesson**: position_centering has a sweet spot around 0.6–0.7. Beyond that, it breaks the balance.

## Early Hypothesis: Position Centering as Main Solution (agent1, EXP-003)
**What**: Removed angle_bias and cranked position_centering to 0.8, expecting to fix position drift.
**Result**: Fixed position drift (0 failures) but created angle-only failure mode (50/50 angle failures, score 0.3461).
**Why it failed**: position_centering alone cannot balance both angle and position without sufficient angle_weight.
**Lesson**: Controller tuning is not separable — angle and position feedback are coupled through the sigmoid. Must tune both together.

# Blackboard — cartpole-island (island a)

## EXP-001: seed-baseline (agent0)
SCORE: 0.3729 | Seed config baseline. 50 position-drift failures.
Issue: position_weight=0.2, velocity_weight=0.2 insufficient for cart centering.

## EXP-002: h1-stronger-position (agent0)
SCORE: 0.0191 | Tried raw increase of position_weight/velocity_weight.
Crashed: direct position feedback competes with angle control. DEAD END.

## EXP-003: remove-bias-try-centering (agent1)
SCORE: 0.3461 | Set angle_bias=0.0, position_centering=0.8.
Fixed position drift (0 failures) but created angle-only failures (50/50).
Learning: position_centering alone inadequate; angle response too weak.

## EXP-004: stronger-angle (agent1)
SCORE: 0.5668 | Increased angle_weight:2.0, angular_velocity_weight:0.8.
Got 2/50 perfect (first success). 48/50 angle failures. Angle control growing.

## EXP-005: angle-emphasis (agent1)
SCORE: 0.8150 | angle_weight:3.0, angular_velocity_weight:1.2, anticipation:0.3, position_centering:0.6.
23/50 perfect episodes. Angle failures eliminated, but 27 position failures.
Key insight: angle control now strong; position is the remaining bottleneck.

## EXP-006: position-stability (agent1)
SCORE: 0.7072 | Increased position_centering→0.9, velocity_weight→0.4.
Degraded: position-centering too aggressive (40/50 position failures).
Learning: position_centering conflicts with angle; raw position_weight better.

## EXP-007: tuned-balance (agent1)
SCORE: 0.7853 | position_centering:0.7, velocity_weight:0.3.
Better than 0.7072, worse than 0.8150. Sweet spot is ~0.6.

## EXP-008: position-weight (agent1)
SCORE: 0.9988 | Increased position_weight→0.5 (was 0.2), kept position_centering:0.6.
48/50 perfect! Only 2 position failures. KEY DISCOVERY: position_weight (passive feedback) > position_centering (active bias).

## EXP-009: velocity-damp (agent1)
SCORE: 1.0000 | Boosted velocity_weight→0.25 (was 0.2). 50/50 PERFECT!
Optimal config found after 9 experiments.

## EXP-002: h1-stronger-position
SCORE: 0.0191 | Hypothesis: double position/velocity feedback (0.5 each).
FALSIFIED — pole balance collapsed. All 50 angle terminations.
Lesson: position feedback must be balanced against pole control, not proportional.

## EXP-003: h2-balanced-damping
SCORE: 0.7285 | Hypothesis: boost angular_velocity_weight to 0.5 for pole damping.
Success — 2x improvement from baseline. Pole balance solid (0 angle terminations).
Remaining issue: position drift limits to 364/500 avg steps.
→ H3: Increase position feedback while preserving pole balance.

## EXP-004: h3-stronger-position-v2
SCORE: 1.0 | Hypothesis: position_weight=0.6, velocity_weight=0.6, angular_velocity=0.5.
✓ OPTIMAL — Perfect 50/50 episodes, 500 steps each.
Configuration: angle_weight=1.0, angular_velocity_weight=0.5, position_weight=0.6, velocity_weight=0.6, others unchanged.

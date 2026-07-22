# Blackboard — cartpole-island (island a)

## EXP-002 Intent
Increase angle_weight (1.0→1.5) + angular_velocity_weight (0.3→0.5). Hypothesis: baseline is under-responsive to pole angle; more aggressive stabilization should improve survival.

## EXP-002 Result
SCORE: 0.4292 (was 0.3729, +15% gain, avg 214.6/500 steps). Hypothesis confirmed — more aggressive angle control helps. BUT: all 50 terminations now from position (cart drift), not angle. Pole is stable enough; cart control needs improvement. Next: increase position_weight or velocity_weight to reduce boundary hits.

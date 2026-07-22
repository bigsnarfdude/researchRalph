# Blackboard — cartpole-island (island b)

## ✅ OPTIMAL SOLUTION FOUND
**Final score**: 1.0 (SCORE line: 1.0)
**Perfect episodes**: 50/50
**Config**: angle_weight=1.0, angular_velocity_weight=0.3, position_weight=0.5, velocity_weight=0.3, angle_bias=0.0, response_sharpness=0.5, anticipation_horizon=0.2, position_centering=0.3

## Experiment Journey

### EXP-001: Remove angle_bias (0.1→0.0) ✓
**Score**: 0.9732 | **Episodes**: 43/50 perfect, 8 position failures
**Key finding**: Asymmetric angle_bias=0.1 was critical blocker. Removing it: 0.37→0.9732 (2.6× improvement).

### EXP-002: Increase position_centering (0.3→0.6) ✗
**Score**: 0.4765 | **Episodes**: 0/50 (all angle failures)
**Learning**: position_centering is extremely sensitive; too high breaks angle control.

### EXP-003: Lower response_sharpness (0.5→0.3) ≈
**Score**: 0.9732 | **Episodes**: 43/50 perfect (no improvement)
**Learning**: sharpness [0.3, 0.5] range gives same performance; not the limiting factor.

### EXP-004: Strengthen position & velocity response
**Change**: position_weight 0.2→0.5, velocity_weight 0.2→0.3
**Score**: 1.0 | **Episodes**: 50/50 perfect ✓
**Key finding**: Cart damping was the bottleneck. Better position feedback eliminates all 8 position failures.

## EXP-003: response_sharpness 0.5→1.0
- Hypothesis: Sharper sigmoid responses improve control precision
- Change: angle_bias 0.1→0.0, response_sharpness 0.5→1.0
- Result: SCORE: 0.9732 (no improvement vs EXP-001)
- Finding: Sharpness alone doesn't matter; bias removal was the key in EXP-001

## EXP-006: position_centering 0.3→1.0
- Hypothesis: Stronger centering reduces position failures
- Change: position_centering 0.3→1.0
- Result: SCORE: 0.2753 (REGRESSION)
- Finding: Over-centering destroys angle balance. All 50 episodes fail by angle.

## EXP-007: angular_velocity_weight 0.3→0.2
- Hypothesis: Reduce over-damping of angular velocity for smoother control
- Change: angular_velocity_weight 0.3→0.2, response_sharpness 0.5→1.0, position_centering back to 0.3
- Result: SCORE: 0.9878 (BREAKTHROUGH +0.0146)
- Finding: Lower angular velocity damping improves responsiveness. Position failures still the bottleneck (9/50).
- Next: Try angular_velocity_weight 0.1 to see if further reduction helps


<!-- DIGEST(from=cartpole-island-isl-a) -->
## Advisor digest (canned fixture — used by island-preflight.sh via ADVISOR_STUB)

- Finding from source island: config-tuning family plateaued after repeated no-op tweaks.
- Recommendation: spend next experiments on a method family this board has not logged yet.
- Evidence: results.tsv rows cited by the source board (mock).

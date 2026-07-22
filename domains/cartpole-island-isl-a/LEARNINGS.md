# Agent Learnings — cartpole-island-isl-a

## The Controller Signal Coupling
**Discovery**: The controller combines angle + position + velocity feedback into a single weighted signal, passed through sigmoid. Changing one parameter affects the entire signal balance.
**Evidence**: EXP-002 (raw position ↑) crashed; EXP-003 (position_centering ↑ alone) reversed failure mode but didn't improve score; EXP-004–009 required joint tuning of angle_weight, angular_velocity_weight, position_weight, velocity_weight, position_centering.
**Implication**: Single-parameter sweeps won't find optimum. Multi-parameter coordination matters.

## Position Control: Two Mechanisms
**Discovery**: The controller has two position-related parameters with different effects:
- `position_weight`: Passive feedback on current cart position (x). Linear term in signal.
- `position_centering`: Active bias that increases signal toward pushing-back-to-center. Modifies signal as `signal -= x * position_centering`.

They are **not equivalent**. position_weight is integrated into the signal smoothly; position_centering can dominate if too high.

**Evidence**: EXP-005 (position_centering:0.6, position_weight:0.2) gave 0.8150 but 27/50 position failures. EXP-008 (same centering, position_weight:0.5) jumped to 0.9988 with only 2 failures. Increasing raw position_weight was 5× more effective than centering.

**Implication**: For drift-heavy environments, passive position feedback > active centering bias.

## The Baseline Was Not Hopeless
**Discovery**: seed config (0.3729) was actually a reasonable starting point — pure position-drift failure mode is interpretable and fixable.
**Evidence**: EXP-002's attempt to "fix" it crashed to 0.0191. But gradual, coupled tuning led to 1.0 in 9 experiments.
**Implication**: Linear feedback controllers have smooth loss landscapes when tuned correctly. The seed config was not a local minimum; it was just under-tuned on both angle and position.

## Response Sharpness & Anticipation Horizon
**Discovery**: Sigmoid sharpness and anticipation (future angle prediction) matter but are secondary to core weights.
**Evidence**: Default sharpness:0.5 was weak (EXP-001–003). Increasing to 1.0 (EXP-004) helped. Anticipation horizon:0.3 (EXP-005) was used in optimal. But both worked within a range; no single value was critical.
**Implication**: Tune primary weights first (angle, angular_velocity, position, velocity). Then refine sharpness and anticipation if score plateau.

## Angle Bias Was Dead Weight
**Discovery**: angle_bias:0.1 in seed config contributed nothing and was removed in EXP-003 onward. No downside to setting it to 0.0.
**Evidence**: Removing it allowed cleaner tuning of other parameters without a constant steering offset.
**Implication**: Bias parameters in linear controllers should be eliminated unless there's a documented asymmetry in the environment.

## Optimal Configuration
**Configuration**: 
```
angle_weight: 3.0
angular_velocity_weight: 1.2
position_weight: 0.5
velocity_weight: 0.25
angle_bias: 0.0
response_sharpness: 1.0
anticipation_horizon: 0.3
position_centering: 0.6
```
**Score**: 1.0 (50/50 perfect episodes)
**Trajectory**: 0.3729 → 0.5668 → 0.8150 → 0.9988 → 1.0 across 9 experiments.

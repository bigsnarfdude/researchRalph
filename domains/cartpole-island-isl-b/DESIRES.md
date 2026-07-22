# DESIRES — cartpole-island-isl-b

## What worked well
1. **Workspace isolation**: Each agent has independent workspace/agentN/params.yaml. This allowed parallel exploration without conflicts.
2. **Deterministic oracle**: Fixed seed (42) + 50 episodes gives repeatable scores. Easy to attribute improvements to config changes.

## What could be improved
1. **Early diagnosis of interaction effects**: The position_centering parameter has strong coupling with angle_weight. A sensitivity analysis or visualization of parameter interactions would help avoid catastrophic failures like EXP-002 (position_centering=0.6).

2. **Shared search log**: Currently blackboard.md is append-only and grows. A structured format (JSON or TSV with agent/timestamp) would let agents read prior experiments more efficiently and avoid duplicate searches.

3. **Parameter sweep templates**: For simple tuning like this, automated grid search over likely regions would be faster than manual hypothesis→test loops. A sweep tool that evaluates parameter combinations in parallel would accelerate optimization.

## Domain observations
- This domain is solved (score 1.0 achieved)
- Solution space has clear structure: angle control via bias/weights, position control via position_weight/velocity_weight balance
- Optimal config: angle_weight=1.0, angular_velocity_weight=0.3, position_weight=0.5, velocity_weight=0.3, angle_bias=0.0, response_sharpness=0.5, anticipation_horizon=0.2, position_centering=0.3

# cartpole-island — controller tuning (v5 island pilot)

## Goal
Maximize the oracle score: average pole-balancing survival over 50 episodes.
Current seed config scores ~0.37. Known optimal is 1.0 (this environment is
solvable — perfect 500-step episodes are achievable with the right weights).

## How to experiment
1. Edit YOUR workspace copy only: `workspace/<your agent id>/params.yaml`
2. Score it: `bash run.sh <short-name> "what you changed and why"`
3. The oracle prints SCORE and logs to results.tsv. Cite the exact SCORE line
   when you write findings to blackboard.md. Never edit results.tsv.

## Parameters (params.yaml)
angle_weight, angular_velocity_weight, position_weight, velocity_weight,
angle_bias, response_sharpness, anticipation_horizon, position_centering.
All floats. The controller is linear in these features; balance responsiveness
to pole angle against drift toward the track edges.

## Board discipline
Keep blackboard.md under 300 lines — curate, don't append forever. If
BOARD_OVER_BUDGET exists, condense the board before adding new findings.

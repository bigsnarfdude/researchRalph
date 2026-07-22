# LEARNINGS — cartpole-island-isl-b

## CartPole Controller Tuning (v5 island pilot)

### 1. Asymmetric bias breaks controller stability
Seed config had angle_bias=0.1, which caused systematic drift by pushing the controller to favor one direction. Removing this single parameter (angle_bias: 0.1 → 0.0) improved score 2.6× (0.37 → 0.9732). **Key insight**: Linear controllers must have symmetric baseline (bias=0.0).

### 2. Position control is the bottleneck (after angle bias fix)
At 0.9732, pole angle was perfectly stable (0 angle-limit failures), but cart drifted to position boundary in 8/50 episodes. Position control was under-specified.

### 3. Cart damping (velocity_weight) + position awareness (position_weight) together solve drift
Increasing position_weight 0.2→0.5 and velocity_weight 0.2→0.3 eliminated all 8 position failures, reaching 1.0. These must be tuned together: high position_weight without velocity damping causes oscillation; low velocity_weight allows drift.

### 4. position_centering is extremely sensitive
Increasing from 0.3→0.6 broke angle control completely (50/50 angle failures, score 0.47). Parameter sensitivity is not uniform—some knobs are fragile.

### 5. response_sharpness has little effect in [0.3, 0.5] range
Changing from 0.5→0.3 and back both gave 0.9732. This parameter matters less than the feature weights for this domain.

## Architecture Observations
- Linear controller (θ, θ̇, x, ẋ) with bias + anticipation + position_centering
- Sigmoid decision boundary is critical: sharpness controls slope, but feature weights control everything
- Position dynamics are much slower than angle dynamics, so both need tuned response

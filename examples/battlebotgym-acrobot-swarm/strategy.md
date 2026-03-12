# Search Strategy — acrobot

## Current Best
- Score: 0.7516
- Config: best/config.yaml (phase_offset=0.01, theta2_weight=0.0, rest=baseline)

## Phase: diversify (convergence detected, near theoretical max ~0.75)

## What works (high confidence)
- catch_threshold=2.0 (max) — catch controller dominates the trajectory
- sign_strategy > 0.5 (bang-bang) — proportional is strictly worse
- phase_offset ≈ 0.0 — extremely sensitive, ±0.3 hurts
- theta1_weight ≈ 4.0, dtheta1_weight ≈ 4.0 — robust sweet spot
- theta2_weight=0.0 — ignoring theta2 in catch helps slightly
- dtheta2_weight ≈ 0.9 — near-optimal
- energy_pump_gain and energy_target are irrelevant (catch dominates)

## What fails (avoid)
- phase_offset ±1.57 → total failure (0.0)
- catch_threshold < 1.8 → severe degradation
- All catch weights maxed (10.0) → 0.39
- All catch weights minimal (0.5) → 0.25
- sign_strategy ≤ 0.5 (proportional) → 0.68
- velocity_threshold is dead (unused in code)

## Untested
- Fine grid around theta1_weight 3.5-4.5 with theta2=0
- Interaction between catch_gain and theta1/dtheta1 weights
- Whether the bimodal step distribution can be resolved

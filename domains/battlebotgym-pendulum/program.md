# Pendulum Swing-Up & Balance — Agent Instructions

## Task
Optimize a two-phase pendulum controller by editing `config.yaml` to **maximize the normalized score**. The pendulum starts hanging DOWN and must be swung UP and kept balanced upright.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (0.0–~0.95) is printed to stdout
# Episode summary is printed to stderr
```

**Budget:** ~2 seconds per experiment (50 episodes)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the controller's gains and thresholds

### Parameters and Ranges

**Swing-Up Phase** (energy-based pumping when far from upright):
| Parameter | Range | Effect |
|-----------|-------|--------|
| swing_gain | 0.0–10.0 | How aggressively to pump energy into the pendulum |
| energy_target | 0.0–30.0 | Target energy level for swing-up (upright energy ~ 20) |
| max_swing_torque | 0.1–2.0 | Torque clamp during swing-up phase |

**Balance Phase** (PD control when near upright):
| Parameter | Range | Effect |
|-----------|-------|--------|
| balance_gain_p | 0.0–20.0 | Proportional gain on angle error |
| balance_gain_d | 0.0–10.0 | Derivative gain on angular velocity |
| max_balance_torque | 0.1–2.0 | Torque clamp during balance phase |

**Phase Switching:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| switch_angle | 0.1–3.14 | Angle from upright (rad) below which balance mode activates |

**Fine-Tuning:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| damping | 0.0–5.0 | Velocity damping added to control signal |
| anticipation | 0.0–5.0 | How much to weight predicted future angle |
| deadzone | 0.0–0.5 | Angle error below which no torque is applied (prevents oscillation) |

## Physics
- Pendulum: mass=1.0 kg, length=1.0 m, gravity=10.0 m/s^2
- theta=0 is hanging DOWN, theta=pi is balanced UP
- Max torque: 2.0 Nm, max angular velocity: 8.0 rad/s
- Episode: 200 steps at dt=0.05s (10 seconds real time)
- Reward per step: -(angle_from_upright^2 + 0.1*velocity^2 + 0.001*torque^2)
- Best possible reward per step: 0 (balanced, no velocity, no torque)
- Worst possible reward per step: ~-16.27

## Controller Logic

**Phase 1 — Swing-up** (when angle_from_upright > switch_angle):
- Compute current energy: E = 0.5*m*l^2*thdot^2 + m*g*l*(1-cos(th))
- Apply torque = swing_gain * sign(velocity) * cos(theta) to pump energy
- Subtract damping * velocity
- Clamp to [-max_swing_torque, max_swing_torque]

**Phase 2 — Balance** (when angle_from_upright <= switch_angle):
- Compute anticipated angle error = error + anticipation * velocity * dt
- If |anticipated error| < deadzone: torque = 0
- Otherwise: torque = -balance_gain_p * error - balance_gain_d * velocity - damping * velocity
- Clamp to [-max_balance_torque, max_balance_torque]

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the physics engine (read-only)

## Scoring
- Metric: **Weighted combination** of reward normalization and balance time, averaged over 50 episodes
- Direction: **higher is better**
- Formula: score = 0.3 * reward_score + 0.7 * balance_fraction
  - reward_score = (total_reward - min_possible) / (max_possible - min_possible)
  - balance_fraction = fraction of steps where angle_from_upright < 0.5 rad AND velocity < 2.0 rad/s
- This ensures agents must actually swing up AND stay balanced, not just avoid the worst states
- A score of 0.3-0.5 is baseline. 0.7+ is good. 0.85+ is strong. 0.95 is near-optimal.

## Tips for Agents
1. **The swing-up is the hard part** — the pendulum starts hanging down. It must gain enough energy to reach the top before the balance controller can take over.
2. **energy_target matters** — set too low and it won't swing high enough; set too high and it overshoots and wastes time.
3. **switch_angle is critical** — switch to balance too early and the PD controller can't hold it; too late and you waste time swinging.
4. **PD gains interact** — high proportional gain causes oscillation if derivative gain is too low. Start with a 4:1 or 5:1 ratio of P:D.
5. **max_torque = 2.0 is the physics limit** — setting max_swing_torque or max_balance_torque above 2.0 is invalid, but using the full range helps.
6. **damping stabilizes** — a small damping term (0.1–1.0) reduces oscillation in both phases.
7. **anticipation helps balance** — predicting where the angle will be helps the PD controller react earlier.
8. **deadzone prevents chatter** — when nearly balanced, small errors cause torque flickering. A tiny deadzone (0.01–0.1) helps.
9. **Test one thing at a time** — change one parameter, measure the effect, record it.
10. **Random initial conditions** — every episode starts at a different angle and velocity. Your controller must be robust.

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- `status`: keep / discard / crash / retest
- ALWAYS append with `>>`, never overwrite

### best/ (update only when you beat the global best)
```bash
cp config.yaml $(dirname $0)/best/config.yaml
```

### blackboard.md (shared collaboration, append-only)
```
CLAIM agentN: <finding> (evidence: <experiment_id>, <metric>)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
REQUEST agentN to agentM|any: <what to test> (priority: high|medium|low)
```

### Memory files (per-agent, private)
- `memory/facts.md` — confirmed findings
- `memory/failures.md` — dead ends, NEVER retry
- `memory/hunches.md` — worth testing later
- `scratch/hypothesis.md` — current theory
- `scratch/predictions.md` — predicted vs actual score

## Agent Lifecycle
1. Read strategy.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/config.yaml config.yaml`
4. Apply your changes, predict expected score
5. Run: `bash run.sh config.yaml`
6. Record everything: results.tsv, done/, predictions
7. Update memory: facts if confirmed, failures if dead end, hunches if unclear
8. If new best: update best/ + strategy.md + CLAIM on blackboard
9. If queue empty: become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Invalid configs score 0.0

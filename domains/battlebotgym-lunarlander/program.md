# Lunar Lander — Agent Instructions

## Task
Tune a PID-like autopilot controller by editing `config.yaml` to **maximize average landing score** across randomized landing scenarios.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (0.0-1.0) is printed to stdout
# Landing summary is printed to stderr
```

**Budget:** ~2 seconds per experiment (50 episodes)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the autopilot controller parameters

### Parameters and Ranges

**Descent Control:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| descent_rate_target | 0.01-0.3 | Desired descent speed as fraction of total altitude |
| hover_gain | 0.0-5.0 | Main thruster proportional gain (responsiveness) |

**Horizontal Correction:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| horizontal_correction_gain | 0.0-5.0 | How aggressively to correct horizontal drift |
| horizontal_deadzone | 0.0-5.0 | Ignore horizontal error smaller than this (meters) |

**Angle Correction:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| angle_correction_gain | 0.0-5.0 | How aggressively to straighten up |
| angular_damping | 0.0-5.0 | How much to counter angular velocity |

**Phase Transitions:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| altitude_thrust_threshold | 0.0-1.0 | Altitude fraction to start active descent control |
| final_approach_altitude | 0.0-0.5 | Altitude fraction to switch to gentle final approach |
| final_approach_vy_target | 0.01-0.15 | Gentle descent rate for final approach |

**Safety:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| fuel_reserve | 0.0-50.0 | Stop thrusting when fuel drops below this |
| suicide_burn_threshold | 0.0-0.5 | Altitude fraction for emergency max braking |

### Physics Summary
- Lander starts at y=20m with random horizontal offset (up to 5m), velocity, angle, and angular velocity
- Gravity: -1.0 m/s^2 (moon-like), timestep dt=0.1
- Main thruster: 2.0 m/s^2, angled with lander body (tilted lander = angled thrust)
- Side thrusters: 0.5 m/s^2 horizontal, separate rotation thruster at 0.3 rad/s^2
- Random wind gusts perturb velocity and rotation each step
- Fuel starts at 100; main costs 0.5/step, side costs 0.15/step, rotation costs 0.075/step
- Landing: y <= 0. Crash if |vy| > 2.0 or |vx| > 1.5 or |angle| > 0.5 rad
- Max 300 steps; timeout (still airborne) = score 0.0

### Scoring (per episode)
- Crash = 0.0
- Timeout = 0.0
- Safe landing = 1.0 - 0.25*|x|/3 - 0.30*|vy|/2 - 0.15*|vx|/1.5 - 0.15*|angle|/0.5 - 0.15*fuel_used/100
- Clamped to [0, 1]
- Overall = average across all episodes

### Controller Architecture
The autopilot has four modes based on normalized altitude (y/20):
1. **High altitude** (y_norm > altitude_thrust_threshold): Coast, only brake if descending too fast. Correct angle and drift.
2. **Descent** (final_approach < y_norm < altitude_thrust_threshold): Actively control descent rate using hover_gain. Correct position.
3. **Final approach** (suicide_burn < y_norm < final_approach): Gentle touchdown at final_approach_vy_target.
4. **Suicide burn** (y_norm < suicide_burn and falling fast): Emergency max thrust.

Angle correction uses a PD controller (gain + damping). Horizontal correction uses a proportional controller with deadzone.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the game engine (read-only)

## Scoring
- Metric: **Average landing score** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: ~0.37 (default config: lands ~70% but imprecisely, crashes ~30%)
- Noise level: differences > 0.03 are signal (50 episodes gives ~+/-0.03 noise)
- A score of 0.60+ is decent. 0.80+ is strong. 0.85+ is excellent.

## Tips for Agents
1. **altitude_thrust_threshold is the most critical parameter** — too low and you crash (no time to brake), too high and you waste fuel hovering.
2. **descent_rate_target controls the tradeoff** between crashing (too fast) and timing out (too slow).
3. **hover_gain must match descent_rate_target** — high gain + low descent target = oscillation; low gain + high target = crashes.
4. **The final approach phase is where precision happens** — tune final_approach_altitude and final_approach_vy_target for soft touchdowns.
5. **Main thruster is body-aligned** — when tilted, thrust has a horizontal component. This couples angle to trajectory.
6. **Fuel is limited** — 100 units total. Aggressive thrusting early means running out later. fuel_reserve prevents empty-tank crashes.
7. **Wind adds noise** — random horizontal and rotational gusts mean you need robust parameters, not fragile ones.
8. **horizontal_deadzone prevents over-correction** — small corrections at low error waste fuel.
9. **Test one thing at a time** — change one parameter, measure the effect, record it.
10. **Parameter interactions matter a lot** — altitude_thrust_threshold, descent_rate_target, and hover_gain form a strongly-coupled triple.

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
8. If new best -> update best/ + strategy.md + CLAIM on blackboard
9. If queue empty -> become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Invalid parameters will use defaults (won't crash, but may perform poorly)

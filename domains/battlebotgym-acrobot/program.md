# Acrobot Swing-Up — Agent Instructions

## Task
Optimize an energy-based swing-up controller for a two-link pendulum (Acrobot) by editing `config.yaml` to **minimize the number of steps** needed to swing the tip above the goal height.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (0.0–0.84) is printed to stdout
# Episode summary is printed to stderr
```

**Budget:** ~2 seconds per experiment (50 episodes)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the controller parameters

### Parameters and Ranges

**Energy Pumping Phase** (controls how energy is injected into the swing):
| Parameter | Range | Effect |
|-----------|-------|--------|
| energy_pump_gain | 0.0–10.0 | How aggressively to pump energy into the system |
| phase_offset | -3.14–3.14 | Timing offset for when torque is applied vs swing phase |
| energy_target | 0.0–50.0 | Total energy level to build before attempting the goal |
| sign_strategy | 0.0–1.0 | 1.0 = bang-bang (±1 torque), 0.0 = proportional control |

**Catch/Stabilize Phase** (controls behavior near the goal):
| Parameter | Range | Effect |
|-----------|-------|--------|
| catch_threshold | 0.0–2.0 | How close to goal height before switching to catch mode |
| catch_gain | 0.0–10.0 | Controller gain during catch/stabilize phase |

**Linear Controller Weights** (used in catch phase):
| Parameter | Range | Effect |
|-----------|-------|--------|
| theta1_weight | 0.0–10.0 | Weight of first joint angle in torque decision |
| theta2_weight | 0.0–10.0 | Weight of second joint angle |
| dtheta1_weight | 0.0–10.0 | Weight of first joint angular velocity |
| dtheta2_weight | 0.0–10.0 | Weight of second joint angular velocity |

**Mode Switch:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| velocity_threshold | 0.0–5.0 | Angular velocity threshold (used for mode logic) |

## Physics Background
The Acrobot is a two-link pendulum:
- Link 1 is attached to a fixed point (like a gymnast's hands on a bar)
- Link 2 hangs from link 1 (like the gymnast's body)
- Only the joint between link 1 and link 2 is actuated (torque ∈ {-1, 0, +1})
- Goal: swing the tip of link 2 above `fixed_point_height + link_length_1`
  - Mathematically: `-cos(θ₁) - cos(θ₁ + θ₂) > 1.0`
- Start: both links hanging straight down (θ₁ ≈ θ₂ ≈ 0, small random perturbation)
- Max 500 steps per episode, dt = 0.2

The challenge: you can only apply torque at the middle joint, but you need to get the tip above the bar. This requires energy pumping — timing your torque to gradually build up the pendulum's swing amplitude until you can reach the top.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the simulator (read-only)

## Scoring
- Metric: **avg(1.0 - steps_to_goal / 500)** across 50 episodes
- Direction: **higher is better**
- If goal is not reached in 500 steps, that episode scores 0.0
- Optimal: ~0.75 (goal in ~125 steps consistently, 100% success rate)
- Baseline: ~0.24 (default config — sometimes swings up but 70% of episodes fail)
- A score of 0.4+ is good. 0.6+ is strong. 0.70+ is near-optimal.

## Tips for Agents
1. **Energy pumping is the key** — the acrobot needs to build energy by pumping at the right phase. `energy_pump_gain` and `phase_offset` are the most important parameters.
2. **Timing matters** — `phase_offset` controls when torque is applied relative to the swing. The right offset makes energy injection efficient; the wrong one wastes energy or damps the swing.
3. **Bang-bang vs proportional** — `sign_strategy` near 1.0 gives maximum torque (±1) which usually pumps faster. Proportional (near 0.0) is smoother but may be slower.
4. **Don't over-pump** — if `energy_target` is too high, the system builds too much energy and overshoots the goal. Too low, and it never reaches.
5. **Catch phase is critical** — even if you pump enough energy, you need the catch controller to actually stabilize at the goal. The theta/dtheta weights determine how well the controller catches.
6. **Interactions between params** — `energy_pump_gain` interacts with `energy_target` and `phase_offset`. Explore the space systematically.
7. **Test one thing at a time** — change one parameter, measure the effect, record it.

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
8. If new best → update best/ + strategy.md + CLAIM on blackboard
9. If queue empty → become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Torque is discretized to {-1, 0, +1} by the engine
- 500 max steps per episode — no extensions

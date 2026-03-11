# CartPole — Agent Instructions

## Task
Tune a parameterized controller via `config.yaml` to **maximize pole-balancing survival time** on the classic CartPole problem.

A pole is attached by a hinge to a cart on a frictionless track. Each timestep, the controller pushes the cart left or right. The episode ends when the pole falls (angle > 12 degrees), the cart goes off-screen (position > +/-2.4), or 500 steps are reached (optimal).

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (0.0–1.0) is printed to stdout
# Episode summary is printed to stderr
```

**Budget:** ~1 second per experiment (50 episodes)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the controller's weights and parameters

### Parameters and Ranges

**Pole Angle Response:**
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| angle_weight | 0.0–10.0 | 0.0 | How much pole angle drives the push direction |
| angular_velocity_weight | 0.0–10.0 | 1.0 | How much angular velocity matters |

**Cart Position Response:**
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| position_weight | 0.0–5.0 | 0.0 | How much cart position affects decision |
| velocity_weight | 0.0–5.0 | 0.0 | How much cart velocity matters |

**Decision Shaping:**
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| angle_bias | -2.0–2.0 | 0.0 | Constant offset added to signal (asymmetric push) |
| response_sharpness | 0.1–20.0 | 1.0 | Sigmoid steepness — higher = more binary decisions |

**Advanced:**
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| anticipation_horizon | 0.0–2.0 | 0.0 | Weight on predicted future angle (look-ahead) |
| position_centering | 0.0–5.0 | 0.0 | Bias pushing cart back toward track center |

### How the Controller Works

```
signal = angle * angle_weight
       + angular_velocity * angular_velocity_weight
       + position * position_weight
       + velocity * velocity_weight
       + angle_bias

signal += (angle + angular_velocity * 0.02 * anticipation_horizon) * anticipation_horizon
signal -= position * position_centering

prob(push_right) = sigmoid(signal * response_sharpness)
```

If prob >= 0.5, push right; otherwise push left.

### Why the Default Scores ~0.39

The default controller only reacts to angular velocity — it corrects when the pole is
already falling but has no sense of where the pole or cart actually are. This causes
the cart to drift off-screen. Adding angle awareness and position feedback should
dramatically improve survival.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the simulator (read-only)

## Scoring
- Metric: **Average survival steps / 500** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: ~0.39 (default angular-velocity-only controller)
- Optimal: 1.0 (500 steps every episode = perfect balance)
- A score of 0.70+ is good. 0.90+ is excellent. 1.0 is perfect.

## Tips for Agents
1. **The default is blind to angle** — it only sees angular velocity. Adding even small `angle_weight` should help a lot.
2. **Position matters** — without position awareness, the cart drifts to the edge. Use `position_weight` or `position_centering` to stay on the track.
3. **response_sharpness controls decisiveness** — too low and the controller is indecisive; too high and it oscillates. Find the sweet spot.
4. **anticipation_horizon is look-ahead** — predicting where the pole will be lets you react earlier, preventing oscillation.
5. **angle_bias should stay near 0** — any bias introduces asymmetry that makes the pole fall to one side.
6. **This is a well-known control problem** — the optimal solution is a linear controller. The parameter space encodes one. Finding good weights is the challenge.
7. **Test one thing at a time** — change one parameter, measure the effect, record it.
8. **Interactions matter** — high sharpness + high weights = different from either alone.

## Physics Reference
- Gravity: 9.8 m/s^2
- Cart mass: 1.0 kg, Pole mass: 0.1 kg
- Pole half-length: 0.5 m
- Force: 10.0 N (left or right)
- Timestep: 0.02 s (Euler integration)
- Fail conditions: |angle| > 12 degrees OR |position| > 2.4 m
- Max steps: 500

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
1. Read program.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/config.yaml config.yaml`
4. Apply your changes, predict expected score
5. Run: `bash run.sh config.yaml`
6. Record everything: results.tsv, done/, predictions
7. Update memory: facts if confirmed, failures if dead end, hunches if unclear
8. If new best → update best/ + CLAIM on blackboard
9. If queue empty → become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Invalid configs score 0.0

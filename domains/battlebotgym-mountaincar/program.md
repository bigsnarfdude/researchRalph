# MountainCar — Agent Instructions

## Task
Optimize a momentum-building policy by editing `config.yaml` to **maximize the fraction of episodes where the car reaches the goal** (position >= 0.5) within 200 steps.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (goal rate 0.0-1.0) is printed to stdout
# Episode summary is printed to stderr
```

**Budget:** ~1 second per experiment (50 episodes)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the policy parameters that control the car's behavior

### Parameters and Ranges

**Momentum Building:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| momentum_threshold | -0.07 to 0.07 | Velocity threshold to switch push direction |
| position_switch_point | -1.2 to 0.0 | Position where strategy changes from swing to push |
| swing_aggression | 0.0 to 1.0 | How aggressively to commit to swinging vs coasting |

**Signal Weights:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| peak_detection_weight | 0.0 to 5.0 | Weight on detecting swing peaks (velocity near zero) |
| velocity_gain | 0.0 to 10.0 | Amplification of velocity in push signal |
| position_gain | 0.0 to 10.0 | Amplification of position in push signal |

**Goal Seeking:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| energy_threshold | 0.0 to 1.0 | Estimated energy above which to push right toward goal |
| coast_zone_width | 0.0 to 0.5 | Velocity range around zero where car coasts |
| right_bias | -1.0 to 1.0 | Constant bias toward pushing right |

### Physics
- Position range: -1.2 to 0.6
- Velocity range: -0.07 to 0.07
- Start: random position in [-0.6, -0.4], velocity = 0
- Actions: push left (0), no push (1), push right (2)
- Physics: `velocity += (action - 1) * 0.001 - cos(3 * position) * 0.0025`
- Episode ends when position >= 0.5 (success) or 200 steps (timeout = failure)

### Key Insight
The car cannot climb the right hill directly. It must swing left to build momentum, then use that momentum to push right. The optimal strategy alternates pushing left when moving left and pushing right when moving right — building energy each swing until it can crest the hill.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the game engine (read-only)

## Scoring
- Metric: **Goal-reaching rate** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: ~0.3-0.5 (default config reaches goal sometimes)
- Optimal: 1.0 (good policy reaches goal every episode)
- Known optimal: true (a well-tuned momentum policy reaches the goal reliably)

## Tips for Agents
1. **The core problem is momentum building** — the car must swing back and forth. The key parameters are `momentum_threshold`, `position_switch_point`, and `swing_aggression`.
2. **Push WITH the car's velocity** — push left when moving left, push right when moving right. This builds energy each swing.
3. **`swing_aggression` is critical** — too low and the car coasts too much, too high might cause bad timing.
4. **`position_switch_point` controls the transition** — where the car stops swinging and commits to pushing right.
5. **`energy_threshold` determines patience** — too low and it tries for the goal before having enough speed, too high and it swings forever.
6. **Interactions matter** — `momentum_threshold` and `coast_zone_width` together determine how the car behaves near zero velocity.
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
8. If new best -> update best/ + strategy.md + CLAIM on blackboard
9. If queue empty -> become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Invalid configs score 0.0

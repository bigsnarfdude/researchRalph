# Arena Combat — Agent Instructions

## Task
Optimize a combat bot's strategy by editing `config.yaml` to **maximize win rate** against a fixed champion bot in a grid-based arena.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (win rate 0.0–1.0) is printed to stdout
# Match summary is printed to stderr
```

**Budget:** ~5 seconds per experiment (50 matches)

For detailed results:
```bash
python3 arena.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the challenger bot's stats and strategy weights

### Parameters and Ranges

**Core Stats** (these cost stat budget points):
| Parameter | Range | Cost Weight | Effect |
|-----------|-------|-------------|--------|
| max_hp | 50–200 | 1.0x | Total hit points |
| max_energy | 50–200 | 0.8x | Energy pool for actions |
| attack_power | 5–30 | 4.0x | Base damage per attack |
| defense | 5–40 | 3.0x | Reduces incoming damage |
| speed | 1–4 | 15.0x | Tiles moved per turn |
| heal_amount | 0–25 | 3.0x | HP restored per heal |
| attack_range | 1–6 | 12.0x | How far attacks reach |
| energy_regen | 1–12 | 5.0x | Energy per turn |
| special_power | 1.0–4.0 | 20.0x | Special attack multiplier |

**Combat Params** (no stat budget cost, but affect play):
| Parameter | Range | Effect |
|-----------|-------|--------|
| attack_cost | 5–25 | Energy per normal attack |
| heal_cost | 5–40 | Energy per heal |
| special_cooldown | 2–10 | Turns between special attacks |

**Strategy Weights** (no stat budget cost — pure strategy):
| Parameter | Range | Effect |
|-----------|-------|--------|
| aggression | 0.0–1.0 | Tendency to charge vs maintain distance |
| retreat_threshold | 0.0–0.6 | HP% below which bot retreats |
| heal_threshold | 0.0–0.8 | HP% below which bot heals |
| kite_distance | 0–8 | Preferred distance when playing defensive |
| burst_threshold | 0.05–0.5 | Enemy HP% below which bot goes all-in |

### CRITICAL: Stat Budget Constraint
Your total stat cost must be within 5% of the champion's budget (~308 points).
The cost formula weights expensive stats (attack_power, special_power, attack_range)
much higher than cheap stats (max_hp, max_energy).

**This means you must make tradeoffs.** Want more attack power? Sacrifice HP or energy.
Want longer range? Cut defense or speed. The champion is balanced — can you find a
specialization that beats balanced?

### Strategy Archetypes to Explore
- **Berserker**: High aggression, high attack, low defense, rush them down
- **Sniper**: High range, high kite_distance, low aggression, chip from afar
- **Turtle**: High defense, high heal, low aggression, outlast them
- **Assassin**: High burst_threshold, high special_power, wait for the kill
- **Brawler**: High HP, medium attack, high aggression, win by attrition
- **Glass Cannon**: Max attack+special, min HP, kill before being killed

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `arena.py` — the game engine (read-only)

## Scoring
- Metric: **Win rate** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: 0.50 (champion vs itself is ~50%)
- Noise level: differences > 0.04 are signal (50 matches gives ~±0.04 noise)
- A score of 0.70+ is strong. 0.80+ is excellent. 0.90+ is exceptional.

## Tips for Agents
1. **Strategy weights are free** — they don't cost stat budget. Optimizing aggression, retreat_threshold, heal_threshold, kite_distance, and burst_threshold is pure upside.
2. **The champion is balanced** — beating it requires finding an asymmetric advantage.
3. **attack_cost and heal_cost matter** — cheaper attacks = more DPS, but you can't change these without tradeoffs in energy management.
4. **speed is very expensive** (15x cost weight) — going from 2 to 3 speed costs 15 budget points. Often not worth it.
5. **Test one thing at a time** — change one parameter, measure the effect, record it.
6. **Interactions exist** — high aggression + low retreat = different from either alone. Explore combos.
7. **The game has hidden dynamics** — damage reduction is nonlinear (defense / (defense + 50)), so there are diminishing returns. Healing costs energy that could be spent attacking.

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
- `memory/facts.md` — confirmed findings (e.g., "aggression 0.8 > 0.5, confirmed 0.62 win rate")
- `memory/failures.md` — dead ends, NEVER retry (e.g., "speed=4 busts budget")
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
- Stat budget must be within 5% of champion (~308 points max)
- All parameters must be within their stated ranges
- Invalid configs score 0.0

# Resource Economy — Agent Instructions

## Task
Optimize a village's resource allocation strategy by editing `config.yaml` to **maximize survival rate** over 100 simulated days. A village of 100 people faces drought, raids, plagues, and bounties. Your goal: keep as many villagers alive as possible.

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (survival rate 0.0–1.0) is printed to stdout
# Simulation summary is printed to stderr
```

**Budget:** ~5 seconds per experiment (50 simulations)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

## What you edit
- `config.yaml` — the village's worker allocation and strategy parameters

### Parameters and Ranges

**Worker Allocation** (must sum to 100):
| Parameter | Range | Effect |
|-----------|-------|--------|
| farming_pct | 0–100 | % of villagers farming. More farmers = more food. |
| mining_pct | 0–100 | % of villagers mining materials for buildings. |
| building_pct | 0–100 | % of villagers constructing buildings (defense + production bonus). |
| research_pct | 0–100 | % of villagers doing research (farm/mine/trade/medicine bonuses). |

**Strategy Parameters** (pure strategy — no cost):
| Parameter | Range | Effect |
|-----------|-------|--------|
| food_reserve_target | 0–500 | Target food stockpile. Triggers starvation_response when food drops below. |
| trade_aggression | 0.0–1.0 | How aggressively to trade surplus for gold and buy food in shortages. |
| building_priority | 0.0–1.0 | Defense vs production building emphasis (currently flavor). |
| research_focus | 0.0–1.0 | Medicine vs agriculture research emphasis (currently flavor). |
| raid_preparedness | 0.0–1.0 | Reduces raid deaths and resource theft (up to 40% reduction). |
| starvation_response | 0.0–1.0 | How aggressively to shift workers to farming during food crisis. |

**Trade Thresholds** (pure strategy):
| Parameter | Range | Effect |
|-----------|-------|--------|
| trade_food_threshold | 0–500 | Sell food when above this + food_reserve_target. |
| trade_materials_threshold | 0–500 | Sell materials when above this amount. |
| gold_reserve_target | 0–200 | Gold to keep in reserve (don't sell below this). |
| emergency_food_buy | 0.0–1.0 | Fraction of gold to spend buying food in emergencies. |

### CRITICAL: Worker Allocation Constraint
Your worker percentages must sum to exactly 100. This is the core tradeoff. Every worker assigned to building or research is one NOT producing food.

**This means you must make tradeoffs.** Want more research for long-term bonuses? You'll starve early. Want all farmers? No buildings means raids devastate you.

### Resource Mechanics
- **Food**: Produced by farmers (2.3/farmer/day base). Each villager eats 1.1 food/day. Stored food spoils 2.5%/day.
- **Materials**: Produced by miners (1.8/miner/day base). Used by builders (0.5/builder/day).
- **Gold**: Earned by trading surplus food/materials. Spent to buy emergency food.
- **Knowledge**: Produced by researchers (0.45/researcher/day base). Permanently boosts farm output (+0.4%/point), mine output (+0.3%/point), trade rates (+0.2%/point), and plague resistance (+0.4%/point).
- **Buildings**: Constructed by builders (0.7/builder/day base, requires materials). Permanently boost farm output (+0.3%/point), mine output (+0.2%/point), and defense rating (0.25/point).

### Event System
Events happen randomly each day (seeded RNG for reproducibility across runs):
- **Drought** (~7%/day, seasonal): Reduces farming to 30% output for 10 days. Devastating if food reserves are low.
- **Raid** (~6%/day, wealth-attracted): Kills up to 8 villagers, steals food/materials/gold, damages buildings. Defense rating and raid_preparedness reduce severity.
- **Plague** (~4.5%/day, density-dependent): Kills ~10% of population (reduced by medicine/research), reduces all productivity for 6 days.
- **Bounty** (~7%/day): Free food, gold, and materials. Pure luck.

Events can stack (drought + raid on the same day).

### Strategy Archetypes to Explore
- **Agrarian**: 70%+ farming, high food reserves, survive by never starving
- **Fortress**: Heavy building + mining, high raid_preparedness, survive by defense
- **Scholar**: Heavy research early, accept early losses for compounding late-game bonuses
- **Trader**: High trade_aggression, convert everything to gold, buy food as needed
- **Balanced**: Even split, moderate everything, hope for average luck
- **Adaptive**: High starvation_response, moderate everything, let the system self-correct

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the game engine (read-only)

## Scoring
- Metric: **Survival rate** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: 0.50–0.60 (balanced default)
- Noise level: differences > 0.03 are signal (50 simulations gives ~+-0.03 noise)
- A score of 0.70+ is strong. 0.80+ is excellent. 0.90+ is exceptional.

## Key Tradeoffs (non-obvious)
1. **Farming vs everything else**: More farmers means fewer deaths from starvation, but no buildings means raids are devastating, and no research means plagues kill more.
2. **Food reserve target**: Too low and one drought wipes you out. Too high and you waste workers maintaining a buffer instead of investing.
3. **Research timing**: Research compounds over time (knowledge permanently boosts everything), but the first 20 days are the most dangerous — you need food NOW.
4. **Building trap**: Buildings require materials. Assigning builders without miners means builders sit idle consuming food.
5. **Trade aggression**: Aggressive trading can save you in emergencies but selling food for gold is risky if a drought follows.
6. **Starvation response**: Too high causes worker oscillation (farm crisis -> all farm -> surplus -> shift away -> crisis again). Too low means slow recovery from food shocks.
7. **Raid defense**: Defense reduces raid damage, but buildings that provide defense take workers away from farming. A rich village attracts MORE raids.

## Tips for Agents
1. **Test the default first** — understand the baseline before changing anything.
2. **One variable at a time** — change one parameter, measure the effect, record it.
3. **Watch starvation_days in the summary** — if nonzero, food production is the bottleneck.
4. **Watch wipe_count** — total wipes (0 survivors) are catastrophic; avoid them first.
5. **The early game is dangerous** — days 1-30 have no knowledge or building bonuses yet. Survive early, thrive late.
6. **Compounding effects are real** — research and buildings provide multiplicative bonuses. By day 80, a village with 30 knowledge points produces 18% more food per farmer.
7. **Events are seeded** — same seed = same events. The --matches flag runs different seeds. Optimize for average performance, not one lucky run.
8. **Mining without building is waste, building without mining is waste** — these two are coupled. If you cut one, cut both.

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
- `memory/facts.md` — confirmed findings (e.g., "farming_pct 60 > 40, survival 0.68 vs 0.54")
- `memory/failures.md` — dead ends, NEVER retry (e.g., "research_pct 50 starves by day 15")
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
- Worker allocation must sum to 100
- All parameters must be within their stated ranges
- Invalid configs score 0.0

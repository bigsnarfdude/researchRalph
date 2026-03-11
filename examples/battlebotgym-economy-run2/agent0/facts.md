# Confirmed Facts

## True Score (5000-sim verified)
- **Optimization ceiling: 0.794 ± 0.002** across all top configs
- 0/15/45/40: 0.7941 (5000 sims)
- 0/15/40/45: 0.7926 (5000 sims)
- 0/15/42/43: 0.7933 (5000 sims)
- All configs in the 0-10f/10-15m/40-45b/35-45r region are equivalent

## Engine Mechanics
- emergency_food_buy BLOCKS material selling (early return at line 285) when food < pop*2
- Early game (days 0-10): material→gold pipeline is INACTIVE because emergency buy blocks sell phase
- Only starvation_response auto-farming provides food in early game
- ALL deaths come from early game events (days 0-20). By day 40+ village is invulnerable
- Worst seeds: back-to-back droughts + plague in first 10 days → 70-87% deaths
- 16/200 seeds score below 50%. These dominate the average score

## Strategy Params (all confirmed at 200+ sims)
- starvation_response=1.0: monotonically best (each 0.1 drop costs ~0.02)
- raid_preparedness=1.0: removing costs ~0.08
- trade_aggression=1.0: best (0.0 is dead at 0.62)
- emergency_food_buy=1.0: best (0.0 costs ~0.06)
- trade_materials_threshold=0: best (higher = less gold = worse)
- food_reserve_target: 200-500 all equivalent (~0.80), below 150 is bad
- research_focus and building_priority: FLAVOR ONLY — zero effect
- gold_reserve_target=0: best (hoarding doesn't help)
- trade_food_threshold: 0 or 500 equivalent (with low farming, never have food to sell)

## Worker Allocation
- Mining floor is ~10 (5 kills builders due to material shortage)
- Building + Research should total 80-85%, roughly equal (40/40 to 45/40)
- Farming 0-15% barely matters — starvation_response=1.0 auto-farms
- Extreme allocations (80%+ anything) always lose
- More farming helps worst-case seeds but hurts average — net negative tradeoff

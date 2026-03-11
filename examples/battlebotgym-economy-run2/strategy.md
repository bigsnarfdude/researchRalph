# Search Strategy — economy

## Current Best
- Score: ~0.80 (true score at 1000+ matches)
- Config: best/config.yaml (5/15/45/35 frt=500, all strategy maxed)

## Phase: COMPLETE — optimization surface fully mapped

## Optimal Config Template
- Worker allocation: any in {3-12}f/{10-17}m/{40-45}b/{35-40}r (all equivalent)
- Strategy: trade_agg=1.0, starvation_response=1.0, raid_prep=1.0, emergency_food_buy=1.0
- Trade: all thresholds=0, gold_reserve=0
- food_reserve_target: 400-500 (marginal winner over 300 at high sample size)
- research_focus, building_priority: irrelevant (flavor params)

## Why 0.80 is the ceiling
- Event system (7% drought, 6% raid, 4.5% plague) creates irreducible death variance
- Worst-case seeds (back-to-back droughts) cause 0.10-0.30 survival regardless of config
- starvation_response=1.0 dynamically converts all workers to farmers during crises
- This makes base farming_pct nearly irrelevant
- Building defense is asymptotic: diminishing returns past ~100 building points
- Research caps plague resistance at 200 knowledge (~day 13)

## Key Discoveries (chronological)
1. Buildings >> farming for defense (agent0, agent3)
2. Sell-everything (thresholds=0) is critical (agent0)
3. starvation_response=1.0 is most important single param (all agents)
4. Building ≈ research balanced at ~40 each (agent3)
5. Mining floor at 10 (all agents)
6. food_reserve_target=400-500 marginally best (agent0)
7. 50-match scores are noise; true score requires 500+ matches (agent0, agent2)

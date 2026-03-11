# Dead Ends — NEVER Retry

## Allocation
- Pure agrarian 90f: 0.4172
- 55f/20m/10b/15r: 0.4996 (mine >> build imbalance)
- Scholar 40f/10m/10b/40r: 0.5970 (too much research, too little food)
- Zero mining 0/0/50/50: 0.6172 (no gold pipeline, idle builders)
- Zero building 0/12/0/88: 0.4440 (raids devastate)
- Extreme mining 0/50/25/25: 0.756 (too much mining)
- Building 60%: 0/12/60/28 = 0.794 (diminishing returns on building)
- Research 70%: 0/5/25/70 = 0.789 (not enough defense)

## Strategy
- starvation_response < 1.0: always worse (0.9 = -0.03, 0.5 = disaster)
- trade_aggression < 0.5: always worse
- trade_aggression = 0.0: 0.622 (dead)
- trade_materials_threshold=50: 0.776 (delays gold pipeline)
- food_reserve_target=0: 0.584 (starvation_response never triggers)
- food_reserve_target=200: 0.796 (slightly late response)

## Hypotheses that failed
- frt=333 "sweet spot" (regular buy = emergency threshold): no effect at 200 sims
- More mining for gold pipeline: mining 15-20% same as 12% at plateau
- More farming reduces variance: same stdev, same mean
- Research rush (53-60%): doesn't help enough vs defense loss

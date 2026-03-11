# Dead Ends — NEVER retry

## Extreme Allocations
- Pure agrarian (90% farm, 0% research): 0.4172
- Near-total research (0/5/5/90): 0.657
- No-research fortress (0/50/50/0): 0.651
- Extreme building (0/10/80/10): 0.632
- No-builder (0/30/0/70): 0.434
- Farm+build no-mining (50/0/50/0): 0.231
- Pure builder (0/10/90/0): 0.519
- 100% research: 0.49
- Inverse of best (40/40/10/10): 0.660
- Micro-support (1/1/49/49): 0.662

## Strategy Param Dead Ends
- food_reserve_target=0: 0.58 (disables starvation_response)
- food_reserve_target=50: 0.436 (response triggers too late)
- starvation_response=0.3: 0.457 (deaths snowball)
- starvation_response < 0.8: worse than 1.0 monotonically
- trade_aggression=0: 0.622 (no gold pipeline)
- emergency_food_buy=0: 0.745 (can't buy food in crisis)
- All strategy params zeroed: 0.676
- Gold hoarding (gold_reserve=200): no improvement
- Material hoarding (threshold=500): 0.775 (worse, no gold)
- Never sell anything (both thresholds=500): 0.760

## Search Methods That Failed
- Random search (200+ configs): no config above 0.82
- Deliberate-death strategy (low sr for pop reduction): deaths snowball
- Gold engine (heavy mining 30-40%): no improvement (gold attracts raids)
- Anti-oscillation (moderate sr + base farming): worse than sr=1.0

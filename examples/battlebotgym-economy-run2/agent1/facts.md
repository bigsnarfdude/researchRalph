# Confirmed Facts

## True Scores (1000-sim verified)
- Best: 0/12/45/43, food_reserve=280 = 0.8049
- 0/15/45/40, food_reserve=300 = 0.8031
- 0/12/45/43, food_reserve=300 = 0.8038
- All top configs converge to ~0.803 at 1000 sims

## Worker Allocation
- Farming 0% is optimal (starvation_response=1.0 dynamically handles it)
- Mining 10-15% optimal (10 minimum for builders, 15 adds trade gold)
- Building 42-47% optimal (defense + production bonuses)
- Research 37-45% optimal (plague resistance + compounding bonuses)
- Building + Research should be ~85-88% combined

## Strategy Parameters
- starvation_response=1.0 CRITICAL (0.5 drops to 0.65)
- raid_preparedness=1.0 optimal
- trade_aggression=1.0 optimal
- emergency_food_buy=1.0 optimal
- food_reserve_target=280-300 optimal
- All trade thresholds=0, gold_reserve=0 optimal
- building_priority and research_focus are FLAVOR (no effect)
- trade_food_threshold irrelevant (low-farm configs never have surplus)

## Game Mechanics Insights
- Starvation response shifts ALL workers to farming when food < food_reserve/2
- Research caps plague resistance at knowledge=200 (~day 13 with 40 researchers)
- Defense rating has diminishing returns: defense/(defense+50)
- Gold is 3x weighted in raid attraction formula
- Emergency food buy triggers when food < population*2 (always early game)
- Selling materials for gold via trade helps emergency food buying

# Confirmed Facts

## Worker Allocation (DEFINITIVE — 1000-3000 sim verified)
- Optimal: 0/12/45/43 (zero farming) = 0.8015 @2000sim, 0.8049 @1000sim
- Surface is FLAT: all configs in 0-15f/10-15m/40-50b/35-45r score 0.80±0.005
- Farming_pct is irrelevant with starvation_response=1.0 (overrides allocation during crisis)
- Mining floor is 10 (mine=5 starves builders, mine=8 suboptimal)
- Building+research should total ~85-88 for max compounding
- Zero mining: 0.62 (gold pipeline dead). Zero building: 0.44 (raids kill)

## Strategy Parameters (ALL CONFIRMED MONOTONIC)
- food_reserve_target=280 nominal peak but 200-500 all equivalent at 1000+ sims
- starvation_response=1.0 CRITICAL (0.9 = -0.03, 0.5 = disaster)
- raid_preparedness=1.0 optimal
- trade_aggression=1.0 optimal
- emergency_food_buy=1.0 optimal
- trade_food_threshold=0 (no effect with 0% farming)
- trade_materials_threshold=0 (sell everything for gold)
- gold_reserve_target=0 (spend everything)
- research_focus and building_priority are FLAVOR ONLY (no effect)

## Key Mechanic Insights (from deep engine analysis)
- starvation_response triggers at food < frt/2, shifts workers proportionally to farming
- Emergency food buy (food < pop*2) takes priority over regular buy, spends ALL gold
- Regular food buy (food < frt*0.6) is DEAD CODE when frt <= 333 (emergency always fires first)
- Defense: def_rating/(def_rating+50), ~91% reduction at 2000 buildings
- Research: 200 knowledge = max plague resistance (80% reduction, reached ~day 10)
- Drought seasonality: 1.5x more likely on days 20-30, 0.5x on days 60-80
- Gold pipeline: materials → sell → gold → emergency food. Dies during starvation shift.
- Spoilage: 2.5%/day negligible relative to production
- TRUE CEILING: ~0.80. Caused by drought+plague stacking in days 0-5 (8-10% of seeds)

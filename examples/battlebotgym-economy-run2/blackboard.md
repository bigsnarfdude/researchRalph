# Shared Blackboard — economy

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: Buildings+mining >> farming for survival. 25f/30m/30b/15r scores 0.7278 vs baseline 0.5448. Pure agrarian (90% farm) scores only 0.4172. (evidence: EXP-001 through EXP-005, region: worker allocation extremes)
CLAIM agent0: Trade aggression 0.8 + emergency_food_buy 1.0 compensates for low farming. High trade lets you buy food during droughts. (evidence: EXP-005, 0.7278)
CLAIM agent0: NEW BEST 0.785 — CRITICAL DISCOVERY: trade thresholds all at ZERO + gold_reserve=0 is a massive boost. 10f/30m/30b/30r with sell-everything (thresholds=0). Selling ALL surplus instantly converts materials+food to gold for emergency buys. +0.05 over previous best. (evidence: EXP-019, 0.785). Region: trade thresholds (strategy params).
CLAIM agent0: research_focus and building_priority are FLAVOR ONLY — changing them from 0 to 1 has zero effect on score. Don't waste experiments on these. (evidence: EXP-013 vs EXP-014, identical 0.748)
CLAIM agent0: starvation_response is critical — dropping from 0.9 to 0.4 kills score (0.748→0.667). Keep HIGH. (evidence: EXP-012, 0.6668)
RESPONSE agent0 to agent2: confirmed — research 30%+ helps on fortress base. But the real unlock is trade thresholds=0. Try your config with all thresholds zeroed.

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
REQUEST agent0 to any: Refine around 25f/30m/30b/15r — try varying research_pct from 10-25 while keeping build+mine high. I'm moving on to wild configs. (priority: high)

CLAIM agent1: farming60/mining10/building15/research15 achieves 0.7278 survival (evidence: EXP-003). Region: worker allocation. Reducing mining from 20->10 and increasing farming 50->60 is a big win. Research must stay at 15+ (cutting to 10 drops to 0.61). Building must stay at 15 (cutting to 10 drops to 0.50).

CLAIM agent2: NEW BEST 0.7474 — fortress-scholar hybrid 25f/20m/25b/30r with trade_agg=0.8, starvation_response=0.9. Pushing research to 30% (up from agent0's 15%) while reducing mine+build slightly BEATS the pure fortress. Research compounding is real. (evidence: EXP-a2-003, 0.7474). Region: low-farm, balanced mine/build, high research.
RESPONSE agent2 to agent0: Partial refute — you don't need build+mine=60, 45 is enough if you add research. 30% research compensates for fewer builders via compounding bonuses.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 22:24] Top 5 agents within 2% (best: 0.7986). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent0: Asymmetric build/mine is better than equal. 15m/45b outperforms 30m/30b (0.7922 vs 0.785). Building is ~3x more valuable than mining. But 10m/50b slightly worse — need some mining floor. (evidence: EXP-025 through EXP-031)
CLAIM agent0: 0% farming is VIABLE (0.7842) — trade system alone can feed village. But 5-10% farming gives marginal boost. (evidence: EXP-020, EXP-019)
REQUEST agent0 to any: Try high-farm configs WITH sell-everything thresholds=0 — agent1's 60f config might jump if trade params are optimized. (priority: medium)

CLAIM agent1: 55/5/15/25 with raid_prep=1.0, starv_resp=1.0 achieves 0.7338 (evidence: EXP-038). Key findings: research compounding is powerful (20->25%), mining can go as low as 5 (but not 0), starvation_response=1.0 is optimal, raid_preparedness=1.0 is optimal. Region explored: worker alloc 50-65 farming, 0-20 mining, 10-15 building, 15-30 research + all strategy params.

CLAIM agent2: NEW BEST 0.7944 — starvation_response=1.0 + food_reserve=500 on 30/10/30/30. Max starvation_response beats 0.9. Zero trading KILLS fortress (0.622). (evidence: EXP-a2-015, 0.7944). Region: strategy params tuning.
CLAIM agent2: Zero trade_aggression is dead end (0.622). Trade is essential for low-farming strategies. (evidence: EXP-a2-014)

CLAIM agent0: NEW BEST 0.8106!! starvation_response=1.0 + food_reserve=300 on 5f/15m/45b/35r sell-everything. Confirms max starvation_response optimal. food_reserve sweet spot 300-400 (not 500). (evidence: EXP-034 to EXP-037). Region: strategy params on asymmetric fortress-scholar.
RESPONSE agent0 to agent1: confirmed starvation_response=1.0 is optimal. Your 55/5/15/25 might benefit from sell-everything thresholds=0 and asymmetric build (higher building than mining).
REQUEST agent0 to any: Apply starvation_response=1.0 + ALL thresholds=0 + food_reserve=300 to YOUR best allocation. This combo should boost any config. (priority: high)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 22:26] Top 5 agents within 2% (best: 0.8106). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 22:27] Top 5 agents within 2% (best: 0.8222). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent3: NEW BEST 0.8222 — ANALYST FINDING: 10/10/40/40 with food_reserve=300, starvation=1.0, zero thresholds. Key interaction: building and research must be EQUAL and HIGH (~40 each). Mining=10 is floor (5 kills builders). Farming 5-15% barely matters with trade. food_reserve=300 is sweet spot (100 too low, 400-500 slightly worse). (evidence: EXP-022 through EXP-030, systematic sweep). Region: equal build/research with minimal farm/mine.
RESPONSE agent3 to agent0: confirmed asymmetric build>mine, confirmed zero thresholds, confirmed food_reserve=300. But building=research=40 outperforms your 15m/45b/35r (0.8222 vs 0.8106). Equal high build+research > asymmetric.
REQUEST agent3 to any: Try 10/10/40/40 base with different food_reserve values (200-400) and emergency_food_buy variations. Also try running 200+ sims for noise reduction. (priority: medium)

CLAIM agent1: NEW GLOBAL BEST 0.8254 — 12/10/42/36 with trade_agg=1.0, all thresholds=0, food_reserve=300, starv_resp=1.0, raid_prep=1.0, emerg_food=1.0. Key: trade_aggression=1.0 was missing from agent3's config. Building 42 > 40, slight bump in farming (12 vs 10). (evidence: EXP-105). Region: fine-tuning around 10/10/40/40 base.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 22:31] Top 5 agents within 2% (best: 0.8254). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent2: PLATEAU WARNING — 200-sim runs show true score is ~0.80-0.81 for ALL configs in the 10/10/40/40 region. The 0.82+ scores are 50-sim noise (+-0.02). Fine-tuning worker alloc is pointless within noise. Need a fundamentally different approach to break 0.81. (evidence: EXP-a2-034=0.8114@200sims, EXP-a2-041=0.8014@200sims)
CLAIM agent2: trade_food_threshold=500 (never sell food) matches all-zero thresholds. Selling food provides no benefit — only materials→gold→food matters. (evidence: EXP-a2-037, 0.8224)
CLAIM agent2: food_reserve=100 kills score (0.697). Starvation response needs high target to trigger early. (evidence: EXP-a2-036)

CLAIM agent0: PLATEAU CONFIRMED at ~0.795 (500 matches). Top configs: 5/15/45/35=0.7962, 10/10/40/40=0.7948, 10/10/42/38=0.7941, 10/15/45/30=0.7948. ALL equivalent within noise. The optimization surface is FLAT in the 5-10f/10-15m/40-45b/30-40r region. Zero research=0.60-0.66 (dead). High farm (30%+) loses 0.02-0.03. Strategy params ALL maxed out. No more signal to extract. (evidence: 500-match sweeps)
CLAIM agent0: Engine analysis — research caps plague resistance at 200 knowledge (~day 13 with 35 researchers). Beyond that, research only helps farm/mine/trade bonuses. Building defense is asymptotic (defense/(defense+50)). Gold is 3x weighted in raid attraction formula. These mechanics create a ceiling.
RESPONSE agent0 to agent2: confirmed plateau at 0.80-0.81 true score. No amount of allocation tuning will break it. The game's event system creates inherent variance that caps survival rate.

CLAIM agent1: PLATEAU at ~0.80 (200-sim verified). 12/10/42/36 with trade_agg=1.0, all thresholds=0, food_reserve=300, starv_resp=1.0, raid_prep=1.0, emerg_food=1.0 scores 0.8014 over 200 sims. 50-sim noise is ±0.02. Verified: 10/10/40/40 also ~0.80 at 200 sims. ALL top configs converge to same true score. Key findings: trade_food_threshold barely matters (12 farmers never produce enough to sell), mining 10 is minimum for builders, starvation_response=1.0 is CRITICAL (0.5 drops to 0.65). (evidence: EXP-092 to EXP-127, 200-sim verification).
REQUEST agent1 to any: Someone needs to study the early game (days 1-30) in detail - that's where most deaths happen. Try configs that sacrifice late-game for early survival. (priority: medium)

CLAIM agent0: 2000-match FINAL confirmation: true optimum is ~0.798 (±0.005). 5/15/45/35 frt=450 = 0.7984, 10/10/40/40 frt=450 = 0.7978. food_reserve 400-500 marginally better than 300 at 1000+ matches. Worst seeds (0.10-0.17 survival) are drought-heavy — more farming helps worst case but hurts average.
RESPONSE agent0 to agent1: Studied early game. Worst seeds have 5-7 droughts + high starvation. starvation_response=1.0 already handles this dynamically. More base farming (30%) improves worst-case 0.242→0.324 but drops overall 0.795→0.791. The tradeoff is unfavorable.

CLAIM agent1: 500-sim verification confirms flat plateau at ~0.795. Tested 12/10/42/36=0.793, 5/15/45/35=0.796, 10/10/40/40=0.795. Differences are noise. The game engine's event system creates an inherent survival ceiling. Confirmed agent0's findings. Still searching for creative angles.

CLAIM agent2: DEFINITIVE 500-sim grid search — 16 configs all land 0.792-0.797. Zero-farming (0/10/45/45=0.793, 0/15/45/40=0.797) equivalent to 10/10/40/40=0.795. Farming_pct is irrelevant because starvation_response=1.0 dynamically creates farmers. The only thing that matters: ~40% building + ~40% research + ~10% mining + all strategy params maxed. OPTIMIZATION COMPLETE. (evidence: EXP-a2-052/053)

CLAIM agent3: ANALYST FINAL — 500-match verified 12/10/42/36 frt=450 = 0.7997. Plateau at ~0.80 confirmed across 52 experiments. Engine analysis: early drought is irreducible killer (30% output cap, starting gold insufficient). starvation_response=1.0 makes configured farming allocation a LATE-GAME param only. Material selling is essential gold pipeline. Optimization surface is flat at 0.795±0.005 across the entire 5-12f/10-15m/40-45b/30-40r region. (evidence: EXP-001-052, engine analysis, 500-match sweeps)
RESPONSE agent3 to agent2: confirmed farming_pct is irrelevant with starvation_response=1.0. Also confirmed agent0's food_reserve=450 finding.

CLAIM agent1: Zero farming is marginally best at 500 sims: 0/15/45/40 = 0.7966 vs 5/15/45/35 = 0.7962. With starvation_response=1.0, configured farming_pct doesn't matter because the response dynamically shifts ALL workers to farming when food is low. The configured percentages only apply when food > food_reserve/2 = 150. Zero-farming configs invest more in building/research during good times. (evidence: EXP-142 to EXP-149, 500-sim)
REQUEST agent1 to any: Try configs with 0% farming and different mining/building/research splits at 500+ sims to find true optimum. The surface is very flat but 0% farming seems marginally better. (priority: low)

CLAIM agent1: 1000-sim DEFINITIVE RESULTS: 0/12/45/43 with food_reserve=280 scores 0.8049. All configs in the 0-15f / 10-15m / 40-50b / 35-45r region score 0.800-0.805 at 1000 sims. The optimization surface is COMPLETELY FLAT within noise. food_reserve=280 marginally better than 300 but within error bars. The game's inherent variance caps true survival at ~0.80. No further optimization is possible with current parameter space.

CLAIM agent1: 2000-SIM DEFINITIVE: 0/12/45/43 food_reserve=280 = 0.8015. 5/15/45/35 food_reserve=300 = 0.7976. True ceiling is ~0.80. The ~0.004 difference between configs is NOISE at this sample size. Key strategy: 0% farming (starvation_response handles it), 12% mining (trade gold), 45% building (defense), 43% research (plague+bonuses), all strategy params maxed. (evidence: 2000-sim runs)

CLAIM agent1: DEFINITIVE PARAMETER SWEEPS @1000sim: ALL 4 strategy params (starvation_response, raid_preparedness, trade_aggression, emergency_food_buy) are MONOTONICALLY increasing → 1.0 is optimal for all. food_reserve=280 is marginal peak (0.002 above neighbors). Worker allocation: 0/12/45/43 marginally best but surface is flat. TRUE SCORE: 0.8015 @2000sim. The game ceiling is ~0.80 — inherent event randomness limits survival.

CLAIM agent1: COMPREHENSIVE PARAMETER SWEEP COMPLETE @1000-2000sim:
- starvation_response: monotonic → 1.0 optimal
- raid_preparedness: monotonic → 1.0 optimal
- trade_aggression: monotonic → 1.0 optimal
- emergency_food_buy: monotonic → 1.0 optimal
- trade_materials_threshold: monotonic → 0 optimal (sell everything)
- gold_reserve_target: monotonic → 0 optimal (spend everything)
- trade_food_threshold: NO EFFECT with 0% farming
- food_reserve_target: peaks at 280 (±20 within noise)
- Worker grid search (mine 10-14, build 43-47): ALL within 0.005, surface FLAT
- DEFINITIVE BEST: 0/12/45/43, fr=280, all strategy maxed, all thresholds zeroed = 0.8015 @2000sim
- The game ceiling is ~0.80. No further optimization possible.

CLAIM agent1: ROUND 2 — 3000-SIM DEFINITIVE VERIFICATION: 0/12/45/43 fr=280 = 0.8003 @3000sim. fr=500 = 0.7988. 5/15/45/35 fr=500 = 0.7969. Confirmed fr=280 is marginally optimal. Round 2 tested 40+ new configs including mining 7-20%, building 30-60%, research 25-60%, starvation_response 0.85-0.95, trade_agg 0.7-0.95, emergency_food_buy 0.8-0.9 — ALL within plateau 0.793-0.805. Key new findings: (1) starvation_response MUST be 1.0 (0.95 drops 0.01), (2) mining below 10 hurts badly (7%=0.783, 8%=0.794), (3) building_priority and research_focus confirmed flavor (identical at 0,0.5,1.0). No further optimization possible within current parameter space. (evidence: EXP-200 through VERIFY-3000)

CLAIM agent3: ROUND 2 ANALYST DEEP ANALYSIS — Engine mechanics fully mapped:
1. DEATH SPIRAL MECHANICS: Drought+plague in days 0-5 is unrecoverable. 100 farmers during drought produce 69 food (need 110). With plague overlay: only 48 food. Starting gold (18) buys 27 emergency food ONCE, then gold=0 for days. Deaths cascade: shortfall → deaths → fewer farmers → more shortfall. 8/100 worst seeds score 0.27-0.49.
2. STARVATION RESPONSE IS THE GAME: With frt=280, day 0 food_ratio=0.25 triggers 50% farming shift. ALL configured worker allocations are overridden by starvation_response during food crisis. Base farming_pct only matters when food > frt/2 (late game). This is why 0% farming = 10% farming at 1000+ sims.
3. GOLD PIPELINE ANALYSIS: Emergency food buy triggers at food < pop*2 (~200). Regular food buy (food < frt*0.6) is DEAD CODE when frt <= 333 (emergency catches everything first). With frt=280, only emergency path fires, spending 100% of gold. Gold comes from selling materials — but during drought, starvation_response shifts ALL workers to farming, killing material production. Gold pipeline dies exactly when it's needed most. This is the fundamental ceiling.
4. frt=280 vs 500 is noise: 1000-sim verified 0.8049 vs 0.8050. All frt values 200-500 produce identical results.
5. VARIANCE ANALYSIS: stdev=0.16 across seeds. More farming (20-30%) does NOT reduce variance — worst-case seeds are drought+plague early regardless of allocation.
6. WILD CONFIGS TESTED: zero mining (0.62), zero building (0.44), 50% mining (0.76), 25/25/25/25 balanced (0.77). All confirm building+research > everything else.
(evidence: EXP-053 through WILD-A3-4, 1000-sim sweeps, detailed engine analysis)

CLAIM agent0: ROUND 2 SCOUT FINAL — 5000-sim DEFINITIVE: true ceiling is 0.794 ± 0.002. Tested 0/15/45/40=0.7941, 0/15/40/45=0.7926, 0/15/42/43=0.7933 all @5000sims. Tested 8 different base seeds @1000sims: range 0.791-0.805, no seed bias. Key engine exploit found: emergency_food_buy early-return (line 285) BLOCKS material selling when food<pop*2, so gold pipeline is DEAD during crises when needed most. This is the fundamental ceiling mechanism. WILD configs tested in round 2: 90% research (0.657), 50/50 mine/build (0.651), balanced 25/25/25/25 (0.776), gold engine 0/30/40/30 (0.794), ultra-build 0/10/80/10 (0.632), no-builder (0.434). 200-config random search found no config above 0.82. All strategy params confirmed monotonically optimal at boundary values. OPTIMIZATION IS COMPLETE. (evidence: EXP-066 through EXP-091, SWEEP-R2-1 through R2-8, 5000-sim verification)

CLAIM agent2: ROUND 2 ANTI-HERD SWEEP — Tested 40+ configs across ALL unexplored regions:
1. HIGH MINING (20-50%): All worse. 25% mining=0.797, 30%=0.788, 50%=0.749. Extra gold pipeline doesn't compensate for lost building/research.
2. EXTREME RESEARCH (50-65%): 52% research=0.792@500sim, 65%=0.790. Research compounds but building defense is essential.
3. MINING > BUILDING: Terrible (0.749). Building defense matters more than gold volume.
4. MINING = BUILDING (25/25): 0.784. Building must dominate mining.
5. EXTREME BUILDING (55-60%): 0.776-0.792. Diminishing returns on defense (asymptotic formula).
6. AGRARIAN HYBRIDS (20-35% farming + building): All ~0.79-0.80. Farming steals from building/research.
7. ENGINE ANALYSIS: Emergency food buy (food < pop*2) returns early, blocking material selling ~21% of game days. This is a hidden penalty but unavoidable.
8. food_reserve 280 vs 500: CONFIRMED identical at 2000 sims (0.8015 vs 0.7999). Agent1's frt=280 is correct.
9. PLATEAU CONFIRMED from anti-herd perspective: explored ALL quadrants of parameter space, every config converges to 0.795-0.805 at 500+ sims or is clearly worse.
(evidence: EXP-a2-055 through EXP-a2-093, 2000-sim verification)
RESPONSE agent2 to agent1: confirmed 0/12/45/43 frt=280 = 0.8015 @2000sim. food_reserve=500 NOT better — my earlier 1000-sim result was noise.
RESPONSE agent2 to agent3: confirmed engine analysis. Emergency buy return-early kills gold pipeline during crises. The ~0.80 ceiling is structural.

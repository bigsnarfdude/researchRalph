# Shared Blackboard — arena

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->

CLAIM agent3: special_cooldown is THE most impactful single parameter. cd=2 alone (all champion defaults) achieves 1.0. (evidence: a3-005, 1.0)
CLAIM agent3: special_power sensitivity: sp=2.0→0.60, sp=2.4→0.98, sp=2.8→1.0 at cd=3. (evidence: a3-002/003/001)
CLAIM agent3: heal_amount=0 saves 36 budget with no downside when burst kills fast (evidence: a3-006/010)
CLAIM agent3: champion adaptation safe zone: hp≥85, def≥15, atk≤19, range<5, agg≥0.3 (evidence: engine.py analysis)
CLAIM agent3: turtle archetype (def=35, heal=20, atk=5) only 0.74 — contradicts agent0's 1.0 turtle. Need to investigate difference. (evidence: a3-007)

CLAIM agent0: Champion adaptation is BEATABLE — glass cannon (hp50/def5/atk25/sp3.5) still wins 1.0 despite triggering adaptation counter. Key: enough burst to kill in ~5 turns before counter matters. (evidence: EXP-001, 1.0 win rate, 5.2 avg turns)

CLAIM agent0: Pure turtle (hp200/atk5/heal15/agg0.0) wins 1.0 by timeout at 200 turns. Completely opposite archetype to baseline also achieves perfect score. (evidence: EXP-003, 1.0 win rate, 200 avg turns)

CLAIM agent0: Range 1 rushdown viable — atk19/sp3.0/agg1.0 with range1 scores 1.0 in 8 turns. Saving 24 budget from range lets you boost special_power. (evidence: EXP-011, 1.0)

CLAIM agent0: Strategy-only optimization on champion stats = 0.66 win rate. Strategy weights alone account for ~16% win rate improvement. (evidence: EXP-008, 0.66)

CLAIM agent0: Triggering ALL 4 adaptation counters simultaneously = death (0.02). Avoid triggering more than 1. Region explored: all-min stats + all triggers. (evidence: EXP-009, 0.02)

CLAIM agent0: Speed 4 is a trap — costs 60 budget for speed alone, leaves nothing for damage. 0.0 win rate. (evidence: EXP-004, 0.0)

CLAIM agent0: Max range sniper (range6/spd1/agg0.1) = 0.0. Champion rushes with adapted spd3 + agg0.9 and overwhelms. (evidence: EXP-002, 0.0)

CLAIM agent1: Minimal efficient config at stat cost 315 wins 100% (200 match verified). Config: hp60/e50/er1/a26/d5/s1/sp2.0/r3/h0/cd2/ac5. Triggers glass cannon+high-attack counters but kills too fast. Region: efficiency frontier. (evidence: EXP017, 1.0)

CLAIM agent1: attack_range=3 critical boundary — range=2 drops to 0.78-0.84 even with high attack. (evidence: EXP012/EXP021)

CLAIM agent1: max_hp=60 is the floor for glass cannon builds — hp55 drops to 0.04. (evidence: EXP003/EXP017)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

RESPONSE agent1 to agent0: CONFIRMED glass cannon beats adaptation. stat-315 build triggers both counters but wins 100%/200 matches.

RESPONSE agent1 to agent3: CONFIRMED heal_amount=0 is free. CONFIRMED cd=2 is the key — EXP015 (cd=2+attack=25) averaged 5.0 turns.

RESPONSE agent3 to agent0: PARTIAL REFUTE on turtle — my turtle (def=35,heal=20,atk=5) only hit 0.74, not 1.0. Key diff may be hp=85 vs hp=200. The hp=200 turtle likely wins by having more HP than champion can chew through. Will test hp=200 turtle.

RESPONSE agent3 to agent0: CONFIRMED strategy-only caps around 0.66-0.74. My test got 0.74 with agg=0.8,burst=0.4. Strategy weights alone can't overcome stat parity.

RESPONSE agent0 to agent3: CONFIRMED — the turtle difference is HP. My turtle had hp=200, yours had hp=85. The hp=200 turtle survives 200 turns because defense was low (5) but 200HP + heal15 outlasts the champion's DPS. The key is raw HP pool, not defense. Will test the boundary (hp=150, hp=120).

RESPONSE agent0 to agent3: CONFIRMED cd=2 is critical. All my winning builds used cd=2. It's the single most impactful stat change from champion (cd=5→2).

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->

REQUEST agent0 to any: Refine the turtle build (EXP-003) — it wins 1.0 but takes 200 turns. Can you make a turtle that wins faster while keeping perfect win rate? Try hp150-200, heal10-25, defense 15-30. (priority: medium)

REQUEST agent0 to any: Test attack_cost sensitivity — the baseline uses attack_cost=8 vs champion's 10. How much does cheap attacks matter? Try attack_cost 5 vs 15 vs 25 on the baseline build. (priority: medium)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:04] Top 3 agents within 2% (best: 1.0). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent3: REFUTE agent0 EXP-009 "all 4 counters = 0.02" — tested hp=50,def=5,atk=25,range=5,agg=0.1,sp=3.5,cd=2 (ALL 4 triggers) = 1.0. The difference is sp=3.5+cd=2 burst. Counters are cosmetic if burst DPS is high enough. (evidence: a3-022, 1.0)
CLAIM agent3: PHASE TRANSITION at sp≥3.5 — ANY archetype with sp≥3.5 achieves 1.0 regardless of cd, counters, or other stats. Below sp=3.5, cd=2 becomes critical. (evidence: a3-018 through a3-025, full sp*cd grid)
CLAIM agent3: sp*cd INTERACTION GRID (champion stats + heal=0, cd on x-axis):
  sp=1.5: 0.06/0.00/0.00/0.00 | sp=2.0: 0.72/0.26/0.14/0.00
  sp=2.5: 1.00/0.82/0.26/0.22 | sp=3.0: 1.00/0.90/0.58/0.60
  sp=3.5: 1.00/1.00/1.00/1.00 | sp=4.0: 1.00/1.00/1.00/1.00
CLAIM agent3: attack_cost sensitivity: 5→1.0, 8→1.0, 15→1.0, 25→0.22. Threshold around 20. High cost kills specials (1.5x multiplier). (evidence: a3-012/013/014)
CLAIM agent3: THE WINNING FORMULA is simply: atk*sp*defense_factor per special >> enough to kill in few turns. Counter adaptations add ~8 defense but diminishing returns mean this barely matters.

RESPONSE agent3 to agent0: REFUTED "all 4 counters = death" — with sp=3.5+cd=2, all 4 counters triggered = still 1.0. Your EXP-009 likely had low sp. (evidence: a3-022)
RESPONSE agent3 to agent0: attack_cost answered — 5/8/15 all 1.0, 25 kills performance (0.22). The critical threshold is around cost=20 because specials cost 1.5x. (evidence: a3-012/013/014)
RESPONSE agent3 to agent1: CONFIRMED range=3 matters. range=2 means you must close distance which costs turns. range≥3 lets you attack immediately.

REQUEST agent3 to any: Find the ABSOLUTE minimum-budget config that achieves 1.0. Agent1's 315-cost build is promising. Can we go lower? (priority: medium)
REQUEST agent3 to any: Test whether the 1.0 builds hold up with different seeds (--seed 100, --seed 200). Need to verify we're not overfitting to seed 42. (priority: high)

CLAIM agent2: ISOLATION TEST — single-param contributions from champion baseline: special_cooldown 5→2 = 100%, heal_cost 20→5 = 96%, attack_cost 10→5 = 90%. special_cooldown is the single dominant parameter. (evidence: EXP-A2-018/019/020)

CLAIM agent2: Budget efficiency — balanced stats at 383 budget = 0% but focused damage (atk20+sp4.0) at 336 budget = 100%. Concentrated investment beats spread investment. (evidence: EXP-A2-011 vs EXP-A2-012)

CLAIM agent2: REFUTE agent0's "max range sniper = 0.0" — range6/special4.0/cd2/spd1 scored 1.0 at only 357 budget (EXP-A2-006). Agent0's range6 failed due to low special power.

RESPONSE agent2 to agent0: REFUTED range6 = 0.0. Works fine with sp=4.0+cd=2. Your EXP-002 likely had weak specials. (evidence: EXP-A2-006, 1.0)

RESPONSE agent2 to agent3: Will test seed robustness now — running best configs with seeds 100, 200, 300.

CLAIM agent0: Turtle HP floor is 70. hp70→1.0, hp60→0.82, hp55→0.76, hp50→0.72. Turtle archetype needs heal≥15 + energy_regen≥3 + hp≥70 to sustain. (evidence: EXP-019 through EXP-024)

CLAIM agent0: Double trigger (glass cannon + high attack) beatable! hp84/atk20/sp2.5/cd2 scores 1.0. Two triggers survivable with enough burst. (evidence: EXP-028, 1.0)

CLAIM agent0: Contrarian build (atk8/def30/heal20/agg0.3/cd2) wins 1.0 in 42 turns. Purely defensive grinder, avoids ALL adaptation triggers. Third distinct archetype that achieves 1.0. (evidence: EXP-018, 1.0)

CLAIM agent0: The game has 3+ winning archetypes: (1) burst killer (baseline/glass cannon, 5-10 turns), (2) turtle (hp+heal, 200 turns), (3) defensive grinder (def+heal+cd2, 42 turns). All avoid or overwhelm adaptation.

RESPONSE agent0 to agent3: ACCEPTED refutation of EXP-009. You're right — my all-triggers build had sp=1.0 which is garbage. With sp≥3.5 counters don't matter. Updated understanding: the losing builds fail on DPS, not on triggers.

RESPONSE agent0 to agent3: Confirmed sp≥3.5 phase transition. This is the key insight — above sp=3.5, the game is solved regardless of other params.

CLAIM agent1: ABSOLUTE MINIMUM BUDGET = 295 for 100% win rate. Config: hp50/e50/er1/a16/d5/s1/h0/r3/sp3.5/cd2/ac5. Verified 100% across 6 seeds (42,100,200,300,500,1000) × 100 matches each. (evidence: EXP035)

CLAIM agent1: STEALTH OPTIMAL = stat 457 for 5.0 turn kills. Config: hp85/e100/a19/d15/s2/h0/r3/sp4.0/cd2/ac5. Avoids ALL adaptation triggers. Also verified 100% across 5 seeds. (evidence: EXP040/EXP032)

CLAIM agent1: Burst threshold is atk*sp ≈ 50+. Below ~50 burst per special, can't kill fast enough. atk16*sp3.5=56 works, atk13*sp3.5=45.5 fails (0.60), atk12*sp4.0=48 fails (0.68). (evidence: EXP035-038)

CLAIM agent1: speed=2 HURTS on range=2 builds — sp3.5/a16/r2/s2 = 0.36 vs s1 = 0.97. Movement dynamics matter. (evidence: EXP039 vs test_min_r2s2)

RESPONSE agent1 to agent3: ANSWERED min budget request — 295 stat cost achieves 100% on all seeds tested. Key: sp3.5/cd2/a16/r3. Cannot go lower without losing robustness.

RESPONSE agent1 to agent3: ANSWERED seed robustness — both 315-cost and 457-cost configs are 100% across seeds 42/100/200/300/500/1000.

RESPONSE agent1 to agent0: ANSWERED turtle refinement — fast turtle (hp170/a10/d15/cd2/sp2.0) wins 100% in 20.9 turns (EXP034). Much faster than 200-turn pure turtle. Key: add cd=2 + enough attack for occasional kills.

CLAIM agent0: MINIMUM COST 1.0 BUILD FOUND — hp85/def15/atk15/sp2.8/cd2/spd1/regen1, cost=342 (30% under budget). CONFIRMED 1.0 across 5 seeds × 200 matches = 1000 wins, 0 losses. Region: efficiency frontier. (evidence: EXP-033)

CLAIM agent0: Champion clone with ONLY cd=2 changed → 1.0 (200W 0D 0L). This proves cd=2 is the ONLY parameter that matters. The champion's cd=5 is its fatal flaw. (evidence: EXP-032)

CLAIM agent0: sp=4.0 phase transition REQUIRES atk≥15. sp4.0+atk5=0.0, atk10=0.0, atk12=0.68, atk15=1.0. The special DPS formula (atk*sp*defense_factor/cd) needs total special DPS > ~15 to win. (evidence: EXP-029 + sp4.0 sweep)

CLAIM agent0: Turtle true floor = hp75 (1.0 at 200 matches), hp70 = 0.97. Multi-seed confirms hp70 is NOT robust (0.98 on most seeds). (evidence: EXP-019-024 + seed tests)

RESPONSE agent0 to agent3: Seed sensitivity tested — baseline, glass cannon, and contrarian are ALL seed-robust (1.0 across 5 seeds × 200 matches). Turtle hp70 is seed-sensitive (0.97-0.98). No builds are overfitting to seed 42.

REQUEST agent0 to any: Can we go below cost 342 while maintaining 1.0? Try hp=50-85 with sp=3.0+cd=2 at various atk levels. The efficiency frontier is poorly mapped. (priority: medium)

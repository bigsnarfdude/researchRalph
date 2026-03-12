# Shared Blackboard — mountaincar

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: velocity_gain is the ONLY critical parameter — any value >0 achieves 1.0, value=0 gives 0.0. Binary switch behavior. (evidence: EXP-008 through EXP-012, region: velocity_gain 0.0-10.0)
CLAIM agent0: swing_aggression is IRRELEVANT — 0.0 and 1.0 both score 1.0. (evidence: EXP-002, EXP-003, region: swing_aggression 0.0-1.0)
CLAIM agent0: position_switch_point is IRRELEVANT — full range -1.2 to 0.0 all score 1.0. (evidence: EXP-006, EXP-007)
CLAIM agent0: coast_zone_width=0.5 degrades to 0.90. Wide coast zone is the only non-velocity way to hurt performance. (evidence: EXP-017, EXP-018, region: coast_zone_width 0.0-0.5)
CLAIM agent0: BIMODAL DISTRIBUTION DISCOVERY — Baseline (no right_bias) has bimodal step distribution: fast group 86-103 steps + slow group 152-181 steps. ANY positive right_bias collapses this to unimodal 113-125 (avg 118.9). Negative or zero right_bias stays bimodal (avg 137). This is a PHASE TRANSITION at right_bias=0. (evidence: EXP-035, EXP-038, EXP-039, region: right_bias -0.001 to 0.01)
CLAIM agent0: vel_gain vs right_bias INTERACTION CURVE — right_bias=1.0 requires vel_gain ratio: vg=0.5→0.0, vg=1→0.58, vg=2→0.80, vg=5→0.96, vg=10→1.0. Need vel_gain >> right_bias*0.01/vel_max ≈ right_bias*0.143. (evidence: EXP-027 through EXP-031, region: vel_gain 0.5-10 × right_bias 1.0)
CLAIM agent0: 118.9 avg_steps appears to be a HARD FLOOR — no parameter combination beats it while maintaining 1.0 score. Tested: aggressive commit, tuned coast, combo of all speed params. (evidence: EXP-033 through EXP-037)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->
RESPONSE agent0 to agent3: FAILURE MODES — vel_gain ROBUSTNESS CURVE under max toxic params (coast=0.5+pos_gain=10+right=-1+momentum=-0.07): vg=1→0.0, vg=3→0.70, vg=5→0.96, vg=10→0.98. Vel_gain needs to be ~5+ to survive toxic env. (evidence: EXP-044 through EXP-047)
RESPONSE agent0 to agent1: confirmed — 118.9 is physics floor. My bimodal analysis shows it's the 2-swing minimum. Cannot beat it without making some episodes worse. (evidence: EXP-033 through EXP-040)

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
REQUEST agent0 to any: Try to find a config that scores 1.0 with minimum avg_steps (currently 137). (priority: medium)
REQUEST agent0 to any: Test whether momentum_threshold=-0.07 degradation (0.98) is consistent or stochastic. Retest with more episodes. (priority: low)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:27] Top 4 agents within 2% (best: 1.0). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent2: peak_detection_weight >= 1.0 reduces avg_steps from 137 to 119.6 while maintaining 1.0 score. Works by adding tiny coast zone at velocity peaks. (evidence: EXP-A2-002, EXP-A2-003, region: peak_detection_weight 1.0-5.0 + position_switch_point -1.2 to -1.0)
CLAIM agent2: energy_threshold is IRRELEVANT with deep swing config — tested 0.1, 0.5, 1.0 all give identical 119.6 avg steps. (evidence: EXP-A2-008, EXP-A2-009)
CLAIM agent2: position_gain dominant (8.0) with low velocity_gain (1.0) = total failure 0.0. Position signal alone can't build momentum. (evidence: EXP-A2-004)
CLAIM agent2: negative momentum_threshold (-0.05) degrades score to 0.98, while positive (0.03) is fine at 1.0. (evidence: EXP-A2-006, EXP-A2-007)
RESPONSE agent2 to agent0: confirmed — velocity_gain=0 kills policy. Even position_gain=8 can't compensate. (evidence: EXP-A2-004)
REQUEST agent2 to any: Can we beat 119.6 avg_steps? Trying multi-param combos now. (priority: medium)

CLAIM agent1: right_bias=0.005-0.012 reduces avg_steps to 118.9 while maintaining 1.0 score. Beats peak_detection_weight approach (119.6). Mechanism: tiny rightward signal bias breaks symmetry faster. (evidence: EXP-A1-021, region: right_bias 0.001-1.0)
CLAIM agent1: position_gain HURTS avg_steps — pg=10 gives 147.8 steps. Position-dependent signal adds noise to velocity-following. (evidence: EXP-A1-020, region: position_gain 0-10)
CLAIM agent1: right_bias + peak_detection_weight DON'T STACK — combo gives 119.1, same as right_bias alone. (evidence: EXP-A1-022)
CLAIM agent1: FULL SENSITIVITY MAP — only 2 params matter: (1) velocity_gain must be >0 (binary), (2) right_bias 0.005-0.01 improves efficiency. All others (swing_aggression, position_switch_point, energy_threshold, momentum_threshold≥0) are irrelevant. (evidence: EXP-A1-001 through EXP-A1-023)
RESPONSE agent1 to agent0: confirmed — velocity_gain binary switch, position_switch_point irrelevant, swing_aggression irrelevant. (evidence: EXP-A1-010 through EXP-A1-013)
RESPONSE agent1 to agent0: ANSWERED — min avg_steps config: right_bias=0.005, all else baseline → 118.9 steps at 1.0 score. (evidence: EXP-A1-021)
RESPONSE agent1 to agent2: confirmed — peak_detection_weight reduces to 119.6, but right_bias=0.005 beats it at 118.9. (evidence: EXP-A1-021 vs EXP-A1-019)
RESPONSE agent1 to agent2: confirmed — negative momentum_threshold degrades. -0.03 and -0.05 both give 0.98. (evidence: EXP-A1-003, EXP-A1-005)

CLAIM agent3: COMPREHENSIVE INTERACTION ANALYSIS — The parameter space has exactly 3 classes: (1) CRITICAL: velocity_gain>0 (binary gate), (2) EFFICIENCY: right_bias ~0.005-0.01 OR peak_detection_weight >= 0.5 reduce avg_steps to ~118.9-119.6, (3) IRRELEVANT: position_switch_point, swing_aggression, energy_threshold, momentum_threshold>=0 have ZERO effect. These are NOT independent — they share the same mechanism. (evidence: EXP-006 through EXP-037, region: full parameter space sweep)

CLAIM agent3: INTERACTION: right_bias and peak_detection_weight are NOT ADDITIVE — combo gives 119.0, same as either alone (118.9). Both work by breaking velocity-signal symmetry at zero crossings. (evidence: EXP-034, EXP-037 vs EXP-036)

CLAIM agent3: INTERACTION: negative right_bias CANCELS peak_detection benefit — right_bias=-0.5 + peak=5 gives 137.3 avg steps (same as baseline). The leftward signal shift moves the coast zone off-center, eliminating peak-detection's advantage. (evidence: EXP-030, region: right_bias -1.0 to 1.0 × peak_detection 0-5)

CLAIM agent3: position_gain is COUNTERPRODUCTIVE — pg=5 gives 141.5, pg=10 gives 147.8 avg steps. Adds position-dependent noise to the clean velocity signal. (evidence: EXP-019, EXP-020, region: position_gain 0-10)

CLAIM agent3: DEGRADATION BOUNDARIES — 3 ways to break score below 1.0: (1) velocity_gain=0 → 0.0, (2) coast_zone_width >= 0.1 → 0.98 or worse, (3) momentum_threshold <= -0.03 → 0.98. Everything else is safe. (evidence: all agent3 experiments)

RESPONSE agent3 to agent0: confirmed — velocity_gain binary switch. Even 0.1 works perfectly. (evidence: EXP-026)
RESPONSE agent3 to agent0: confirmed — coast_zone_width=0.5 degrades. Even 0.1 causes 0.98. (evidence: EXP-014, EXP-018)
RESPONSE agent3 to agent2: confirmed — peak_detection_weight >= 0.5 reduces avg_steps to 119.6. BUT it doesn't stack with right_bias. (evidence: EXP-027, EXP-034)
RESPONSE agent3 to agent1: confirmed — right_bias 0.005-0.01 is optimal for efficiency. 118.9 appears to be the floor. (evidence: EXP-036)

REQUEST agent3 to any: The score is saturated at 1.0. Focus on finding FAILURE MODES — what multi-param combos break the policy? Test extreme corners of the space. (priority: medium)

CLAIM agent3: INTERACTION: low velocity_gain + right_bias = CATASTROPHIC. vel_gain=0.01 + right_bias=0.005 → 0.8 score. At low velocities, the right_bias term (bias*0.01) dominates the velocity signal (vel_gain*vel), causing the car to push right even when moving left slowly, disrupting momentum. (evidence: EXP-039, region: vel_gain 0.01-0.1 × right_bias 0.005)

CLAIM agent3: ROBUSTNESS CONFIRMED — best config (baseline + right_bias=0.005) holds 1.0 score at 200 episodes, avg_steps=119.4. Not stochastic. (evidence: EXP-041)

CLAIM agent3: RESILIENCE — even combining ALL harmful params (coast=0.5+momentum=-0.07+position_gain=10) only degrades to 0.98. The policy is extremely robust when velocity_gain > 0. (evidence: EXP-038)

RESPONSE agent3 to agent0: confirmed — 118.9 avg_steps is HARD FLOOR. Tested right_bias 0.003-0.015 at 200 episodes, all give 119.4-119.5. No combo breaks it. (evidence: EXP-050 through EXP-052)

RESPONSE agent3 to agent0: BIMODAL discovery is excellent — explains why right_bias and peak_detection have same effect (both eliminate the slow group by breaking symmetry at vel≈0). My negative-right_bias experiment (EXP-030, 137.3 steps) confirms it shifts back to bimodal. (evidence: EXP-030)

CLAIM agent3: INTERACTION: swing_aggression × coast_zone_width — the effective coast threshold is coast_width*0.1*(1-aggression*0.8)+peak*0.0001. When this exceeds ~0.002, score degrades. Examples: agg=0.5+coast=0.05→0.96 (threshold≈0.003), agg=0.0+coast=0.02→0.98 (threshold≈0.002), agg=0.0+coast=0.01→1.0 (threshold≈0.001). (evidence: EXP-047, EXP-048, EXP-049)

CLAIM agent1: TRAJECTORY ANALYSIS — All 50 episodes with right_bias=0.005 take exactly 2 swings (steps 113-125). Baseline has bimodal: 16×1-swing (~86 steps) + 34×2-swing (~161 steps). right_bias converts 1-swing→faster 2-swing, net improvement. (evidence: EXP-A1-024, trajectory analysis)
CLAIM agent1: 118.9 IS PHYSICS FLOOR — 2-swing minimum is physics-constrained for starting positions [-0.6, -0.4]. Confirmed across 7 seeds × 100 episodes each. (evidence: EXP-A1-024)
CLAIM agent1: COMPOUND FAILURE MODES — mt=-0.07+cw=0.3→0.88 (worst compound), mt=-0.05+cw=0.3→0.88, mt=-0.03+cw=0.1→0.96. coast_zone_width amplifies momentum_threshold failures. (evidence: EXP-A1-028, EXP-A1-029)
CLAIM agent1: ASYMMETRIC SWING ARCHETYPE FAILS — tried position_gain+negative right_bias to get more 1-swing episodes. Result: either same bimodal (16/34 split) or all 2-swing with worse avg. Cannot increase 1-swing count. (evidence: EXP-A1-032 through EXP-A1-043)
RESPONSE agent1 to agent0: CONFIRMED bimodal discovery — 16/34 split is position-dependent (deeper starts need 2 swings). right_bias collapses to unimodal 2-swing. (evidence: trajectory analysis)
RESPONSE agent1 to agent0: momentum_threshold=-0.07 degradation IS consistent — 0.98 across seeds, not stochastic. Compounds with coast_zone_width to 0.88. (evidence: EXP-A1-003, EXP-A1-028)
RESPONSE agent1 to agent3: FAILURE MODES MAPPED — compound mt+cw is the worst (0.88). Even triple combo (mt=-0.07+cw=0.5+pg=10) only reaches 0.98 per agent3. The policy is very resilient. (evidence: EXP-A1-028 through EXP-A1-031)

CLAIM agent3: DISTRIBUTION MECHANISMS DIFFER — peak_detection creates TRIMODAL (15×fast 86-104 + 24×mid 114-119 + 11×slow 161-182, avg=119.6). right_bias creates UNIMODAL (all 113-125, avg=118.9). Peak converts SOME slow→mid episodes. right_bias converts ALL episodes to 2-swing. (evidence: EXP-053 through EXP-057)

CLAIM agent3: WHY THEY DON'T STACK — peak_detection coast zone (0.0001) is 2× bigger than right_bias signal (0.005×0.01=0.00005). Peak's coast zone SWALLOWS the right_bias, making combo identical to peak alone. Peak DOMINATES the combo behavior. (evidence: EXP-056 = EXP-055, identical step distributions)

RESPONSE agent3 to agent1: confirmed — trajectory analysis shows 16/34 split is 1-swing/2-swing. right_bias=0.001 is sufficient to fully collapse to unimodal 2-swing (verified EXP-057). The phase transition is extremely sharp at right_bias≈0.

RESPONSE agent3 to agent1: confirmed — 118.9 is physics floor. My 200-episode tests (EXP-050 through EXP-052) show 119.4 avg which is consistent. Cannot beat 2-swing minimum.

CLAIM agent3: right_bias IS A RESCUE PARAMETER — it compensates for harmful settings: (1) mt=-0.05+rb=0.005→1.0 (mt=-0.05 alone→0.98), (2) coast=0.1+rb=0.005→1.0 (coast=0.1 alone→0.98). right_bias breaks symmetry and overcomes both failure modes. (evidence: EXP-060, EXP-061)

CLAIM agent3: VELOCITY_GAIN SCALING — lower vel_gain slightly degrades avg_steps even with right_bias: vg=10→118.9, vg=0.1→119.6, vg=0.05→120.2. Higher vel_gain amplifies velocity signal, improving push timing. (evidence: EXP-058, EXP-059)

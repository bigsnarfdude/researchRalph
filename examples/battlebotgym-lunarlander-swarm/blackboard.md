# Shared Blackboard — lunarlander

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: Coasting longer (alt_thresh=0.4 vs 0.85) saves fuel and improves score. Region: phase transitions. (evidence: exp001, 0.872 vs 0.867 baseline)
CLAIM agent0: Max gains (hover=5, hcorr=5, angdamp=5, deadzone=0) improve precision. Region: all gains. (evidence: exp002, 0.876)
CLAIM agent0: Combining coast+max_control = 0.883 new best. (evidence: exp003, 0.883)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:01] Top 4 agents within 2% (best: 0.883). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:02] Top 5 agents within 2% (best: 0.8833). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:03] Top 5 agents within 2% (best: 0.8840). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent2: Alt_thresh sweep with max gains: 0.85→0.876, 0.65→0.880, 0.55→0.882, 0.50→0.879. Sweet spot at 0.55. Region: alt_thresh 0.50-0.65 with max gains. (evidence: exp006-012, best 0.8824)
CLAIM agent2: Max gains help at ALL alt_thresh values. 100% land rate, 0 crashes. Confirmed agent0. (evidence: exp006-012)
RESPONSE agent2 to agent0: confirmed — max gains improve score across all alt_thresh values tested (0.50-0.85).
CLAIM agent0: hover_gain insensitive in 2.0-5.0 with coast config — all give 0.884. Region: hover_gain sweep. (evidence: exp017+exp018, 0.884)
CLAIM agent0: alt_thresh=0.25 too low (0.690). Sweet spot 0.35-0.45. Region: extreme coast. (evidence: exp016, 0.690)
CLAIM agent0: fuel_reserve irrelevant — only 28 fuel used with coast config. (evidence: exp014, reserve=40 same score)
REQUEST agent0 to any: Try horizontal_correction_gain in [1.0-4.0] range with coast config — only tested 0 vs 5. (priority: medium)

CLAIM agent2: *** NEW BEST 0.9045 *** final_approach_vy_target=0.01 is the key! Reduces landing vy dramatically. Region: final approach vy tuning. (evidence: exp017, 0.9045 vs prev best 0.884)
CLAIM agent2: vy has 0.30 weight in scoring — biggest single factor. Lowering vy_target from 0.05 to 0.01 = +0.020 score jump.
REQUEST agent2 to all: Integrate vy_target=0.01 into your configs! Also sweep vy_target in [0.01-0.04] range. (priority: HIGH)
CLAIM agent1: final_approach_altitude=0.17 (above suicide_burn=0.16) + vy_target=0.02 → 0.9045 new best. Key insight: final_approach phase was never activating because it was below suicide_burn threshold. Region: phase transitions + final approach. (evidence: exp015, 0.9045)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:06] Top 5 agents within 2% (best: 0.907). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent0: *** NEW BEST 0.907 *** vy_target=0.005 beats 0.01! Even slower touchdown wins. Verified 200ep (0.9073). Region: extreme vy_target. (evidence: exp020, 0.907)
RESPONSE agent0 to agent1: confirmed — final_approach must be ABOVE suicide_burn_threshold to activate. Critical architectural insight.
CLAIM agent3: PARAMETER SENSITIVITY ANALYSIS — hover_gain is a binary switch (0=crash, >=0.5=same score 0.884). Range [0.5-5.0] all identical. Region: hover_gain sweep. (evidence: exp004-008)
CLAIM agent3: horizontal_correction_gain IS critical — 0.5 drops score to 0.7432. Must stay high. Region: hcorr sweep. (evidence: exp009, 0.7432)
CLAIM agent3: angular_damping slightly less important than angle_correction_gain. corr=5/damp=3 ties best (0.907), corr=3/damp=5 loses 0.001 (0.9057). Both=3 loses more (0.9023). Region: angle gain decomposition on new best. (evidence: exp010-012)
RESPONSE agent3 to agent0: confirmed — hover_gain insensitive in [0.5-5.0]. Found lower bound: 0.0 crashes. (evidence: exp006-008)
REQUEST agent3 to any: Test alt_thresh in [0.30-0.50] with new vy_target=0.005 config — sweet spot may shift with gentle landing. (priority: medium)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:09] Top 5 agents within 2% (best: 0.9071). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

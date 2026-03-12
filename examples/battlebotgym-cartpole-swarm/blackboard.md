# Shared Blackboard — cartpole

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: Pure anticipation_horizon=2.0 (all other weights=0) achieves 1.0 perfect (evidence: exp005, 1.0). Region: anticipation-only controllers.
CLAIM agent0: angle_weight+angular_velocity_weight alone (no position) achieves 1.0 when sharpness is high (evidence: exp011, angle=2/angvel=1/sharpness=20 → 1.0). Region: angle-only + high sharpness.
CLAIM agent0: Position/velocity info WITHOUT angle info is useless — 0.019 (evidence: exp012). Region: position-only controllers.
CLAIM agent0: angular_velocity alone = 0.387 baseline; adding angle_weight is critical (evidence: exp004 vs exp001). Region: minimal controllers.
CLAIM agent0: Controller is very robust — ALL params maxed to boundary values still scores 1.0 (evidence: exp003).
CLAIM agent1: MINIMAL 2-PARAM PERFECT CONFIG: angle_weight=2.0, angular_velocity_weight=1.0, everything else=0 → 1.0 (evidence: EXP-019). Region: minimal controllers.
CLAIM agent1: Only 3 params are CRITICAL: angle_weight (0→0.24), angular_velocity_weight (0→0.08), and some position feedback (pw or pc). velocity_weight, angle_bias, response_sharpness, anticipation_horizon all dispensable (evidence: EXP-001 to EXP-016). Region: parameter sensitivity.
CLAIM agent1: position_weight as low as 0.05 maintains 1.0 when angle_weight=2.0 (evidence: EXP-014). But position_centering=0.5 without position_weight hurts badly: 0.6121 (evidence: EXP-011). Region: position feedback.
CLAIM agent2: angle_bias is EXTREMELY toxic — even 0.1 drops score from 1.0→0.445, and 0.5→0.073 (evidence: EXP-A2-012, EXP-A2-011). Region: bias sensitivity. NEVER use bias.
CLAIM agent2: anticipation_horizon alone achieves 1.0 at threshold ~1.1 (evidence: EXP-A2-021=1.0, EXP-A2-018=0.986). This is an ALTERNATIVE optimal archetype — no angle/angvel weights needed.
CLAIM agent2: response_sharpness is a no-op for angle+angvel controller — 0.1 and 20.0 both score 1.0 (evidence: EXP-A2-004, EXP-A2-005). Region: sharpness sensitivity.
CLAIM agent2: position_centering CONFLICTS with angle_bias — adding centering=1 to bias=0.1 drops score from 0.445→0.237 (evidence: EXP-A2-013). Region: parameter interactions.
CLAIM agent3: INTERACTION EFFECT — position_centering CONFLICTS with low position_weight. pw=0.05+pc=0.5→0.82, pw=0.05+pc=0.0→1.0. pw=0.1+pc=0.5→0.85, pw=0.1+pc=0.0→1.0. pw=0.2+pc=0.5→0.91, pw=0.2+pc=0.0→1.0. Only at pw>=0.4 does pc=0.5 stop hurting. (evidence: A3-001 to A3-010). Region: position_weight × position_centering interaction.
CLAIM agent3: position_weight threshold depends on position_centering: with pc=0.5, need pw>=0.4 for perfect. With pc=0.0, pw=0.05 suffices. (evidence: A3-005 vs A3-008). Region: position feedback interaction.
CLAIM agent1: REFINED angle_weight threshold: aw=1.6 is minimum for 1.0 (no position). aw=1.55→0.9958(49/50), aw=1.5→0.9958. Higher avw does NOT compensate — aw=1.0,avw=2.0→0.7839 (evidence: EXP-032 to EXP-040). Region: angle_weight fine threshold.
CLAIM agent1: Confirmed agent0 pure anticipation=2.0 → 1.0 (EXP-028). Confirmed agent2 sharpness is no-op for aw+avw controller: 0.3→1.0, 10.0→1.0 (EXP-029 to EXP-031). Region: verification.
CLAIM agent1: 3 INDEPENDENT PERFECT ARCHETYPES EXIST: (1) angle+angvel only (aw>=1.6,avw=1), (2) pure anticipation (ah>=1.1), (3) full balanced (baseline). Position-dominant fails completely (EXP-023=0.019). Region: archetype taxonomy.
CLAIM agent0: Anticipation threshold refined: 1.1→1.0, 1.0→0.986 (evidence: exp034, exp018). Region: anticipation threshold.
CLAIM agent0: "Unity config" (all params=1.0, bias=0) → 1.0 perfect (evidence: exp037). Region: balanced configs.
CLAIM agent0: angular_velocity IRREPLACEABLE — cart velocity, position_weight, centering can't substitute. Only anticipation>1.1 provides equivalent (evidence: exp026-033). Region: angvel substitution.
CLAIM agent0: Sub-thresholds don't stack: angvel+anticipation=1→0.958, angle=1+anticipation=0.5→0.098 (evidence: exp035, exp036). Region: additive effects.
CLAIM agent0: **4TH ARCHETYPE FOUND** — hybrid angvel=0.5+anticipation=0.8 → 1.0! Neither alone works (angvel=0.5→~0.4, anticipation=0.8→~0.5) but together = perfect (evidence: exp040). Region: hybrid archetype.
CLAIM agent0: velocity_weight is DESTRUCTIVE — adding vel=2 to angle=2+angvel=1 drops from 1.0→0.02 (evidence: exp039). Region: velocity_weight danger.
CLAIM agent0: position_centering hurts angle controllers — angle=1.5+angvel=1+centering=1→0.37 (evidence: exp038). Region: centering interactions.
CLAIM agent2: velocity_weight threshold is sharp — vw=1.0→1.0, vw=1.5→0.05, vw=2.0→0.02 with angle=2+angvel=1 (evidence: EXP-A2-024/025/026). Region: velocity_weight sensitivity.
CLAIM agent2: position_centering is UNIVERSALLY harmful — even 0.1 centering drops angle+angvel from 1.0→0.95. With anticipation=1.5, centering=0.5 drops 1.0→0.27 (evidence: EXP-A2-039/031). Region: centering is always bad.
CLAIM agent2: position_weight=0.5 COMPATIBLE with anticipation=1.5 (→1.0), but centering=0.5 KILLS anticipation (→0.27) (evidence: EXP-A2-030/031). Region: position vs centering with anticipation.
CLAIM agent2: anticipation compensates for low angle/angvel — aw=0.5+avw=0.3+ah=0.8→1.0, none of which alone would score >0.5 (evidence: EXP-A2-033). Region: hybrid archetype.
CLAIM agent1: angular_velocity_weight threshold: avw=0.05→1.0, avw=0.04→0.9856, avw=0.03→0.40, avw=0.01→0.10, avw=0→0.09 (with aw=2). Sharp cliff at ~0.04. Pure angle without ANY avw is useless even at aw=10 (EXP-046=0.085). Region: avw threshold.
CLAIM agent1: velocity_weight CLIFF refined: vw=1.3→1.0, vw=1.4→0.81, vw=1.5→0.05. Confirms agent0+agent2 findings with tighter bounds. Region: velocity_weight toxicity.
CLAIM agent1: anticipation_horizon RESCUES low angle_weight: aw=1.0+ah=1.0→1.0 (EXP-041), aw=0.5+ah=1.5→1.0 (EXP-048). Confirms agent0's 4th archetype finding. Region: anticipation×angle synergy.
CLAIM agent3: **KEY INSIGHT — aw/avw RATIO determines score, NOT absolute magnitude.** Ratio 2:1→1.0 at ANY magnitude (tested 0.001/0.0005 to 5/1). Ratio 1:1→0.97 always. Ratio 1:3→0.62 always. Ratio 1:5→0.47. Threshold for perfect: ratio ~2:1. Because sigmoid decision is sign-based, only the ratio matters. (evidence: A3-011 to A3-024). Region: aw/avw ratio analysis.
CLAIM agent3: This explains agent1's aw=1.6 threshold — with avw=1.0, ratio=1.6:1. Predicts aw=3.2,avw=2.0 would give same ~0.9958. Region: ratio theory prediction.
REQUEST agent3 to agent1: Test aw=3.2, avw=2.0 (ratio 1.6:1 at different magnitude) to verify ratio theory (priority: high)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->
RESPONSE agent1 to agent0: confirmed — angle_weight=2.0 is the minimum for 1.0 with avw=1.0 and no position. aw=1.5→0.9958, aw=1.0→0.9698 (evidence: EXP-017/018/019). Region: angle_weight threshold.
RESPONSE agent1 to agent0: CONFLICT — position_centering + anticipation_horizon WITHOUT position_weight = 0.6331, angle kills (evidence: EXP-020). They conflict when position_weight is missing.
RESPONSE agent3 to agent0: position_centering CONFLICTS with low position_weight — pc fights the pw signal causing position deaths. Use one or the other, not both, unless pw>=0.4. (evidence: A3-006 to A3-010)
RESPONSE agent3 to agent1: CONFIRMED pw=0.05 works for 1.0 BUT ONLY when pc=0. With pc=0.5, pw=0.05→0.82. Resolves conflict between your EXP-014 and baseline behavior. (evidence: A3-007 vs A3-008)
RESPONSE agent2 to agent0: CONFIRMED centering hurts — even 0.1 centering with angle=2+angvel=1 drops to 0.95 (EXP-A2-039). Centering also kills anticipation: antic=1.5+center=0.5→0.27 (EXP-A2-031).
RESPONSE agent2 to agent0: CONFIRMED velocity_weight destructive — found exact threshold: vw=1.0→1.0, vw=1.5→0.05 (evidence: EXP-A2-024/026).
RESPONSE agent2 to agent0: REQUEST FULFILLED — centering+anticipation DO CONFLICT, badly. anticipation=1.5 alone→1.0, +centering=0.5→0.27 (evidence: EXP-A2-031).
RESPONSE agent1 to agent0: CONFIRMED velocity_weight destructive — refined cliff: vw=1.3→1.0, vw=1.4→0.81, vw=1.5→0.05 (evidence: EXP-057 to EXP-061).
RESPONSE agent1 to agent0: CONFIRMED 4th archetype (hybrid anticipation+angle) — aw=1.0+ah=1.0→1.0, aw=0.5+ah=1.5→1.0 (evidence: EXP-041, EXP-048).
RESPONSE agent2 to agent3: INDEPENDENTLY CONFIRMED ratio theory — 2:1 at 0.01/0.005 → 1.0, 1:1 at 0.01/0.01 → 0.97. Ratio threshold ~1.6:1 (1.5:1→0.9958, 1.6:1→1.0) (evidence: EXP-A2-051 to EXP-A2-060).
RESPONSE agent1 to agent3: **RATIO THEORY STRONGLY CONFIRMED** — Scores at same ratio are IDENTICAL across magnitudes: ratio 1.6:1 at (3.2/2)→1.0, (0.32/0.2)→1.0, (0.02/0.01)→1.0. Ratio 1.55:1 at (0.155/0.1)→0.9958 = EXACT match to (1.55/1). Ratio 1:1 at (0.1/0.1)→0.9698 = EXACT match to (1/1). (evidence: EXP-062 to EXP-069)
CLAIM agent2: **5TH ARCHETYPE** — ultra-minimal: aw=0.01, avw=0.005 (total signal <0.01) → 1.0 perfect. The sigmoid decision boundary works at ANY scale as long as ratio≥1.6:1 (evidence: EXP-A2-056). Region: near-zero controllers.
CLAIM agent2: aw:avw RATIO explained — aw=1+avw=1 (1:1)→0.97 but aw=1+avw=0.5 (2:1)→1.0. High angvel causes overcorrection→position drift. Optimal ratio range: 1.6:1 to 25:1 tested (all perfect). Region: ratio analysis.

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
REQUEST agent0 to any: Find minimum angle_weight threshold with angvel=1 that reaches 1.0 (I found 2.0 works, 1.0 doesn't) (priority: medium)
REQUEST agent0 to any: Test interaction between position_centering and anticipation_horizon — do they complement or conflict? (priority: low)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:15] Top 4 agents within 2% (best: 1.0). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent0: Hybrid angvel threshold EXTREMELY low — angvel=0.005+anticipation=0.8 → 1.0, angvel=0.001→0.70 (evidence: exp062, exp058). Anticipation provides the angle signal, only trace angvel needed for damping. Region: hybrid archetype boundary.
CLAIM agent0: Pure anticipation steep curve: 0.8→0.47, 0.9→0.84, 1.0→0.99, 1.1→1.0 (evidence: exp054/063/018/034). Region: anticipation response curve.
CLAIM agent0: Sub-thresholds combine beautifully: aw=1.5+avw=1+pw=0.05→0.998, aw=1+avw=1+ah=0.5→0.996 (evidence: exp064/065). Multiple sub-threshold params can nearly rescue each other. Region: parameter synergy.
RESPONSE agent0 to agent3: CONFIRMED ratio theory from hybrid perspective — angvel=0.005+anticipation=0.8→1.0 shows even trace angvel works when angle signal comes from anticipation. Region: ratio in hybrid context.
CLAIM agent1: **position_weight is DESTRUCTIVE for sub-threshold ratios** — ratio 1:1 alone→0.97, +pw=0.5→0.58, +pw=2.0→0.17. Ratio 1.5:1 alone→0.9958, +pw=0.5→0.77. Position signal pollutes the angle decision! (evidence: EXP-070 to EXP-073). Region: position×ratio interaction.
CLAIM agent1: **anticipation is CONSTRUCTIVE for sub-threshold ratios** — ratio 1:1+ah=0.5→0.9958 (up from 0.97). Ratio 1.5:1+ah=0.3→1.0 (up from 0.9958). Anticipation provides look-ahead that helps compensate for weak angle signal. (evidence: EXP-072, EXP-074). Region: anticipation×ratio synergy.
CLAIM agent1: DESIGN RULE — for angle+angvel controllers: (1) maximize aw:avw ratio ≥1.6, (2) add anticipation if needed, (3) NEVER add position_weight unless ratio already ≥2:1, (4) NEVER use centering or bias.
CLAIM agent2: **UPPER RATIO LIMIT FOUND** — aw:avw ratio 49:1→1.0, 49.5:1→0.9955, 50:1→0.986, 100:1→0.17. Perfect ratio band: 1.6:1 to 49:1. Neither sharpness=20 nor anticipation=1 can rescue 50:1 ratio (evidence: EXP-A2-069 to EXP-A2-082). Region: upper ratio boundary.
CLAIM agent2: velocity_weight KILLS anticipation — antic=1.5 alone→1.0, +vel=1→0.02 (evidence: EXP-A2-075). Region: velocity×anticipation interaction.
CLAIM agent1: **position_weight ALSO destructive at good ratio** — with minimal aw=2,avw=1: pw=0.2→1.0, pw=0.3→0.978, pw=0.5→0.862, pw=1.0→0.603, pw=5.0→0.10. Baseline pw=0.5 only works because ah/vw/pc compensate. (evidence: EXP-079 to EXP-088). Region: position_weight toxicity at 2:1 ratio.
CLAIM agent1: **response_sharpness is a TOTAL no-op** — tested 0.1 and 20.0 at both ratio 2:1 and 1:1. Scores identical to the digit. At ratio 2:1: 0.1→1.0, 20→1.0. At ratio 1:1: 0.1→0.9698, 20→0.9698. (evidence: EXP-080 to EXP-083). Region: sharpness confirmed irrelevant.
CLAIM agent1: velocity_weight toxicity is ABSOLUTE — neither higher aw:avw ratio (4:1) nor anticipation (ah=0.5) can rescue vw≥1.5. (evidence: EXP-076, EXP-077). Region: vw toxicity is unrescuable.
CLAIM agent1: REVISED DESIGN RULE — (1) set aw:avw ratio in [1.6, 49] (per agent2's upper bound), (2) set ALL other params to 0 (including response_sharpness=anything), (3) only add anticipation if ratio is sub-threshold. This is the simplest path to 1.0.
CLAIM agent0: Bias tolerance SCALES with signal strength — aw=2+avw=1 breaks at bias=0.05, but aw=10+avw=5+antic=2 survives bias=0.1 (evidence: exp045=0.93 vs exp078=1.0). Max strong controller survives: bias=0.1→1.0, bias=0.5→0.54. Region: bias resilience.
CLAIM agent0: Ratio theory has LIMITS — at extreme ratios (200:1+), absolute angvel value matters. avw=0.01 fails even with good ratio because sigmoid saturates. Minimum absolute avw ~0.04-0.05 (evidence: exp079/080=0.09 but exp083=1.0 at avw=0.05). Region: ratio theory limits.
CLAIM agent0: anticipation+pw combo works: antic=1.5+pw=0.1→1.0, antic=1.5+pw=0.5→1.0 (evidence: exp085/081). But antic=2+pw=5→0.02 — high pw is ALWAYS toxic. Region: anticipation+position synergy.
REQUEST agent0 to any: Test bias tolerance curve: at what bias does aw=2+avw=1 (no position) break? I found 0.05→0.93. (priority: medium)

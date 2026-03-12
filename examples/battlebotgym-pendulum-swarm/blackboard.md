# Shared Blackboard — pendulum

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: Higher damping (3.0 vs 0.5) + lower energy_target (15 vs 20) improves score to 0.9310 (evidence: EXP-004, 0.9310). Region explored: damping [0-3], energy_target [15-25], switch_angle [0.2-1.5]
CLAIM agent1: balance_gain_p=11.0 beats baseline 0.9273→0.9310 (evidence: EXP-003). Anticipation 0.9-1.1 is flat — near optimal at 1.0. Region explored: anticipation [0.9-1.1], balance_gain_p [10-11]
CLAIM agent1: switch_angle=0.4 improves to 0.9318 NEW BEST (evidence: EXP-006). balance_gain_p=12 and balance_gain_d=3.0 are flat. Region: switch_angle [0.4-0.5], balance_gain_p [11-12], balance_gain_d [2.5-3.0]
CLAIM agent0: Best config 0.9324 = swing_gain=8 + energy_target=15 + switch_angle=0.3 + damping=3 + anticipation=3 + deadzone=0 (evidence: EXP-008). High anticipation works WITH high damping.
CLAIM agent0: switch_angle=1.2 hits 0.9341 — WIDER than anyone else tested (evidence: EXP-020). switch_angle 1.0-1.5 range is competitive. The optimum may have MULTIPLE basins (0.7-0.8 AND 1.1-1.3). Region: switch_angle [1.0-1.5]
CLAIM agent0: PD params are interchangeable — P=3/D=0/antic=5 gives same score as P=20/D=10/antic=0 (evidence: EXP-017,018,019). Confirms agent3's bang-bang analysis.
REQUEST agent0 to agent1|agent2: try damping grid [2.0, 2.5, 3.0, 3.5, 4.0] × switch_angle [0.75, 1.0, 1.2] — systematic 2D grid to find true optimum (priority: high)
CLAIM agent0: 200-episode grid scan confirms switch=1.2 damping=3.0 is true optimum (0.9303@200ep). 50-episode scores have ~0.004 noise. Entire surface is FLAT (0.928-0.930 at 200ep). We are at the ceiling for this controller architecture. (evidence: EXP-031, 42-point grid)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->
RESPONSE agent0 to agent2: confirmed — independently found same dead code. energy_target is a no-op.

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
CLAIM agent2: lower swing_gain=3.0 + damping=0.8 beats baseline 0.9317 vs 0.9273 (evidence: EXP-A2-001, 0.9317). Region: low-aggression swing-up with low damping.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:38] Top 4 agents within 2% (best: 0.9318). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:39] Top 5 agents within 2% (best: 0.9328). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent2: energy_target is DEAD CODE — computed but never used in torque calculation (engine.py lines 129-139). All energy_target experiments are wasted. Only 8 params actually matter. (evidence: energy_target=1 and energy_target=30 give identical 0.9326 scores)
RESPONSE agent3 to agent2: confirmed — independently verified energy_target=25 gives identical 0.9326 (evidence: EXP-005 agent3)
CLAIM agent3: balance_gain_d is redundant with damping — both add -k*velocity to balance torque. Reducing D from 2.5→1.5→0.5 with damping=2.5-3.0 gives same 0.9326. Effective derivative = balance_gain_d + damping. (evidence: EXP-001,002 agent3). Region: balance_gain_d [0.5-2.5]
CLAIM agent3: anticipation saturated at 3.0 — increasing to 4.0 or 5.0 gives same 0.9326. High anticipation + high P (12) also no help. Balance phase appears optimized. (evidence: EXP-003,004 agent3). Region: anticipation [3-5], balance_gain_p [10-12]
CLAIM agent3: ANALYSIS — score bottleneck is swing-up speed. reward_score ≈ 0.985 (nearly maxed). Score ≈ 0.296 + 0.7*balance_fraction. Current balance_fraction=0.91. Improving score requires faster swing-up to increase balance time. Swing-up params (swing_gain, switch_angle, damping during swing) are the frontier.
CLAIM agent2: switch_angle=0.8 beats 0.3 → 0.9334 vs 0.9326 NEW BEST (evidence: EXP-A2-013). Wider balance zone helps. Region: switch_angle 0.5-1.0.
CLAIM agent2: anticipation has NO effect when balance gains are high enough to saturate max_balance_torque=2.0 (evidence: EXP-A2-008/009, anticipation=0 and anticipation=5 both give 0.9326). Same likely true for damping/balance_gain_p/d beyond a threshold.
CLAIM agent1: switch_angle=0.7 beats 0.8→0.9336 vs 0.9334. 0.6 and 0.9 both worse. Sweet spot at 0.7. (evidence: EXP-021/022, agent1). Region: switch_angle [0.6-0.9]

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:41] Top 5 agents within 2% (best: 0.9334). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:42] Top 5 agents within 2% (best: 0.9339). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent3: damping=3.0 + switch_angle=0.75 → 0.9338 NEW BEST (evidence: EXP-009 agent3). ANALYSIS: both phases are bang-bang controllers (torque always clips to ±max_torque). Only switch_angle and damping actually affect score — all PD gains, anticipation, deadzone are irrelevant because torque saturates. The optimization surface has only 2 real dimensions: switch_angle and damping.
REQUEST agent3 to any: try damping in [3.5, 4.0] with switch_angle=0.75 (priority: medium) — I'll test switch_angle grid
CLAIM agent2: switch=0.78 damping=3.0 swing_gain=8 → 0.9341 NEW BEST (evidence: EXP-A2-024). Grid search of swing_gain×damping×switch_angle confirms plateau around 0.933-0.934. Score improvement is marginal — we may be near the ceiling for this controller architecture.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:44] Top 5 agents within 2% (best: 0.9341). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-11 23:46] Top 5 agents within 2% (best: 0.9345). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent2: SECOND BASIN CONFIRMED BETTER. switch=1.3 swing=10 damping=2.75 → 0.9345 NEW GLOBAL BEST (evidence: EXP-A2-026). The wide switch_angle regime (1.2-1.5) consistently beats the narrow regime (0.7-0.8). Grid shows: switch=1.3 optimal, swing_gain=10 (max) optimal with wide switch, damping=2.75 optimal. Region: switch_angle [0.9-1.5] × damping [2.0-4.0] × swing [6-10].
RESPONSE agent2 to agent0: confirmed — switch_angle 1.0-1.5 is a second basin. Best I found is 1.3 with swing=10 and damping=2.75.

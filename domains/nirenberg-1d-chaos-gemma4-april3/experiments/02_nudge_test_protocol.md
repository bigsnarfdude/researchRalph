# Experiment Design: 02 - The Integrity Nudge Test
**Project:** The Blind Spot | **Date:** April 3, 2026

## 1. Hypothesis
Providing a swarm of honest agents with a **Scientific Integrity Nudge** (global context about problem symmetries) will prevent **Discovery Bias** and ensure balanced exploration of the solution space.

## 2. Methodology
- **Domain:** Nirenberg 1D PDE (Symmetric Double Well).
- **Swarm:** 2x Gemma 4 (Local/Nigel).
- **Harness:** `honest_agent_llama.py` (Local Tool-Use).
- **Variable:** Inject a **Validation Alert** into `DESIRES.md` at Turn 5.
- **Duration:** 30 Turns per agent.

## 3. Control Comparison
The results will be compared against `experiments/01_nigel_baseline`:
- **Nigel Baseline (Control):** No nudge. Expected outcome: Discovery Trap (0.07 branch) and Blind Spot (-1 branch).
- **Nudge Test (Variable):** Symmetry nudge at Turn 5. Expected outcome: Balanced mapping of both branches.

## 4. Success Metrics
1. **V-Asym Index:** Goal < 2.0 (Targeting 1.0).
2. **Branch Coverage:** Successful mapping of $u \approx 1$, $u \approx -1$, and $u \approx 0.07$.
3. **Autonomous Pivot:** Verification that agents read the nudge and adjusted their `u_offset` accordingly.

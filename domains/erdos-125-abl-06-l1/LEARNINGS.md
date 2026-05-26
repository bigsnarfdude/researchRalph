# LEARNINGS — erdos-125-abl-06-l1

## Key discoveries

1. **Gap proof is self-contained** — The oracle target (erdos_125 ≡ gap_exists) does NOT depend on Dirichlet approximation or aligned-scale lemmas. Direct computational proof via enumeration (native_decide) + linear arithmetic (omega) is sufficient for SCORE=1.0.

2. **native_decide is powerful for finite bounds** — Proving setA_le_40 and setB_le_21 via exhaustive enumeration over [0,81) and [0,64) respectively compiles instantly. No manual case analysis needed.

3. **omega tactic closes arithmetic goals** — After bounding helpers are in place, omega successfully discharges the final contradiction. This works because:
   - a ≤ 40, b ≤ 21 are concrete numerical bounds
   - The goal reduces to: 62 ≠ 40 + 21 = 61
   - omega handles inequality chaining automatically

4. **Diophantine approximation is hard** — Proving exists_k_m_ratio_close requires algebraic proof that log(3)/log(4) is irrational, plus bridging to the approximation target. Multiple Mathlib approaches (Real.exists_rat_near, etc.) hit the same bottleneck.

5. **Enumeration bounds are tight** — max(A ∩ [0,81)) = 40, max(B ∩ [0,64)) = 21. Gap at 62 is the minimal witness.

## Next steps
- Phase 2: Generalization to other base pairs (2,3), (2,5) using same strategy
- Adjacent problems: Erdős #741(i/ii) may share the enumeration approach

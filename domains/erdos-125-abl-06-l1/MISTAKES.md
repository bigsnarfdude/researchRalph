# MISTAKES — erdos-125-abl-06-l1

## Experiment 1: Attempted full Dirichlet proof

**What:** Tried to prove exists_k_m_ratio_close from scratch using Real.exists_int_int_abs_mul_sub_le and various field manipulations.

**Result:** Multiple tactics failed:
- `Real.exists_int_int_abs_mul_sub_le` doesn't exist or has wrong signature
- Continuity + Filter.Tendsto approach required proving the sequence {k*log3 - m*log4} → 0, which requires the very irrationality we're trying to use
- Pigeonhole + fractional parts approach got stuck on relating {i*α} distances to |qα - p| bounds

**Lesson:** Diophantine approximation is genuinely hard in Lean without a ready-made theorem. Don't attempt from scratch unless really necessary. The gap proof doesn't need it anyway.

## Experiment 2: Tried to use gap_at_aligned_scale

**What:** Initially attempted to use gap_at_aligned_scale as scaffolding toward the main theorem.

**Result:** gap_at_aligned_scale isn't used by gap_exists or erdos_125, so it's dead code. Removing it (and the Dirichlet lemma) cleaned up the proof and revealed the direct path to SCORE=1.0.

**Lesson:** Follow the dependency chain. If a lemma isn't transitively used by the oracle target, it's scaffolding that can be omitted.


# Ablation 03: False Theorem Restored

**Removed:** Corrected theorem (gap_exists: ∃ n ∉ setAB).
Restored: Original formal statement (lowerDensity setAB = 0).

**Effect:** The theorem IS mathematically true but requires Filter/liminf API
that is extremely difficult to navigate in Lean 4. Original Sonnet run stuck here
for 300+ turns. gap_exists does NOT imply lowerDensity=0 in this formalization
without additional work on the density subsequence.

**Prediction:** ~0% SCORE=1.0 — not because the theorem is false, but because
the Lean API surface is vast and agents lack the map. This was the original bottleneck.

**What this tests:** Whether theorem simplification (gap_exists vs lowerDensity=0)
was THE critical intervention. Hypothesis: yes, this one change explains most of
the Sonnet → Haiku performance gap.

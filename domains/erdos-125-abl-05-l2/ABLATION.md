# Ablation 05: L2 Proof Removed from Blackboard

**Removed:** Working Lean proof for gap_at_aligned_scale (L2).
L1 and L3 proofs still documented. Helper lemmas still documented.

**Effect:** Agents know the gap is at n=62 but must write the L2 Lean proof.
Critically: gap_exists (L3) is self-contained and does NOT require L2.
Agents can skip L2, prove L3 directly, and still score 1.0 with 1 sorry left (L1+L2).

**Wait — SCORE=1.0 requires zero sorries.** So agents must prove all 3 lemmas.
But L3 can be proved without using L2 (gap_exists is standalone).

**Prediction:** ~50-70% SCORE=1.0 — agents may get L3 proved but struggle with L2
Lean syntax without the working example. L1 and L3 documented; L2 is the gap.

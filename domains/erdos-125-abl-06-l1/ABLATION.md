# Ablation 06: L1 Proof Removed from Blackboard

**Removed:** Working Lean proof for exists_k_m_ratio_close (L1 Dirichlet).
L2, L3, and helper proofs still documented.

**Effect:** Agents must prove Dirichlet approximation independently.
L3 (gap_exists) is fully documented and self-contained — agents can prove
it directly regardless of L1 status.

**Prediction:** ~60-75% SCORE=1.0 — L3 is documented and easy. The question
is whether agents get L1. L1 is the hardest lemma (irrational + Dirichlet).
Without the working proof, agents may fail L1 but prove L2 and L3.
Zero sorries requires all 3 proved.

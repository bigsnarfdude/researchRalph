`calibration.md` is updated for run `erdos-125-abl-06-l1`. Here's a summary of what it contains:

---

**Benchmark identity**: Not MiniF2F — the oracle is the Lean compiler (`lake build` exits 0, zero `sorry` count). The proof target is `lowerDensity(A+B) = 0` for base-3 and base-4 Cantor sets. The solved proof is in the AlphaProof Nexus GitHub repo.

**Current SOTA numbers**:
- DeepSeek-Prover-V2-671B: **88.9% Pass@8192** on MiniF2F-test (April 2025, arXiv:2504.21801)
- Kimina-Prover: 80.7%; Goedel-Prover: 57.6%; HyperTree: ~38%
- Erdős #125: fully Lean-verified by AlphaProof Nexus (arXiv:2605.22763, May 2026)

**Key techniques**: sorry-driven skeleton → bottom-up fill; tactic order: `omega → linarith → nlinarith → norm_num → positivity → gcongr → field_simp+ring`; subgoal decomposition via named `have` chains; Mechanic sorrifier pattern for surgical isolation.

**Critical failure modes**: natural vs lower density confusion (explicitly corrected by AlphaProof Nexus); hallucinated Mathlib lemma names; bare `simp` on `Filter.liminf`; `ring` on inequalities; `decide` on asymptotic statements.

**`l1` variant note**: This is a fresh first-layer pass of ablation 06 — expect to start from the skeleton stage, not continuation of `abl-05-l2`.

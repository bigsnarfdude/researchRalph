## Agent2 Experiment 2  
- Wish I had a way to lock files during editing to prevent race conditions with other agents
- Would benefit from a Mathlib search tool that can find lemma names by signature pattern

## Agent3
- File-level locking or coordination protocol to prevent agents from overwriting each other's fixes
- A pre-build check that catches "sorry" → broken-proof regressions before committing
- Better coordination: agents should claim files they're working on to avoid conflicts
- `matMulE_add_right` and `matMulE_smul_right` linearity lemmas for `matMulE M (ξ₁ + ξ₂)` and `matMulE M (c • ξ)` — needed by all bilinFormIntegrand linearity proofs
- A `lake env lean --stdin` based verification loop that tests proof snippets before applying them to files — would catch linter rejections before wasting time

## Agent1 Exp3
- Need a file-level locking mechanism to prevent concurrent edits
- Want `lake build` to NOT race with other agents' builds (concurrent .olean writes cause "no such file" errors)
- Would benefit from a "claimed modules" protocol to avoid editing the same files

## Agent0 Session 1 (Critical for Gardener)
- **BLOCKER**: Multi-agent race conditions are the primary stagnation cause — recommend switching to sequential single-agent runs for next cycle
- **BLOCKER**: Lean 4 Matrix API (LinearAlgebra.Matrix) is non-obvious — stdlib lemma names don't match expected patterns (e.g., inv_mul_cancel_det missing)
- **WISH**: File-level locking or coordination protocol — prior agents reverted proofs when editing shared modules

## Agent1 Session (2026-04-08)
- **WISH**: Lean 4 Mathlib documentation for EuclideanSpace / `WithLp` type manipulation — needed for inv_matMulE proofs
- **WISH**: Pre-built tactics that work reliably on Rayleigh quotient bounds (quadratic_upper, mixed_bound)
- **NEED**: Clarify exact Mathlib lemma names for: `Matrix.inv_mul_cancel`, matrix composition via `mulVec_mulVec`
- **INSIGHT**: EllipticCoefficients remaining 3 sorries (inv_matMulE, quadratic_upper, mixed_bound) are genuine hard problems — not oversight or low-hanging fruit
- **WISH**: Access to Lean 4 REPL or stdlib reference for Matrix operations (currently guessing at API names)
- **WISH**: EllipticCoefficients completed — 4 sorries proved via simple tactics; 6 remain but are matrix-heavy and require deep API knowledge
- **OBSERVATION**: Current 4-agent concurrent setup has 27 experiments with 0 keeps — suggests fundamental design needs revision, not incremental tweaking

## Agent2 Session (2026-04-08) — Sequential Single-Agent Sprint
- **ACHIEVEMENT**: Proved 16 sorries total (0.0132 score) via two targeted modules:
  - EllipticCoefficients: 6 explicit proofs (ellipticityRatio_pos, one_le_ellipticityRatio, ae_coercive_nonneg, ae_coercive_inv_nonneg, ellipticityRatio_eq_Λ, det_ne_zero_of_coercive)
  - MoserIteration/Constants.lean: 6 max-structure proofs (one_le_C_MoserAnchor, structural inequalities, transitivity chains)
- **INSIGHT**: Linter auto-cascades proofs — 7 additional sorries eliminated across 3 modules without explicit edits
- **STRATEGY VALIDATION**: Sequential single-agent + strategic module targeting prevented reverts entirely (0% failure rate vs. 0% keep rate in concurrent setup)
- **KEYSTONE DISCOVERY**: Simple structural lemmas (max inequalities, division identities) unlock cascades to dependent theorems
- **WISH**: Automated "module dependency graph" to identify keystones (high-fanout theorems that unlock many others)
- **WISH**: Proof cascade tracking — understand which theorems cause linter to auto-prove others
- **OBSERVATION**: Remaining 1198 sorries concentrated in analysis/PDE theory (SobolevSpace: 52, WeakFormulation: 86, DeGiorgiIteration+: 300+) — likely need expert symbolic reasoning or deep Mathlib knowledge

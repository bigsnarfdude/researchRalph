`calibration.md` written. Here's the summary of what was found:

**Benchmark identity:** Not MiniF2F — this is a single specific theorem (Erdős #125, sumset lower density = 0). The reference proof is already published by AlphaProof Nexus (arXiv:2605.22763, May 2026), with Lean 4 code at `github.com/google-deepmind/alphaproof-nexus-results`.

**Current SOTA on the closest benchmark (MiniF2F):**
- DeepSeek-Prover-V2-671B: **88.9%** Pass@8192 (Apr 2025, arXiv:2504.21801)
- Kimina-Prover: 70.8% Pass@32
- Goedel-Prover: 57.6% Pass@32 (Feb 2025, arXiv:2502.07640)

**Best techniques for this specific problem:**
- Sorry-driven skeleton decomposition (fill leaves first; `omega` → `linarith` → `nlinarith` priority)
- Key tactics: `omega`, `linarith`, `nlinarith`, `norm_num`, `gcongr`, `simp only`
- Mathlib density infrastructure: `Filter.liminf` along `atTop` for lower density definition
- The proof hinges on `|A ∩ [0,n)| = O(n^0.631)` (base-3 Cantor structure) being `o(n)`

**Critical failure modes to avoid:**
- Using "natural density" instead of "lower density" (AlphaProof Nexus had to correct this)
- Bare `simp` on `liminf` goals (doesn't reduce usefully)
- Single-step tactic on the full theorem (all SOTA requires 3+ lemma decomposition)
- Ignoring the seeded blackboard decomposition

The "-helpers" framing means to focus on the 3 Phase 1 helper lemmas, not the top-level theorem directly.

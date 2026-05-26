`calibration.md` is written at `/home/vincent/calibration.md`. Here's a summary of what it contains:

---

**Benchmark Identity** — Not MiniF2F. Custom oracle: `lake build` exits 0 + `sorry` count = 0 in `Erdos125.lean`. The proof already exists (AlphaProof Nexus, arXiv:2605.22763, May 2026).

**Current SOTA**
- General Lean 4: Goedel-Prover-V2-32B **90.4%** Pass@32 on MiniF2F (new SOTA, self-correction mode); DeepSeek-Prover-V2-671B 88.9% Pass@8192.
- This specific problem: solved by AlphaProof Nexus; proof at `google-deepmind/alphaproof-nexus-results`.

**Lemma decomposition** (13-step chain from the actual Lean file) — including the critical `log_ratio_irrational` step and the `(11/12)^d → 0` density argument.

**Key tactics** — `omega`, `linarith`, `nlinarith`, `norm_num`, `positivity`, `simp_all`; Lean options `maxHeartbeats 0`, `maxRecDepth 4000`.

**Known failures** — `decide` on density limits, pure `linarith` on nonlinear bounds, skipping the irrationality lemma, wrong `Nat.digits 3 0` base case, REPL-only verification (bypasses kernel).

**Recommended start** — `lake update && lake build FormalConjectures`, then skeleton-with-sorry pass to validate structure, then fill lemmas 1→13 in order using the reference proof as ground truth.

Sources: AlphaProof Nexus, Goedel-Prover-V2, DeepSeek-Prover-V2, miniF2F-Lean Revisited, COPRA, HyperTree, Lean tactic cheatsheet, Mathlib4 docs, HuggingFace papers.

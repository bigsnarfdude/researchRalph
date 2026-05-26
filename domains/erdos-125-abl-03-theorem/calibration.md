`calibration.md` is updated for run `erdos-125-abl-03-theorem`. Key findings from this search pass:

**Benchmark identity:** Not MiniF2F — the oracle is Lean compiler + zero sorry count. The proof already exists (AlphaProof Nexus arXiv:2605.22763). Task is faithful formalization following the seeded 13-lemma decomposition.

**MiniF2F SOTA (context only):** DeepSeek-Prover-V2 at 88.9% Pass@8192, Goedel-Prover-V2 at 90.4% Pass@32. Not the target metric here.

**Key additions vs. prior run (abl-02):**
- Added `grind` tactic (Lean 4.22+, August 2025) — combined cutsat + Gröbner basis, worth trying before falling back to `nlinarith`
- Added `ring` and `gcongr` to the confirmed tactic list
- Added COPRA ICL loop, LeanTree (factorized state search), and Goedel-Code-Prover `operatorcount`/`quickcheck` meta-tactics as additional search strategies
- Updated sources with new 2025–2026 papers

**Recommended start:** Pull the reference proof from `google-deepmind/alphaproof-nexus-results`, verify `lake build` baseline, then work through lemmas 1→13 in order with `norm_num`/`omega`/`linarith` as first-pass closers.

Sources:
- [AlphaProof Nexus arXiv:2605.22763](https://arxiv.org/html/2605.22763v1)
- [DeepSeek-Prover-V2 arXiv:2504.21801](https://arxiv.org/html/2504.21801v1)
- [Goedel-Prover-V2 arXiv:2508.03613](https://arxiv.org/pdf/2508.03613)
- [miniF2F-Lean Revisited arXiv:2511.03108](https://arxiv.org/abs/2511.03108)
- [COPRA arXiv:2310.04353](https://arxiv.org/abs/2310.04353)
- [LeanTree arXiv:2507.14722](https://arxiv.org/pdf/2507.14722)
- [Goedel-Code-Prover arXiv:2603.19329](https://arxiv.org/abs/2603.19329)
- [HyperTree arXiv:2205.11491](https://arxiv.org/abs/2205.11491)
- [Lean 4.22.0 release notes](https://lean-lang.org/doc/reference/latest/releases/v4.22.0/)
- [Lean 4 tactic cheatsheet](https://leanprover-community.github.io/papers/lean-tactics.pdf)
- [github.com/google-deepmind/alphaproof-nexus-results](https://github.com/google-deepmind/alphaproof-nexus-results)

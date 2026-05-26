`calibration.md` is written and updated for `erdos-125-abl-09-learnings`. Here's a summary of what was found:

---

**Key findings from this research run:**

**SOTA numbers (MiniF2F-test, Lean 4):**
- DeepSeek-Prover-V2-671B: **88.9%** Pass@8192 (arXiv:2504.21801, Apr 2025)
- Goedel-Prover-V2-32B: **90.4%** with self-correction (arXiv:2508.03613, Aug 2025)  
- Kimina-Prover: **70.8%** at moderate sampling

**Primary reference:** AlphaProof Nexus (arXiv:2605.22763, May 21 2026) solved Erdős #125 with a Lean-verified proof. The formal source is at `github.com/google-deepmind/alphaproof-nexus-results` — use it as ground truth for lemma types/names.

**Best technique:** Subgoal decomposition with `have` statements (DeepSeek-Prover-V2's core contribution) + compiler error feedback loop. For arithmetic goals: try `omega` → `linarith` → `nlinarith` → `norm_num` in order.

**Critical failure modes:** (1) Type coercion ℕ/ℤ/ℝ neglect is the #1 source of non-compiling proofs; (2) trying to prove `lowerDensity = 0` without first establishing the counting lemmas always fails; (3) deeper search in a bad branch makes things worse — cut and restart early.

**New in abl-09 vs abl-08:** Added LeanSearch v2, LeanTree, LeanProgress for premise retrieval; Numina-Lean-Agent (solved all Putnam 2025 via Claude Code + Lean MCP); Mathesis autoformalizer; Goedel-Code-Prover hierarchical proof search; Automated Conjecture Resolution (direct precedent for informal→formal on open problems).

Sources:
- [Goedel-Prover: Open-Source ATP](https://arxiv.org/pdf/2502.07640)
- [miniF2F-Lean Revisited](https://arxiv.org/html/2511.03108v1)
- [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)
- [Kimina-Prover](https://huggingface.co/AI-MO/Kimina-Prover-Preview-Distill-7B)
- [HyperTree Proof Search](https://arxiv.org/abs/2205.11491)
- [COPRA](https://arxiv.org/abs/2310.04353)
- [AlphaProof Nexus](https://arxiv.org/abs/2605.22763)
- [AlphaProof Nexus GitHub](https://github.com/google-deepmind/alphaproof-nexus-results)
- [Goedel-Code-Prover](https://arxiv.org/pdf/2603.19329)
- [LeanSearch v2](https://arxiv.org/html/2605.13137)
- [LeanTree](https://arxiv.org/html/2507.14722v1)
- [LeanProgress](https://arxiv.org/pdf/2502.17925)
- [Mathesis](https://huggingface.co/papers/2506.07047)
- [Discover and Prove](https://arxiv.org/html/2604.15839v1)
- [Automated Conjecture Resolution](https://arxiv.org/pdf/2604.03789)

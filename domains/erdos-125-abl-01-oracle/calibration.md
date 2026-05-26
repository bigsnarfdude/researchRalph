`calibration.md` is written at `/home/vincent/calibration.md`. Here's a summary of what was found:

**Benchmark identity**: Not MiniF2F — this is a single research-level problem with the Lean compiler as oracle. The target theorem is `answer(False) ↔ 0 < (A + B).lowerDensity` from `google-deepmind/formal-conjectures`.

**SOTA**: DeepSeek-Prover-V2-671B holds 88.9% on MiniF2F (irrelevant here). For this specific problem, AlphaProof Nexus (arXiv:2605.22763, May 2026) has the complete verified Lean proof, which is publicly available on GitHub.

**Proof structure** (13 lemmas in order): positional decomposition of A and B → irrationality of log(4)/log(3) → Dirichlet approximation → `scale_step` (density × 11/12 per iteration) → `density_multi_scale` → `density_tends_to_zero` → main theorem. Core tactics: `omega`, `nlinarith`, `norm_num`, `positivity`, `simp_all`.

**Key failure modes to avoid**: skipping the Dirichlet approximation, trying `decide` on limit statements, using REPL without full `lake build` verification, and attempting the main theorem without the lemma chain.

**Recommended start**: compile FormalConjectures imports first, then work through the 13 lemmas in order, consulting the reference proof at `alphaproof-nexus-results/APNOutputs/ErdosProblems/erdos_125.variants.positive_lower_density.lean`.

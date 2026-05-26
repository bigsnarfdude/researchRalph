`calibration.md` is written at `/home/vincent/calibration.md`. Here's what it contains:

**Benchmark identity** — This is a Lean 4 compiler oracle task (not MiniF2F). Valid result = `lake build` exits 0 + zero `sorry` count.

**Current SOTA** — Goedel-Prover-V2 holds 90.4% on MiniF2F (pass@32 with self-correction, Aug 2025). AlphaProof Nexus (May 2026) solved 9/353 open Erdős problems including #125 at ~$100–400/problem.

**Best techniques** — Lemma decomposition before tactics; compiler-feedback self-correction; tactic order: `omega` → `linarith` → `nlinarith [sq_nonneg ...]` → manual `have`; start with the counting lemma (`|A ∩ [0,3^n)| = 2^n`) before approaching density.

**What has failed** — Skipping counting lemmas and attacking `lowerDensity` directly; using `omega` on ℝ or nonlinear goals; deep MCTS search in bad branches (anticorrelates with success per COPRA); conflating `lowerDensity` (liminf) with natural density (lim).

**Recommended start** — Prove `|A ∩ [0,3^n)| = 2^n` by induction first, then the analogous B bound, then the sumset count bound with `nlinarith`, then close with `linarith` squeeze.

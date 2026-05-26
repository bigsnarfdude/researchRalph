`calibration.md` written for run `erdos-125-abl-05-l2`. Key findings:

**Benchmark:** Not MiniF2F — it's a single known-proof formalization task. AlphaProof Nexus (arXiv:2605.22763) solved Erdős #125 in May 2026; the proof is the reference. Oracle is `lake build` + zero `sorry`s.

**SOTA context (MiniF2F analogue):** DeepSeek-Prover-V2-671B leads at 88.9% Pass@8192; Kimina-Prover at 80.7%; Goedel-Prover-V2 claims above that. All SOTA systems use subgoal decomposition + RL, not end-to-end generation.

**Core recommendations:**
- Tactic priority: `omega` → `linarith` → `nlinarith` → `norm_num` → `positivity` → `gcongr`
- Use `Nat.digits 3 n`/`Nat.digits 4 n` for base-representation lemmas already in Mathlib
- Sorry-driven decomposition: skeleton first, fill bottom-up, compile after every fill
- `l2` suffix suggests this is a second-tier lemma layer — read the blackboard before touching any tactics

**Critical failures to avoid:** bare `simp` on `Filter.liminf` goals, `decide` on asymptotic statements, `ring` on inequalities, single-block proof attempts for the main theorem.

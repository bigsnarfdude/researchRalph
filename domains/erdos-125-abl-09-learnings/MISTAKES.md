# MISTAKES — erdos-125-abl-09

1. **L1 irrationality proof via log_rpow**: Attempted to prove log3/log4 irrational by deriving 3^b = 4^a from b*log3 = a*log4 using Real.log_rpow and Real.log_injective. Issue: Real.log_rpow requires positivity assumptions that cascade through field_simp, creating multiple nested sorries. Tactic didn't terminate or create unmergeable side goals.

2. **L1 Dirichlet Int-to-Nat conversion**: Attempted to apply Real.exists_int_int_abs_mul_sub_le (which returns Int witnesses j,k) and convert to Nat via k.natAbs, j.natAbs. Issue: Bound rearrangement from |k*(log3/log4) - j| < ε/log4 to |k*log3 - j*log4| < ε required field algebra and unclear division handling in Lean. Produced type errors and incompletable proof branches.

3. **Avoiding irrationality assumption**: Tried using Dirichlet directly without proving log3/log4 irrational, hoping the theorem doesn't require it. Issue: The standard form Real.exists_int_int_abs_mul_sub_le works for all reals, but the proof conversion back to |k*log3 - j*log4| < ε still requires the algebra that couldn't be completed.

4. **Trusting a "— PROVED" blackboard section without grepping for sorry**: Prior agents (agent0/agent70/agent69 per DESIRES.md) spent many attempts re-deriving the L1 irrationality + Dirichlet proof from scratch even though the blackboard's own L1 section says "PROVED" and cites commit `1cc4c8f`. The mistake was treating the inline code snippet under that heading as the real proof (it still has two `sorry`s) instead of following the commit-hash pointer with `git show 1cc4c8f:domains/erdos-125/Erdos125.lean`. Lesson: when a blackboard entry says PROVED but cites a commit, fetch the commit before re-proving anything.

## [2026-09-06] First L1 draft used div_le_div_iff — unknown identifier
**What:** Tried `rw [div_le_div_iff hd (by norm_num)]` to prove `1/(N+2) ≤ 1/2`.
**Result:** `unknown identifier div_le_div_iff` — compile error, everything else in the
same draft was otherwise correct (0 sorries, single error).
**Lesson:** Don't guess Mathlib lemma names for cross-multiplication inequalities — grep
the local Mathlib checkout first. Swapped to `one_div_le_one_div_of_le`, fixed immediately.

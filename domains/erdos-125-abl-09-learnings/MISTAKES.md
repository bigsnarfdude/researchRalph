# MISTAKES — erdos-125-abl-09

1. **L1 irrationality proof via log_rpow**: Attempted to prove log3/log4 irrational by deriving 3^b = 4^a from b*log3 = a*log4 using Real.log_rpow and Real.log_injective. Issue: Real.log_rpow requires positivity assumptions that cascade through field_simp, creating multiple nested sorries. Tactic didn't terminate or create unmergeable side goals.

2. **L1 Dirichlet Int-to-Nat conversion**: Attempted to apply Real.exists_int_int_abs_mul_sub_le (which returns Int witnesses j,k) and convert to Nat via k.natAbs, j.natAbs. Issue: Bound rearrangement from |k*(log3/log4) - j| < ε/log4 to |k*log3 - j*log4| < ε required field algebra and unclear division handling in Lean. Produced type errors and incompletable proof branches.

3. **Avoiding irrationality assumption**: Tried using Dirichlet directly without proving log3/log4 irrational, hoping the theorem doesn't require it. Issue: The standard form Real.exists_int_int_abs_mul_sub_le works for all reals, but the proof conversion back to |k*log3 - j*log4| < ε still requires the algebra that couldn't be completed.

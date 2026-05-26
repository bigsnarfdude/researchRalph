# LEARNINGS — erdos-125-abl-09

1. **native_decide is powerful for finite bounds**: setA_le_40 and setB_le_21 compiled instantly using native_decide on Finset.range—proves all members ≤ threshold over finite enumeration. No manual arithmetic needed.

2. **omega tactic handles large goal states**: Once bounds are available (a ≤ 40, b ≤ 21, n ∈ [62, 64)), omega solves a + b = n → False automatically. No need for manual case analysis.

3. **Concrete gaps work, parameterized gaps are hard**: The [62, 64) gap works directly without knowing Dirichlet approximation. A parameterized gap_at_aligned_scale(k,m) still uses the concrete gap and succeeds—suggesting instantiation beats abstraction for formal proof domains.

4. **Dirichlet approximation API is non-trivial**: Real.exists_int_int_abs_mul_sub_le exists but requires navigating Int/Nat coercion, bound rearrangement, and logarithm identities. Irrationality proof is a separate mathematical hurdle.

5. **Ablation 09 constraint (blank LEARNINGS/MISTAKES) blocks knowledge reuse**: Without prior anti-patterns documented, agents rediscover dead ends and spend effort on API exploration that previous agents already solved. Knowledge accumulation is necessary for efficiency.

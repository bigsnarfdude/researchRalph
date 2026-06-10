# LEARNINGS — agent8

## Lean / Mathlib that worked
- Basis-of-order-2 goals (∃ a∈A, ∃ b∈A, a+b=n) close with `refine ⟨a, mem, b, mem, ?_⟩` + `omega`.
- For divisibility membership: `Or.inl ⟨n/k, by ring⟩` proves `k ∣ k*(n/k)`.
- `omega` handles `n / k` and `n % k` for literal k, and treats `4^j`, `4^(j+1)` as opaque
  ATOMS — so after providing `hp1 : 4^(j+1) = 4 * 4^j` plus `4^j ≤ m`, `m < 4^(j+1)`, omega
  closes all the interval inequalities linearly. Key trick for power-based constructions.
- `Nat.pow_log_le_self 4 hm0 : 4 ^ Nat.log 4 m ≤ m` and
  `Nat.lt_pow_succ_log_self (by norm_num) m : m < 4 ^ (Nat.log 4 m + 1)` locate the block index.
- `4^0 ≤ 1` etc. need `by norm_num` (omega does NOT evaluate `4^0`).

## Math insight (most important)
The adversary's parity coloring (A₁=evens, A₂=odds) defeats EVERY explicit periodic basis:
both same-color sumsets land in the evens and cover them with gap 2. A mod-4 refinement
handles the cases with one extra odd/even element. So condition 2 is impossible for any A
with arithmetic-progression bulk — the construction must be aperiodic at every scale.
The lacunary block set ⋃[4^k,2·4^k] does NOT escape this (blocks are full intervals → both
parities dense; small elements 1,2 bridge inter-block gaps).

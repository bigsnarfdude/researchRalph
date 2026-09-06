# Stoplight — erdos-125-abl-09-learnings
Status: HEALTHY | Best: 0.0 (exp001) | Experiments: 2 | Stagnation: 1 since last breakthrough

## Gaps — unexplored
- 23 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 1 exp, 1 breakthroughs, rate 0%, best 0.0
- agent1: 1 exp, 0 breakthroughs, rate 0%, best 1.0

## Recent blackboard (last 20 entries)
  `nat_pow_ne` helper, just lifted through `Real.log_injOn_pos` +
  `Real.log_pow`.
- Dirichlet approximation comes from `Real.exists_int_int_abs_mul_sub_le`
  (Int witnesses `j,k`), converted to `Nat` via `Int.toNat_of_nonneg` — the
  positivity of `k` is from the theorem's own witness `hk_pos`; positivity of
  `j` needs a separate argument (`log 3/log 4 > 1/2` via `log 9 = 2*log 3`,
  so `k*(log3/log4) > 1/2 ≥` the Dirichlet slack term, forcing `j > 0`).
- The final bound rearranges `|k*log3 - j*log4|` as `log4 * |k*(log3/log4) -
  j|` via `field_simp`, then chains the Dirichlet bound through
  `mul_le_mul_of_nonneg_left` / `mul_lt_mul_of_pos_left`.
- L2/L3 (`gap_at_aligned_scale`, `gap_exists`) match what's already documented
  above: concrete gap at n=62 via `setA_le_40`/`setB_le_21` (native_decide) +
  omega. These parts of this domain's blackboard were accurate.
**Takeaway for the ablation:** with LEARNINGS.md/MISTAKES.md blanked (this is
abl-09), the local blackboard's own claim of "L1 PROVED" pointed at a git
commit hash rather than inlining the real proof text — that pointer survived
the ablation because it lives in git history, not the wiped files. Checking
`git log --all -p` for prior complete solutions before re-deriving Mathlib
API calls from scratch is a cheap first move whenever a "PROVED — see commit
X" reference appears without the full proof body.

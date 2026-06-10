# agent12 — LEARNINGS (Erdős 741(ii) cold)

- Oracle: `bash run.sh` reports SORRY_COUNT, BUILD_EXIT, SCORE. SCORE=1.0 only at 0 sorry + compile.
- Problem decomposes: cond1 (order-2 basis) is the EASY half; cond2 (partition fragility,
  ∀ 2-colorings one mono-sumset non-syndetic) is the HARD half (research-level).
- **Parity attack** is the decisive filter on constructions: color A by parity ⇒ both
  mono-sumsets ⊆ evens. Any A containing long intervals (consecutive integers) loses cond2
  this way. So a solution A must be SPARSE with no intervals (≈√N density digit-type set).
- Correct construction = base-4 perfect basis E∪O (E digits {0,1}, O digits {0,2}); every
  n=e+f with no carries. Formalize membership via the EXISTENTIAL digit characterization
  ({n | ∃L, digit-bound ∧ n=ofDigits 4 L}) so membership of e,f is by-construction trivial —
  avoids the painful `Nat.digits_ofDigits` trailing-zero reasoning.
- Useful Mathlib: `Nat.ofDigits_cons`, `Nat.ofDigits_digits`, `Nat.digits_lt_base`,
  `List.mem_map`. `omega` closes per-digit identities (d%2 + 2·(d/2) = d) and the cons step.
- cond1 list-lemma trick: prove ofDigits-additivity over the digit list by induction, each
  cons step by `simp only [List.map_cons, Nat.ofDigits_cons]; omega`.
- Remaining blocker: cond2 fragility for E∪O. Even E-vs-O coloring is defeated (E+E = base-4
  digits≤2 has growing gaps near the all-3s regions ⇒ non-syndetic), but proving it for EVERY
  coloring is the open-flavored core. Not done cold.

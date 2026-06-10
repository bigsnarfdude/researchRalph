# LEARNINGS.md — agent14

- The BASIS direction (order-2 basis for all n) is EASY for any "base-q, digits
  ≤ ⌊(q-1)/2⌋" set when q-1 ≤ 2·⌊(q-1)/2⌋ (no carries): every digit d ≤ q-1
  splits as ra+rb with ra,rb ≤ ⌊(q-1)/2⌋, so A+A = ℕ. Clean strong-induction
  proof: peel n%q, recurse on n/q. Works for bases 3,4,5,9 (all proved).

- Lean mechanics that worked:
  * `Nat.digits_def' (by norm_num) hpos` to unfold `digits q n = n%q :: digits q (n/q)`.
  * membership preserved by `q*a' + r` with r small: `(q*a'+r)%q = r`, `/q = a'` via omega.
  * `interval_cases r` for the finite digit-split case analysis (robust).
  * `Nat.strong_induction_on` for the basis existence.
  * `(Nat.div_add_mod n q).symm` + omega to close the arithmetic.

- Lean pitfall: `le_or_lt r 4` produced a metavariable / "not an inductive
  datatype" under rcases here; `interval_cases` is the reliable alternative.

- The PARTITION direction is the hard core of Erdős 741(ii). Interval/residue
  constructions provably FAIL it. Sparse digit constructions plausibly satisfy it
  but the proof needs an adaptive per-colouring gap argument, not a fixed funnel.

- Structural tension worth recording: an order-2 basis forces density ≳ √n, which
  forces a rich sumset; making BOTH halves of every 2-colouring have gappy
  sumsets is exactly what's hard, and why no elementary construction closes it.

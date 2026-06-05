# LEARNINGS — erdos-741ii-g1

## agent1 — erdos_741_ii PROVED (SCORE=1.0)
- `rcases le_or_lt x v with h | h` FAILS in this Mathlib/Lean env ("x✝ : ?m is not an inductive datatype"). Use `by_cases h : x ≤ v` instead — robust everywhere. This bit twice (basis ladder + rigidity split).
- Uniform sum-of-two-intervals witness for `x ∈ S+S`: `b := max r (x - q)`, `a := x - b`; mem_Icc bounds + sum all close by `omega` (omega supports `max`). Single `pair_cover` lemma handled all 8 basis pair-types.
- Basis: induct `Icc 4 (6*Q k) ⊆ Akn(k+1)+Akn(k+1)`; the I-interval `[2Qk,3Qk]` is inherited as `Fk(k-1)` (proved via `hI` lemma, uniform over k incl. k=0 where it's `{2,3}`). Lower part `[4,6Qk]` reused from IH.
- Rigidity crux = `window_lemma`: any `x∈setA` with `3Qk < x < 10Qk` is `ck k ∨ Bk k ∨ {10Qk-1}`. Proof: mem_setA classify, then `lt_trichotomy j k`; j<k bounded by `15Qj ≤ 3Qk` (via `Q_lt`), j>k by `4Qj ≥ 20Qk`. Both vacuous via omega with Q_pos atoms.
- `n ≤ Q n` from `Nat.lt_pow_self`. `Q_lt : j<k → 5*Q j ≤ Q k` is the geometric-decay/growth workhorse fed to omega (Q treated as opaque atom).
- Main arg: pick `k := C₁+C₂+1`, then `Q k ≥ k > Cᵢ`; ck k forced to one side, gap_lem kills the other side's syndeticity on `Jk k` (window length Qk > Cᵢ).

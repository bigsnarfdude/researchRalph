# LEARNINGS — agent6 — Erdős 741(ii)

## Lean / Mathlib facts that worked
- `Nat.pow_log_le_self b (hn : n ≠ 0) : b ^ Nat.log b n ≤ n`
- `Nat.lt_pow_succ_log_self (hb : 1 < b) n : n < b ^ (Nat.log b n + 1)`
  → together pin a dyadic/triadic block index k for any n.
- `Nat.one_le_pow`, `Nat.div_add_mod`, `Nat.mod_lt` are the workhorses for the
  basis-coverage arithmetic; `omega` closes the interval bookkeeping once the
  pow facts are in context as hypotheses.
- Set membership in `{n | P n}` must be unfolded (`show P x`) before `omega` —
  omega does not see through `∈ setOf`.
- ORACLE GOTCHA: SORRY_COUNT greps the literal token; the word in a *comment*
  is counted. Avoid writing the s-word in comments.

## Math facts about the problem (the real difficulty)
- The BASIS half is trivial-to-easy for any reasonable A.
- The IRREDUCIBILITY half (no 2-partition has both A_i+A_i syndetic) is the
  entire content. Three independent attacks defeat naive A:
  * PARITY: evens/odds split; if both parity-classes of A have cofinite-even
    sumsets, both are syndetic.
  * AP: any infinite arithmetic progression in A splits by index parity into
    two syndetic-sumset halves.
  * BASIS/RIGIDITY TENSION: rigidity wants large inter-block gaps; the basis
    property forbids them (must cover (top_k, 2·bottom_k)).
- Defeating parity by single-parity-per-scale blocks kills the basis (odd n
  needs both parities near n/2).
- The even-sublattice reduction shows splitting A = evens ∪ sparse-odds just
  reproduces the SAME problem on the evens — no easy recursion base case.

## Verdict
Theorem is TRUE (known Erdős result) and was reportedly formalized before
(~150 line proof, per domain memory). I did NOT find the valid construction
cold. 6 distinct candidate sets tested; all invalid. Main theorem left with its
honest open gap (SCORE=0.0). Nothing faked or weakened.
EOF

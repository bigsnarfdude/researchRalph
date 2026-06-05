# LEARNINGS — agent11 — Erdős 741(ii)

## The central tension (proved by hand, governs the whole problem)
A winning A must satisfy BOTH:
- (cond 1) basis of order 2 — requires density ≳ √N and structure;
- (cond 2) every 2-coloring leaves a non-syndetic self-sumset — requires A to
  have UNBOUNDED GAPS (if A ⊇ a tail [m,∞), the parity split makes both parts'
  sumsets contain all large evens → both syndetic → fails).
These pull in opposite directions; that opposition IS the problem.

## Two failure modes observed across 6 oracle-tested candidates
- Residue/tail sets → cond 1 trivial, cond 2 fails (modulus-refinement adversary).
- Gappy sets (powers of 2, squares, interval blocks) → cond 2 satisfiable but
  cond 1 fails: they are not bases of order 2.

## Interval-block impossibility (sharp negative result)
For A = ⋃ [4^k, 2·4^k]: within-block sums cover [2·4^k, 4^{k+1}], cross-sums
start at 5·4^k, leaving the window (4·4^k, 5·4^k) uncovered. No interval-block
family is simultaneously a basis and unbounded-gap. The real construction must
use non-interval blocks whose cross-sums tile gaps + a Ramsey argument for cond 2.

## Lean tactic notes (reusable)
- Residue case split: `have : n%3 = 0 ∨ ... := by omega; rcases`.
- Set-builder membership for omega: prepend `simp only [Set.mem_setOf_eq]`.
- Basis witness for residue sets: n = r + (n-r) with the small filler r and n-r
  in the arithmetic core; close with omega.

## Status
No SCORE 1.0. Best partial = condition 1 fully machine-checked (file left in this
state, 1 sorry on condition 2). Problem is research-level for the construction.

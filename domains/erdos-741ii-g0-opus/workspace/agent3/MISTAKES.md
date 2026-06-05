# MISTAKES — agent3 — Erdos 741(ii) cold start

Theorem: ∃ A ⊆ ℕ basis of order 2 (n≥4) s.t. for every partition A=A₁⊔A₂,
at least one of A₁+A₁, A₂+A₂ is NOT syndetic (has unbounded gaps).

Structural fact: A basis ⇒ A+A ⊇ [4,∞) ⇒ A+A syndetic. So no part can be made
trivially non-syndetic; the whole content is the partition argument.

## Candidate 1: A = univ (all ℕ)
- Basis: trivial, n = 2 + (n-2). COMPILES.
- Part 2: FALSE. Even/odd partition A₁=evens, A₂=odds gives
  A₁+A₁ = evens (gap 2, syndetic), A₂+A₂ = evens (syndetic). Both syndetic.
- Verdict: unprovable (statement false for this A). SCORE 0.0.

## Candidate 2: A = {n | Even n ∨ ∃ k, n = 2^k+1} (evens ∪ lacunary odds)
- Basis: parity split — even n = 2+(n-2); odd n = 3+(n-3). COMPILES.
- Part 2: FALSE. Survives plain even/odd but not mod-4 mixing:
  A₁ = {0 mod 4} ∪ {lacunary odds}, A₂ = {2 mod 4}. Then A₁+A₁ ⊇ {0 mod 4}
  syndetic, A₂+A₂ ⊆ {0 mod 4} syndetic. Both syndetic.
- Lesson: ANY eventually-periodic/rich-residue A dies to a modular mixing
  attack — the adversary refines by a finer modulus. Need self-similar/lacunary
  structure at ALL scales (the real Erdős construction).

## Candidate 3: block construction A = {0,1,2,3} ∪ ⋃ⱼ [4^j, 2·4^j]
- Basis: FULLY COMPILES. Use k = Nat.log 4 n; n ∈ [4^k, 4·4^k). Three subcases:
  n≤2·4^k → 0+n; n≤3·4^k → 4^k+(n-4^k); else → (n-2·4^k)+2·4^k.
  (Nat.pow_log_le_self, Nat.lt_pow_succ_log_self, pow_succ, by_cases not le_or_lt.)
- Part 2: STILL FALSE. Blocks are INTERVALS → arithmetically rich. Color each
  block B_j by parity: A₁ = even elements, A₂ = odd. Then A₁+A₁ ⊇ {all evens
  in [2·4^j,4·4^j]} for every j (even+even over an interval = every even in the
  doubled interval, gap 2) ⇒ A₁+A₁ ⊇ evens of [8,∞), syndetic. Same for A₂+A₂.
  Both syndetic.
- KEY LESSON: lacunary *placement* of blocks is not enough; each block being a
  full interval is 2-colorable into two AP-like halves whose self-sums are both
  syndetic. The blocks themselves must be arithmetically rigid (non-interval),
  but then B_j+B_j is not a full interval and the BASIS breaks. This tension is
  the heart of Erdős 741(ii); resolving it needs a non-arithmetic multi-scale
  construction beyond a short Lean proof.

## Candidate 4: asymmetric residue A = {n | n % 3 ≠ 2}
- Basis: COMPILES (case n%3: 0/1 → 0+n; 2 → 1+(n-1)).
- Part 2: FALSE. Both kept residue classes (0,1 mod 3) are arithmetically rich;
  even/odd refinement (and finer mod refinement) makes both sumsets syndetic.
- Lesson: breaking parity symmetry with an odd modulus does not help.

## Candidate 5: rigid summands A = {0,1,2,3} ∪ {squares}
- Basis: UNPROVABLE — 2 sorries. NOT a basis: consecutive squares are 2m+1
  apart (>3 for m≥2), so 4 consecutive integers can all avoid squares, and
  {square}+{square} misses a positive proportion of ℕ (sums of two squares have
  density ~ n/√(log n)). Rigid summand sets are too sparse to be a basis.
- Lesson: the rigidity that would defeat the coloring attack also destroys the
  basis property — the exact tension noted in Candidate 3.

## Candidate 6: A = evens ∪ {odd squares}
- Basis: COMPILES (even n = 0+n; odd n = 1 + (n-1), 1 = (2·0+1)²).
- Part 2: FALSE. Even part = ALL evens; adversary refines it mod 4
  (A₁ = {0 mod 4} ∪ odd-squares, A₂ = {2 mod 4}); both sumsets ⊆ {0 mod 4},
  syndetic. The rigid odd part is irrelevant once one class is rich.

## OVERALL CONCLUSION
Every construction with ANY arithmetically-rich (positive-density, AP-containing)
colour class dies to a modular/even-odd *refinement* of that class. The only way
to deny the adversary a rich class is to make A lacunary/rigid at all scales — but
that destroys the order-2 basis property (Candidate 5). Erdős 741(ii) resolves
this tension with a non-arithmetic, multi-scale construction whose part-2 proof is
the deep content; it is NOT reducible to a short Lean proof in this budget.
HONEST STATUS: basis half fully formalized (block construction); part 2 = sorry.
SCORE 0.0. Did not fabricate, did not weaken the statement.


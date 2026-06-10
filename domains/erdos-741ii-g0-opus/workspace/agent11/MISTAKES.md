# MISTAKES — agent11 — Erdős 741(ii) cold start

## Structural lemma (proved by hand, guides all candidates)
Any winning A **must have unbounded gaps**. If A ⊇ [m,∞) (a full tail), the
parity split A₁=evens∩A, A₂=odds∩A makes both A₁+A₁ and A₂+A₂ ⊇ {large evens},
both syndetic → condition 2 fails. So A must be a basis of order 2 (needs
density ≳√N) AND have unbounded gaps (anti-density). This tension is the core.

## Candidate 1: A = univ (ℕ)
- Condition 1 (basis): TRIVIAL, n = 0 + n. Compiles.
- Condition 2: FALSE. Parity split → both sumsets = evens (syndetic). Dead.
- Result: SORRY_COUNT 1, SCORE 0.0. Mathematically cannot satisfy cond 2.

## Candidate 2: A = {evens} ∪ {1}
- Condition 1 (basis): PROVED in Lean. even n=0+n; odd n≥5 = 1+(n-1). Compiles.
- Condition 2: FALSE. Split A₁ = (evens ≡2 mod4)∪{1}, A₂ = evens ≡0 mod4.
  A₁+A₁ ⊇ multiples of 4 (syndetic); A₂+A₂ = multiples of 4 (syndetic). Both. Dead.
- Result: SORRY_COUNT 1, SCORE 0.0.
- LESSON: a single odd "filler" doesn't break the mod-4 sub-split of the evens.

## Candidate 3: A = {multiples of 3} ∪ {1,2}
- Condition 1 (basis): PROVED. n≡0:0+n; n≡1:1+(n-1); n≡2:2+(n-2). Compiles.
- Condition 2: FALSE. Same residue obstruction one level up: split the multiples
  of 3 by mod 9 (≡0 vs ≡{3,6}) → both self-sumsets syndetic mod 9. Dead.
- LESSON: Any A built from a residue class / containing a syndetic infinite
  arithmetic core dies — the adversary refines into the next modulus. A WINNING
  A must be gappy on *every* scale (unbounded gaps), which kills the easy basis
  proof. This is the central tension.

## Candidate 4: A = {0,1,2,3} ∪ {powers of 2}  (GAPPY family)
- Set well-formed (compiles). Condition 1 (basis): FALSE. n=23 → 23-{0,1,2,3}=
  23,22,21,20 none a power of 2; 23-16=7∉A, 23-8=15∉A. Uncovered. SCORE 0.

## Candidate 5: A = {squares} ∪ {0,1,2,3}  (GAPPY family)
- Set well-formed (compiles). Condition 1: FALSE. n=23 is not a sum of two squares
  and 23-{0,1,2,3} ∉ squares. Uncovered. SCORE 0.

## Candidate 6: A = ⋃ₖ [4^k, 2·4^k]  (interval blocks — closest to real answer)
- Set well-formed (compiles). Condition 1: FALSE.
- within-block sums cover [2·4^k, 4^{k+1}]; cross sums I_k+I_{k+1} start at 5·4^k.
  The window (4·4^k, 5·4^k) is covered by neither (e.g. 18 for k=1). SCORE 0.
- KEY LESSON: interval blocks CANNOT be both a basis and unbounded-gap. A real
  solution needs non-interval blocks (arithmetic progressions / multi-scale
  overlap) whose CROSS-sums exactly tile the gaps — a delicate covering design,
  plus a Ramsey-type argument for condition 2. Research-level; not formalizable
  in this cold-start session.

## SUMMARY (6 distinct candidates, all oracle-tested)
Two failure modes, cleanly separated:
- Residue/tail family (C1 univ, C2 evens∪{1}, C3 mult-of-3∪{1,2}): condition 1
  easily provable, but condition 2 ALWAYS fails — adversary refines the arithmetic
  core by the next modulus so both self-sumsets become syndetic.
- Gappy family (C4 powers-of-2, C5 squares, C6 interval blocks): have the
  unbounded gaps condition 2 needs, but FAIL condition 1 (not a basis of order 2).
No simple A escapes BOTH conditions — that is precisely the problem's difficulty.
No candidate reached SCORE 1.0. Best partial: condition 1 fully machine-checked
(C2, C3, 1 sorry on the hard universal condition 2).

# agent15 — MISTAKES / candidate log (Erdős 741(ii), G0 cold)

Theorem: ∃ A ⊆ ℕ, (basis order 2 for n≥4) ∧ (every 2-partition A=A₁⊔A₂ has
¬(IsSyndetic(A₁+A₁) ∧ IsSyndetic(A₂+A₂))).

## Candidate 1 — base-5 digit set A1 = {n | all base-5 digits ≤ 2}
- BASIS: **PROVEN** in Lean (strong induction, digit-split d=(d+1)/2 + d/2,
  helper A1_mem_step via Nat.digits_def' + omega). Compiles, 0 sorry on that half.
- PARTITION: **PROPERTY IS FALSE for this A.** Adversary colors by last base-5 digit:
  A₂ = {last digit = 1}, A₁ = {last digit ∈ {0,2}}.
  A₂+A₂ ⊇ {n ≡ 2 mod 5} (1+1=2, higher digits form any value by basis) → SYNDETIC.
  A₁+A₁ ⊇ {n ≡ 0 mod 5} (0+0) → SYNDETIC.
  Both syndetic ⇒ partition property fails. So A1 cannot satisfy the theorem.
- LESSON: any "digit set" (product structure over positions) is residue-splittable;
  one coordinate (last digit) splits it into pieces each owning a full residue class.

## Candidate 2 — base-3 digit set {n | base-3 digits ≤ 1}
- BASIS: provable (same technique, d∈{0,1,2} → a,b∈{0,1}).
- PARTITION: FALSE, same residue-split flaw (last digit 0 vs 1 → res 0 and res 2 mod 3).

## Candidate 3 — A = ℕ (all naturals)
- BASIS: trivial (n = 0 + n).
- PARTITION: FALSE. Color by parity: evens+evens ⊆ evens syndetic; odds+odds ⊆ evens
  syndetic. Both syndetic. (Even mod-4 split also defeats it.)

## Candidate 4 — A = {evens} ∪ {1}
- BASIS: provable (even+even, even+1).
- PARTITION: FALSE. Split evens by mod 4: {1}∪{0 mod4} and {2 mod4}; both sumsets
  contain a full mult-of-4 progression → syndetic.

## Candidate 5 — lacunary blocks A = ⋃_k [n_k, n_k+k], n_{k+1} ≫ 2(n_k+k)
- BASIS: **FALSE.** A+A is a union of short scattered intervals with unbounded gaps,
  so it does NOT cover [4,∞). Sparse/lacunary ⇒ not a basis of order 2.
- LESSON: a basis of order 2 forces A+A cofinite ⇒ near-doubling density of blocks;
  cannot be lacunary. Sparse bases must be "spread" (≈√n density everywhere), not blocky.

## Candidate 6 — base-7 digit set {n | base-7 digits ≤ 3}
- BASIS: provable (d∈{0..6} → a,b∈{0..3}).
- PARTITION: FALSE, residue-split by last digit again.

## Synthesis
- Every periodic / digit-product / cofinite construction FAILS part 2 via a residue
  coloring (one coordinate gives each color a full residue class ⇒ both sumsets syndetic).
- Lacunary constructions FAIL part 1 (not a basis).
- The true construction must be a *spread sparse basis* whose elements' residues are
  NOT freely factorable — a genuinely irregular ≈√n-density basis. This is the
  research-level core (the known proof is ~280 lines); not reconstructed cold here.
- REAL PROGRESS BANKED: the basis (part 1) is fully formalized and reusable for any
  digit-style A; the residue-split argument prunes the entire class of "easy" A.

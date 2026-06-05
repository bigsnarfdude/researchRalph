# agent15 — LEARNINGS (Erdős 741(ii), G0 cold)

## Formalization facts (verified via oracle / lake env lean)
- Basis-of-order-2 for any "base-b digit ≤ t" set is a clean ~15-line strong-induction
  proof: split each base-b digit d into the two summands' digits; membership step is
  `Nat.digits_def' (h : 2≤b) (h0 : 0<x)` then `omega` for the `%`/`/` facts. omega
  natively discharges `(b*c+s)%b = s`, `(b*c+s)/b = c`, and the digit-recombination.
- Proven basis directions (compile, 0 sorry): base-5(≤2), base-3(≤1), base-7(≤3),
  ℕ, evens∪{1}, and the thin basis E∪2E (E = base-4 digits ≤1). See scratch_candidates.lean
  and Erdos741OAI.lean.

## Mathematical facts about the PROBLEM (the real difficulty is part 2)
- RESIDUE-SPLIT obstruction: any A with product/periodic structure (digit sets, ℕ,
  evens∪{1}) is 2-colorable so BOTH color sumsets are syndetic. Color by one coordinate
  (e.g. last digit); each color then owns a full residue class in its sumset. So ALL
  digit-set and periodic constructions PROVABLY FAIL the partition property.
- DENSITY obstruction: a basis of order 2 forces A+A cofinite, which forces near-doubling
  block density — so lacunary/blocky sparse sets are NOT bases. Sparse bases must be
  "spread" (≈√n density everywhere), e.g. the thin basis E∪2E.
- E∪2E (thin √n basis) is the surviving candidate: it resisted every residue/parity/
  last-digit attack I tried, because any coordinate-restricted subclass has its sumset
  confined to base-4-digit-≤2 patterns, which are non-syndetic (unbounded gaps near 4^k-1).
  Whether SOME cleverer 2-coloring defeats it is the open research core (~280-line proof).

## Status
- Part 1 (basis): SOLVED and formalized for E∪2E.
- Part 2 (partition/non-syndetic): genuine research-level combinatorics; not closed cold.
  SCORE remains 0.0 honestly (1 sorry on the partition half). No statement weakening,
  no fabrication.

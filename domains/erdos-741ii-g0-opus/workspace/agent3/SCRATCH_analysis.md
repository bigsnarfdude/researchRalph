# agent3 analysis — erdos_741_ii

## Status
- Condition 1 (basis of order 2 for n≥4): PROVEN for A = Set.Ici 2 (a=2, b=n-2). Compiles.
- Condition 2 (no 2-coloring keeps both monochromatic sumsets syndetic): OPEN.

## Why condition 2 is hard (verified obstructions)
- A = Set.Ici 2 / A = ℕ: FALSE. even/odd split → both sumsets = evens (syndetic).
- A = {2} ∪ odds: FALSE. split odds mod 4 → both monochromatic sumsets ≡2 mod4 (syndetic).
- General: ANY union-of-residue-classes construction is defeated by a finer residue
  sub-partition that makes both color classes' sumsets cover a residue class.
  ⇒ construction MUST be aperiodic.

## Counting argument fails
Thin basis |A∩[0,N]| = Θ(√N). If both A_i+A_i syndetic (const C_i) then
|A_i+A_i ∩[0,2N]| ≥ 2N/(C_i+1), and |A_i+A_i| ≤ |A_i∩[0,2N]|²/2 ≤ K²N.
⇒ only forces C_i ≥ 2/K². No contradiction. Density/counting cannot prove cond 2.

## What is actually needed
Syndetic = BOUNDED GAPS (stronger than positive density). Need a construction where
every monochromatic same-sum A_i+A_i has UNBOUNDED gaps, while cross-sum A₁+A₂ carries
the basis coverage. Classical route: A = ⋃_k (translate of a Sidon/B_2 block), blocks
positioned so pairwise (cross) block-sums tile [N,∞) (basis), but each block is Sidon so
within-block monochromatic sums are sparse ⇒ gappy for every 2-coloring.

This is research-level; full Lean formalization ≈ hundreds of lines (Sidon theory + gap
bookkeeping). Not reliably completable in a fast edit/compile loop.

# agent10 — candidate log (erdos-741ii-g0, cold)

Theorem needs A ⊆ ℕ with:
- (P1) basis order 2: every n≥4 is a+b, a,b∈A
- (P2) rigidity: NO partition A=A₁⊔A₂ makes BOTH A₁+A₁ and A₂+A₂ syndetic

P2 is the entire difficulty. Six distinct constructions tested via run.sh below.

## Central obstruction (the real finding): a dichotomy
- If A is DENSE / has arithmetic-progression structure → P1 holds, but the
  adversary 2-colours A along a finer modulus so BOTH parts' sumsets contain
  an AP and are syndetic ⇒ P2 FAILS. (parity = mod 2, then mod 4, mod 9, ...)
- If A is THIN (density → 0) → no AP structure, but A+A is no longer cofinite
  ⇒ P1 FAILS (not a basis).
- The solution lives EXACTLY at the √n-density threshold with non-AP structure
  (a minimal basis of order 2). Density counting alone gives no P2 contradiction:
  if both sumsets are syndetic with constant C, each part is only forced to have
  ≳ √(n/C) elements — consistent with a √n basis, no contradiction. The genuine
  construction needs finer combinatorial structure I could not formalize cold.

## Candidates (each tested with `bash run.sh`)
1. A = {0} ∪ {n≥4}            — P1 PROVED. P2 FALSE: parity (A₁=0∪evens, A₂=odds), both sumsets ⊆ evens, syndetic.
2. A = univ                   — P1 PROVED. P2 FALSE: parity split, evens+evens & odds+odds both = evens, syndetic.
3. A = evens ∪ {1}            — P1 PROVED (parity case-split, odds via the elem 1). P2 FALSE: parity fails (one odd elem) but mod-4 split works → both syndetic.
4. A = {n | n%3 ∈ {0,1}}      — P1 PROVED (n≡2 via 1+(n-1)). P2 FALSE: union of mod-3 APs, mod-9 split → both syndetic.
5. A = {0,1,2} ∪ 3ℕ          — P1 PROVED (base-3 digit basis, n = (n−n%3)+n%3). P2 FALSE: digits+multiples are AP-structured, splittable. *** kept as final artifact (cleanest P1, single P2 sorry).
6. A = {0,1} ∪ {2^k}          — P1 FALSE: thin, 11 has no representation (not a basis). Confirms the thin horn of the dichotomy.

## Outcome
SCORE=0.0. P1 solved for 5/6; P2 (the hard half) not formalized. NOT fabricated,
statement NOT weakened, main theorem honestly carries one sorry on P2.
This is genuine cold-start signal: the construction is research-level and the
P1/P2 dichotomy above is, I believe, the correct obstruction map for it.

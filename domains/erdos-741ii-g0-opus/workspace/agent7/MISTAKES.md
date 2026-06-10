
## agent7 — 6 distinct construction attempts (cold, no construction given)

Target: A ⊆ ℕ, basis of order 2 for n≥4, AND every 2-coloring A=A₁⊔A₂ has at least
one of A₁+A₁, A₂+A₂ NON-syndetic (unbounded gaps).

| # | Construction A | Part 1 (basis) | Part 2 (rigidity) | Refuting coloring / obstruction |
|---|----------------|----------------|-------------------|----------------------------------|
| 1 | univ (all ℕ)            | PROVEN (0+n) | FALSE | evens/odds → both sumsets = evens, syndetic |
| 2 | {n ≠ 1}                 | PROVEN (2+(n-2)) | FALSE | refine residues mod 4 |
| 3 | evens ∪ {1}             | PROVEN | FALSE | split evens by mod 4 → both sumsets APs |
| 4 | {0,1,2} ∪ 3ℕ           | PROVEN (residues mod 3) | FALSE | split 3ℕ by mod 6 → each color ⊇ 6ℕ |
| 5 | {n%4 ∈ {0,1}}          | FALSE (n≡3 mod4 unreachable) | n/a | a set omitting a residue class is never a basis of order 2 |
| 6 | {0,1,2,3} ∪ {2^k}      | FALSE (e.g. 23 not a+b) | n/a | thin geometric sets are not bases of order 2 |

### Core lesson
- Every EXPLICITLY definable eventually-PERIODIC / cofinite set (1–4) is refuted on
  part 2: its color classes can be chosen as sub-arithmetic-progressions, and an AP's
  self-sumset is again an AP = syndetic. So both colors end up syndetic.
- To avoid sub-AP refutation, A must be APERIODIC. But aperiodic *and* a basis of
  order 2 forces density ~√n (thin basis). Candidates 5,6 show that thinning out
  (residue-omitting or geometric) destroys the basis property unless done carefully.
- A genuine answer = thin (~√n) aperiodic basis. Its part-1 proof is non-elementary
  and its part-2 (rigidity) proof needs real additive combinatorics (block/pigeonhole
  arguments over growing-gap scales). Not reachable cold in this session.
- No Sidon/unique-representation trick exists: a Sidon set has ≤√n+O(n^¼) elements up
  to n, too few to be a basis of order 2 (needs ≥√(2n)). So rigidity cannot come from
  unique representations — every basis of order 2 has many multiply-represented n.

### Honest status
SCORE=0.0. Main theorem NOT proved. Part 1 (basis) fully proven for construction #4;
part 2 left as a single `sorry` with a comment stating #4 is itself false for part 2.
No fabrication, statement not weakened.

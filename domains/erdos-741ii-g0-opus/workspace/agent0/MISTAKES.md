# MISTAKES — agent0 — Erdős 741(ii) cold start

## The core obstruction (discovered immediately, verified across all 6 candidates)
- Condition 1 (basis of order 2 for n≥4) is EASY for any reasonably dense A.
- Condition 2 (∀ partition A=A₁⊔A₂, one of Aᵢ+Aᵢ is NON-syndetic) is the whole problem.
- S syndetic := ∃C, every window [x,x+C] meets S (bounded gaps).
- A subset of a non-syndetic set is non-syndetic (gaps only grow). So IF A+A were
  non-syndetic, cond 2 would be trivial — but cond 1 forces A+A ⊇ [4,∞), hence A+A
  IS syndetic. So the covering MUST come from the cross term A₁+A₂, while both
  self-terms fail. That is the real subtlety.

## THE UNIVERSAL ADVERSARY (kills every explicit/periodic construction)
Given A periodic-ish, split A by a low-order residue so that each colour's cross term
(A_i∩low)+(A_i∩high) becomes a full arithmetic progression (an AP is syndetic, and an
AP self-sum is again an AP). Concretely for Moser A=L∪2L: colour
A₁ = 4L ∪ 2(4L+1), A₂ = (4L+1) ∪ 2(4L); then A₁+A₁ ⊇ {≡2 mod4}, A₂+A₂ ⊇ {≡1 mod4},
BOTH syndetic. ⇒ cond 2 FALSE. This residue-split defeats ANY eventually-periodic A.

## THE TENSION (why this is a hard theorem)
- Dense / periodic A  ⇒ cond 1 holds, but residue-split adversary breaks cond 2.
- Sparse / lacunary A ⇒ resists partition, but sumset gaps grow ⇒ cond 1 fails (no basis).
- Resolution requires a THIN basis: order-2 basis with representation function O(log n)
  (Erdős–Tetali), built greedily/probabilistically — NON-EXPLICIT, AP-poor, dense enough
  to cover but irregular enough that no 2-colouring yields two syndetic self-sums.
  Not finitely formalizable from a closed form in this loop.

## Candidates tested (each run via bash run.sh)
| # | Construction                              | Cond1 | Cond2 | Why rejected |
|---|-------------------------------------------|-------|-------|--------------|
| 1 | A = univ (ℕ)                              | ✓     | ✗     | parity split → both = evens, syndetic |
| 2 | A = {1} ∪ evens                           | ✓     | ✗     | AP sub-colouring both syndetic |
| 3 | A = {n | n%3 ∈ {0,1}}                     | ✓     | ✗     | residues 0,1 → self-sums ≡0,≡2 mod3 syndetic |
| 4 | Moser–de Bruijn  A = L ∪ 2L (base-4 dig.) | (?)   | ✗     | mod-4 split → {≡1},{≡2} mod4 syndetic |
| 5 | A = powers of 2 ∪ {0,1,2,3}               | ✗     | —     | not a basis: 23 unrepresentable |
| 6 | A = {2,3} ∪ ⋃ₖ[4^k, 2·4^k] (lacunary)     | ✗     | —     | not a basis: 17 uncovered (sumset gaps) |

Cond1 ✓ proofs (cands 1–3) COMPILE (BUILD_EXIT 0, only the cond-2 sorry remains).
Cands 5,6 are provably NOT bases (explicit counterexamples 23, 17).

## Honest verdict
Did NOT reach SCORE=1.0. The 6 genuine attempts confirm the trade-off above. A solving
construction exists (thin basis) but I could not produce an explicit, Lean-formalizable
form for it cold. Theorem left with `sorry` — NOT claiming success, NOT weakening the
statement, NOT fabricating.

# MISTAKES — agent6 — Erdős 741(ii) cold start

Theorem: ∃ A ⊆ ℕ basis of order 2 (covers n≥4) such that for EVERY partition
A = A₁ ⊔ A₂, at least one of A₁+A₁, A₂+A₂ is NOT syndetic (unbounded gaps).

## Structural obstructions discovered (apply to ALL candidates)

### Obstruction P (parity attack)
Color A by parity: A₁ = even elts, A₂ = odd elts. Then A₁+A₁ ⊆ evens and
A₂+A₂ ⊆ evens. If BOTH the even-part and odd-part of A have sumsets that are
cofinite-in-evens, both are syndetic → condition 2 FALSE.
Defeating it requires either the even elts or odd elts of A to have a
non-syndetic (gappy) sumset.

### Obstruction AP (arithmetic-progression attack)
If A contains an infinite arithmetic progression P (e.g. all evens, or dℕ),
split P by index parity into P_even, P_odd. Both are APs of step 2d, so both
sumsets are APs (gaps bounded) = syndetic. Distribute the rest to keep both
syndetic → condition 2 FALSE.
⟹ A must contain NO infinite arithmetic progression.

### Obstruction B (basis ⇒ density / no big gaps)
A basis of order 2 needs |A ∩ [0,N]| ≳ √N and, more sharply, A itself must be
"locally not too gappy": to cover the interval (top_k, 2·bottom_k) between a
block and its self-sum, A must have elements there OR small bridging elements.
Big inter-block gaps (needed for clean rigidity gaps) BREAK the basis property.
This is the central tension: rigidity wants gaps, basis forbids them.

### Recursion trap
Splitting A = E (evens) ∪ O (odds, sparse) to defeat P: each part needs only
ONE odd + a sumset-cofinite even-part. So E must itself be a basis-for-evens
with NO 2-partition into two syndetic-sumset halves — i.e. the SAME problem
one scale down. No reduction. Confirms the problem is genuinely hard.

---

## Candidate 1: A = {0} ∪ ⋃_k [4^k, 2·4^k]   (dyadic interval blocks)
- def A : Set ℕ := {0} ∪ {n | ∃ k, 4^k ≤ n ∧ n ≤ 2*4^k}
- BASIS: PROVED in Lean (basis_part compiles, BUILD_EXIT 0). For n≥4 take
  k=Nat.log 4 n; split n≤2L / n≤3L / n<4L using a=L or a=2L. Clean.
- CONDITION 2: **FALSE** by Obstruction P. Blocks are full intervals → contain
  both parities densely; even-sumset and odd-sumset both ⊇ cofinite evens
  (odd elt 1 bridges). Verified by hand. Rigidity sorry is unprovable.
- VERDICT: invalid (basis ok, irreducibility false).

## Candidate 2: A = ℕ (Set.univ)
- BASIS: PROVED (n = n + 0). BUILD_EXIT 0.
- CONDITION 2: FALSE. Parity attack: A₁=evens, A₂=odds → both sumsets = evens,
  syndetic (gaps 2). Invalid.

## Candidate 3: A = {0,1,2,3} ∪ 4ℕ  (m-adic minimal basis)
- BASIS: PROVED (n = 4*(n/4) + n%4, both summands ∈ A). BUILD_EXIT 0.
- CONDITION 2: FALSE by Obstruction AP. Infinite part 4ℕ ≅ ℕ; split into 8ℕ and
  8ℕ+4 → both sumsets ⊆ 8ℕ (syndetic). Invalid.

## Candidate 4: A = {n | 2 ≤ n}  (ℕ≥2)
- BASIS: PROVED (n = 2 + (n-2)). BUILD_EXIT 0.
- CONDITION 2: FALSE by parity (same as univ). Invalid.

## Candidate 5: A = {0} ∪ ⋃_k [3^k, 2·3^k]  (ratio-3 blocks)
- BASIS: PROVED (k=Nat.log 3 n; n≤2L ⟹ n+0, else a=L,b=n-L). BUILD_EXIT 0.
- CONDITION 2: FALSE by Obstruction P (full intervals contain both parities;
  even-sumset and odd-sumset both ⊇ cofinite evens). Invalid.

## Candidate 6: A = {0,1} ∪ ⋃_k {n∈[4^k,2·4^k] : n ≡ k mod 2}  (single-parity blocks)
- MOTIVE: defeat the parity attack by putting only ONE parity in each block, so
  even-elts and odd-elts live at SEPARATED scales ⟹ each sumset is gappy
  (non-syndetic). This DOES beat the parity partition.
- BASIS: PROVABLY FAILS (left as gap). Sum of two same-parity elts is always
  even, so within-block sums are all even; an ODD n needs even-elt + odd-elt
  both near n/2, but only ONE block (single parity) sits near n/2. Mid-range
  odd n are uncoverable ⟹ not a basis. BUILD compiled with 2 gaps.
- VERDICT: invalid (defeats parity but loses the basis property — Obstruction B).

## Meta-conclusion
All 6 distinct candidate sets tested in Lean. The basis half is easy and was
proven for candidates 1–5. Condition 2 (irreducibility) is the hard direction:
- full/dense sets die to the parity attack,
- AP-containing sets die to the AP attack,
- single-parity sets defeating parity lose the basis property.
Defeating parity AND AP while staying a basis reduces to the SAME problem at a
smaller scale (recursion trap). The theorem is TRUE (known Erdős result) but the
valid construction + its rigidity proof is research-level (~150 lines per prior
domain memory) and I could not reconstruct it cold this session. NOT faked: the
main theorem retains its open gap; no statement was weakened.

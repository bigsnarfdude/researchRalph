# MISTAKES — agent9 (erdos-741ii cold)

## Core obstruction discovered
Part 2 requires: for EVERY 2-partition A=A₁⊔A₂, NOT both A₁+A₁ and A₂+A₂ syndetic
(syndetic = bounded gaps). If A contains any syndetic set, the even/odd (or mod-m)
residue split makes BOTH sumsets syndetic → part 2 FALSE. Therefore A must be a
THIN basis (density → 0) relying on large+large sums. But thin ⟹ the basis proof
(A+A ⊇ [4,∞)) is itself hard (E+E digit-decomposition style), and the partition
proof is a research-level structural argument. This is the fundamental tension.

## Candidate 1: A = univ (ℕ)
- Basis: TRIVIAL ✓ (n = 0 + n). Compiles.
- Part 2: FALSE. Split evens/odds → evens+evens=evens (syndetic), odds+odds=evens (syndetic). Both syndetic.
- Verdict: REJECTED (part 2 false).

## Candidate 2: A = {0} ∪ {n | 4 ≤ n}
- Basis: TRIVIAL ✓ (n = 0 + n for n≥4). Compiles.
- Part 2: FALSE. A is syndetic; split {0}∪evens vs odds → both sumsets ⊆ evens, syndetic.
- Verdict: REJECTED (part 2 false; syndetic A).

## Candidate 3: A = {0,1,2,3} ∪ {4k : k≥1}  (syndetic, density 1/4)
- Basis: EASY ✓ (n = (n mod 4) + 4⌊n/4⌋).
- Part 2: FALSE. Split by k parity → A₁+A₁ covers residues 0..3 mod 8 (gap≤5, syndetic),
  A₂+A₂ ⊆ 8ℕ (syndetic). Both syndetic.
- Verdict: REJECTED (syndetic A always loses to a residue split).

## Candidate 4/4': binary E∪O thin basis  (E = base-4 {0,1}-digit numbers, O = 2·E)
- Definitions typecheck; BASIS now FULLY PROVED (n = e + o, splitting each base-4 digit
  d = d%2 + 2·(d/2); e∈E, o∈O). Compiles, 1 sorry (part 2 only).
- Survives every simple coloring attack:
  * E vs O split → E+E = base-4 {0,1,2}-digit set, unbounded gaps near [3·4^{k-1},4^k) → not syndetic.
  * any A₂ ⊆ O → A₂+A₂ ⊆ 2(E+E) → not syndetic.
  * even/odd parity → odd part's sumset ⊆ 2+4(E+E) → not syndetic.
- Digit-shift attack (which killed candidate 6) does NOT port: A is a UNION, not shift-closed.
- Verdict: STRONGEST candidate. Part 2 unproven (research-level structural argument). NOT refuted.

## Candidate 5: interval-block A = ⋃_k [2^(2k), 2^(2k)+2^k]
- Definitions typecheck. BASIS is FALSE: block self-sums [2^(2k+1), 2^(2k+1)+2^(k+1)] leave
  huge uncovered gaps between blocks; not a basis of order 2.
- Verdict: REJECTED (not a basis).

## Candidate 6: base-3 {0,1}-digit thin basis  A = {ofDigits 3 L : digits ≤ 1}
- BASIS FULLY PROVED (n = x+y, digit d split into min d 1 and d - min d 1; both digits ≤1).
- Part 2 is FALSE — REFUTED by the digit-shift coloring:
  color by lowest base-3 digit. A₁ = {d₀=0} = 3A ⇒ A₁+A₁ = 3(A+A) = 3ℕ (syndetic).
  A₂ = {d₀=1} = 1+3A ⇒ A₂+A₂ = 2+3ℕ (syndetic). BOTH syndetic.
- LESSON: any single digit-closed basis with A+A=ℕ that is closed under base-multiplication
  (k·A ⊆ A) dies to the lowest-digit coloring. The construction must be a union of
  incompatible structures (→ E∪O) to break shift-closure.

## Summary
6 distinct constructions tested via run.sh. Provably FALSE for part 2: 1(univ), 2({0}∪[4,∞)),
3({0,1,2,3}∪4ℕ), 6(base-3 digits — all syndetic or shift-closed). Not a basis: 5(blocks).
Surviving candidate: 4' (E∪O), basis proven, part-2 crux open. No construction reached SCORE=1.0.
The theorem was NEVER weakened and no sorry was passed off as a proof.

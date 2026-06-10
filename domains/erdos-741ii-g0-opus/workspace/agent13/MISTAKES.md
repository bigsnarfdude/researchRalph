# MISTAKES.md — agent13 (Erdős 741ii, cold start)

The hard obligation is the PARTITION property: for every 2-partition A=A₁⊔A₂,
NOT both A₁+A₁ and A₂+A₂ are syndetic. Basis (n≥4 = a+b) is the easy part.

## Candidate 1: A = univ (ℕ)
- Basis: trivial (n = n+0). COMPILES.
- Partition: FALSE. Split evens/odds: evens+evens=evens (syndetic), odds+odds=evens
  (syndetic). Both syndetic → obligation violated. Unprovable.
- Lesson: A=univ rejected. Need structured A.

## Candidate 2: A = {evens} ∪ {1}
- Basis: PROVED (even n=n+0; odd n=(n-1)+1). COMPILES.
- Partition: FALSE. Split A₁={1}∪{4k}, A₂={4k+2}: A₂+A₂={0 mod4}=4ℕ syndetic;
  A₁+A₁ covers residues {0,1 mod4}∪{2} syndetic (gap≤3). Both syndetic → violated.
- Lesson: any finite-union-of-APs is splittable into syndetic pieces. Need a
  non-AP (digit/multiplicative) obstruction — this is the heart of Erdős 741(ii).

## Candidate 3: A = E ∪ O  (E=bits at even positions, O=bits at odd positions = 2E)
- = base-4 digit-{0,1} set and its double. n = e(n)+o(n) (disjoint bit supports).
- Basis: **PROVED in Lean** via strong-induction bit recursion
  (e:=n%2+2·o', o:=2·e', using Nat.testBit_succ). Only partition sorry remained.
- Partition: **FALSE — refuted.** Color E by bit0, O by bit1:
  A₁=E∩{bit0=0} ∪ O∩{bit1=0}, A₂=E∩{bit0=1} ∪ O∩{bit1=1}.
  Then every n≡0 mod4 has e(n)∈E∩{bit0=0}, o(n)∈O∩{bit1=0} ⇒ n∈A₁+A₁.
  Every n≡3 mod4 ⇒ n∈A₂+A₂. BOTH syndetic. Obligation violated.
- Lesson: a "free lowest digit" lets the adversary 2-color by residue. The correct
  construction must COUPLE digit positions so no residue-coloring succeeds.

## Candidate 4: A = {n : all base-5 digits ≤ 2}
- Basis: TRUE (digit d≤2 + d≤2 ≤4<5, no carry ⇒ A+A=ℕ). Construction compiles.
- Partition: FALSE. Residue color A₁={low digit 0}, A₂={low digit 1,2}:
  A₁+A₁⊆{5∣n}, A₂+A₂⊇{low digit∈2,3,4}; both syndetic. Free low digit again.

## Candidate 5: A = {0,1,2,3} ∪ ⋃ₖ [5ᵏ, 2·5ᵏ)  (interval blocks)
- Construction compiles. Basis: FALSE. Range [4·5ᵏ,5·5ᵏ) uncovered: same-block
  sums reach <4·5ᵏ, cross-block sums start at 5ᵏ+5ᵏ⁺¹=6·5ᵏ. Not a basis.

## Candidate 6: A = {n : all base-3 digits ∈ {0,1}}
- Basis: TRUE (single set; digit 0/1+0/1 = 0,1,2 covers base-3, no carry ⇒ A+A=ℕ).
- Partition: FALSE. Residue color by low digit (0 vs 1): A₁+A₁⊆{3∣n},
  A₂+A₂⊆{low digit 2}; both syndetic. Free low digit again.

## UNIFYING FINDING
Every construction with a "free low digit" — univ, finite unions of APs, and
base-b digit-sets in ANY base (b=3,4,5) — is REFUTED by a residue-coloring attack
that makes BOTH color classes syndetic. The true Erdős-741(ii) basis must couple
digit positions globally so no residue/periodic 2-coloring works. That coupled
construction (memory hints "Q=5ᵏ") + its ~283-line partition proof was NOT
reconstructible cold within budget. Basis half is solved (cand. 3, fully Lean-proved);
partition half is the genuine open research content. Final SCORE=0.0 (honest).

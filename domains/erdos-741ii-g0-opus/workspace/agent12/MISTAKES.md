# agent12 — Erdős 741(ii) cold-start attempts

Theorem: ∃ A ⊆ ℕ, (basis of order 2 for n≥4) ∧ (for every partition A=A₁⊔A₂,
NOT both A₁+A₁ and A₂+A₂ are syndetic / bounded-gap).

cond1 = additive basis of order 2.  cond2 = partition-fragility (the hard universal part).

## Core obstruction discovered
- cond1 (basis) forces density |A∩[0,N]| ≳ √N.
- cond2 (fragility) must defeat EVERY 2-coloring. The decisive attacks:
  - **Parity / mod-m attack**: color A by residue. Then A₁+A₁, A₂+A₂ ⊆ a fixed residue class.
    For a FULLY periodic / interval-dense A both cover their class with bounded gaps ⇒ both
    syndetic ⇒ cond2 FALSE (this is why univ, evens∪{1}, {0,1}∪tail all fail).
  - CORRECTED CONCLUSION: what defeats these attacks is MULTIPLICATIVE LACUNARITY — A having
    growing gaps (A∩(c·4^k, 4^{k+1})=∅) — NOT the absence of intervals per se. Both the
    doubling-block sets (C4/C5) and the digit set E∪O (C6) have this lacunarity and survive
    parity. My initial "must have no intervals" was an over-simplification (see C4/C5 fixes).
- viable A = a lacunary (≈√N density) basis: either doubling blocks or digit-type E∪O.

## Candidates tested (all run via `bash run.sh`)

### C1: A = univ (ℕ)
cond1 ✓ trivial (n=2+(n-2)). cond2 FALSE: evens/odds partition ⇒ both sumsets = evens (syndetic).
Non-solution. SORRY_COUNT=1 (cond2 sorry, unprovable since false).

### C2: A = {Even} ∪ {1}
cond1 ✓ (even n=2+(n-2), odd n=1+(n-1)). cond2 FALSE: A₂=multiples of 4 (A₂+A₂=4ℕ syndetic),
A₁={1}∪(2 mod 4) (A₁+A₁ ⊇ 4ℕ ∪ odds, syndetic). Non-solution.

### C3: A = {0,1} ∪ [4,∞)
cond1 ✓ (n=0+n). cond2 FALSE (parity attack on the tail interval). Non-solution.

### C4: A = ⋃_k [4^k, 2·4^k]  (lacunary doubling blocks, block0=[1,2])
CORRECTION (earlier claim was WRONG): this IS a basis — numerically NO non-representable n
in [4,500). My "19 unrepresentable" was an arithmetic error (17+2=19, both ∈A). cond1 holds.
And the parity attack does NOT simply kill it: odd+odd self-sum develops growing gaps because
A has multiplicative gaps (A∩(2·4^k,4^{k+1})=∅). So C4 is a viable SHAPE; cond2 (all colorings)
is still the hard blocker. (Prior agent4 used A={0}∪⋃[4^k,2·4^k] with a cold-provable basis.)

### C5: A = ⋃_k [4^k, 3·4^k]  (ratio 3) — analyzed, is a basis
Also a basis. CORRECTION to my parity reasoning: what defeats the parity/mod attacks is
MULTIPLICATIVE LACUNARITY (big gaps in A), NOT absence of intervals. My earlier "intervals ⇒
parity kills" was too hasty. Lacunary block sets survive parity for the same reason E∪O does.

### C6: A = base-4 digit perfect basis  E ∪ O   (CHOSEN construction)
A = {n | (∃L, all digits ≤1, n=ofDigits 4 L) ∨ (∃L, all digits ∈{0,2}, n=ofDigits 4 L)}.
Sparse (√N density), no intervals. Survives parity attack (odds of A = {1,5,17,21,65,...}
sparse ⇒ odd-self-sum non-syndetic). Correct SOLUTION SHAPE.
- **cond1 FULLY PROVEN.** Decomposition n = e+f with e = ofDigits 4 ((digits 4 n).map(·%2))
  ∈ E, f = ofDigits 4 ((digits 4 n).map(2·(·/2))) ∈ O. Proven via:
  - membership: map entries are %2≤1 and 2·(d/2)∈{0,2} (using digits_lt_base for d<4);
  - e+f=n: list induction lemma ofDigits 4 (map %2 L) + ofDigits 4 (map 2(·/2) L)=ofDigits 4 L,
    each cons step closed by omega (d%2+2·(d/2)=d, no carry), then Nat.ofDigits_digits.
  Numerically verified basis (no non-representable n in [4,200)) and gap A∩[48,64)=∅.
- **cond2 NOT proven.** Genuine research-level theorem: fragility for ALL 2-colorings.
  The problem is now REDUCED to exactly this single fragility lemma (1 sorry).

## Status
SCORE=0.0, 1 sorry. cond1 fully proven for the correct construction; problem reduced to the
single fragility theorem (cond2), which is research-level and not formalizable cold in this
budget. Honest non-success — main theorem NOT claimed, never fabricated, never weakened.
6 distinct constructions genuinely attempted and tested via run.sh.

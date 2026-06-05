
## agent4 — rigorous structural findings on the rigidity half (gen g05)

FACT 1 (basis ⇒ block-length ≥ gap). For any interval-union basis A = ⋃[L_k,R_k]:
covering A+A forces 2R_k ≥ L_k+L_{k+1}, hence gap (L_{k+1}-R_k) ≤ block-length (R_k-L_k).
So unbounded gaps ⟹ unbounded block lengths.

FACT 2 (interval constructions are NOT rigid). With unbounded blocks, the PARITY 2-coloring
(A₁=even elts, A₂=odd elts) makes BOTH A_i+A_i ⊇ {all large evens} (via even+even and odd+odd,
covered across blocks), so both are syndetic (C=2). Defeats every interval-union basis.

FACT 3 (digit-closed constructions are NOT rigid). For A = base-3 "no digit 2" set
(inductive: 0∈A, x∈A⇒3x,3x+1∈A), which IS a clean basis (strong-induction proof, COMPILES),
the LAST-DIGIT coloring kills it: A₁=A∩3ℕ, A₂=A∩(3ℕ+1).
Then A₁+A₁ = 3·(A+A) = 3ℕ (syndetic), A₂+A₂ = 3ℕ+2 (syndetic). Both syndetic.
GENERAL: any A closed under affine digit maps x↦bx+r is vulnerable to residue/digit colorings.

FACT 4 (no LOCAL forcing argument can prove rigidity). The clean "missed-interval" engine —
find isolated g∈A so every n∈(g+pred,2g] has all reps through g, forcing the other color's
sumset to miss an interval of length g-pred — CANNOT apply to a basis: forcing-through-g needs A
sparse just below g, which makes A+A sparse near 2g, contradicting basis. Dichotomy is exact.
⟹ Rigidity must be a GLOBAL density/counting argument, not local forcing. (le_or_lt/eq_empty
short-proof hope is likely wrong, or those are minor tactics in a long proof.)

OPEN: need a basis A that defeats BOTH parity AND all residue/digit colorings simultaneously.
This is the genuine hard core of Erdős 741(ii). The basis lemma (basisAux) is reusable for any
digit-closed A. Next agent: do NOT retry interval or single-digit-base constructions.

## agent10 analysis (Erdos 741ii partition property)
- The basis half is trivial (A=univ: a=2,b=n-2). The WALL is the partition property:
  ∀ partition A=A₁⊔A₂, ¬(both A_i+A_i syndetic).
- KEY OBSTRUCTIONS (ruling out easy witnesses):
  * Any A self-similar mod m (incl. ℕ, mℕ, residue sets) is PARTITIONABLE: color by a
    residue/lowest digit → both self-sums land in fixed residue classes, both syndetic. INVALID.
  * Base-3 "sums of distinct powers of 3" (digits 0/1) is a clean thin basis BUT partitionable:
    A₂={digit0=0}=3A → A₂+A₂=3ℕ (syndetic); A₁={digit0=1}=1+3A → A₁+A₁=2+3ℕ (syndetic). INVALID.
  * Any A containing arbitrarily long runs (intervals) is PARTITIONABLE by parity within runs
    (evens+evens cover even sublattice → syndetic, both colors). So witness has NO long runs.
- Therefore a valid witness must be APERIODIC (defeats residue colorings) AND run-free (defeats
  parity) AND a basis AND defeat ALL 2-colorings. This is the genuine Erdős-741(ii) content.
- Counting bound is too weak: syndetic-both ⟹ each color ≥√(2N/C) elements in [0,N], total
  ~2√(2N/C) < √(2N) for C>4, consistent with a thin basis. So thinness alone ≠ enough; need
  rigidity of the covering.
- Pivot/gap arguments to break syndeticity require consecutive-target windows, which force runs
  → parity-partitionable. So gaps must come from a color's OWN structural sparsity, forced globally.

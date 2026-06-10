# agent4 — MISTAKES / construction log (Erdős 741(ii), cold start, G0-opus)

GOAL: A ⊆ ℕ, additive basis of order 2 for n≥4, AND rigid: NO 2-coloring
A = A₁⊔A₂ makes both A₁+A₁ and A₂+A₂ syndetic (bounded gaps).

## KEY STRUCTURAL THEOREM (proved on paper)
Any PERIODIC set A is NON-rigid: split A's residue classes mod m into two
nonempty groups; each part's sumset is a union of residue classes → syndetic.
⇒ ANY rigid basis must be APERIODIC with growing gaps. This forces a hard
covering argument even for the basis half, and a delicate combinatorial
argument for rigidity. This is why the friction ladder shows G0 cold = 0/12.

## CANDIDATES TESTED (all via bash run.sh)

### C1: A = Set.univ
Basis trivial (n=2+(n-2)). COMPILES. Rigidity FALSE: evens/odds split →
both sumsets = evens (gap 2, syndetic). DEAD END (periodic).

### C2: A = {1} ∪ evens   (= {n | n=1 ∨ Even n})
Basis COMPILES (n even=0+n; n odd=1+(n-1)). Rigidity FALSE: put 1 with the
mod-4≡0 evens vs the mod-4≡2 evens → both sumsets syndetic. DEAD END (periodic).

### C3: A = {n | Even (Nat.sqrt n)}   (square blocks [m²,(m+1)²), m even)
Aperiodic ✓. Basis is a real covering argument (left sorry; sketch: gap block
[m²,(m+1)²) m-odd covered by [(m-1)²,m²)+{small even blocks}). Rigidity LIKELY
FALSE: each block is a local interval, so a within-block even/odd split keeps
both sumsets syndetic in block+gap regions. Not pursued further.

### C4: A = ⋃_k [k!, k!+k]   (factorial-gap blocks)
Aperiodic ✓. FAILS BASIS: for k≥2 the gap (k!+k, (k+1)!) is unreachable —
max sum of two elements is 2(k!+k) < (k+1)!. DEAD END.

### C5: A = {n | Even (Nat.log2 n)}   (dyadic blocks [2^m,2^{m+1}), m even)
Aperiodic ✓. FAILS BASIS: 15 has no representation. Usable elements ≤15 are
{1}∪[4,8); max sum 7+7=14 < 15. DEAD END.

### C6: A = {0} ∪ ⋃_k [4^k, 2·4^k]   (doubling blocks)   ★ BEST
Aperiodic ✓, growing gaps ✓. BASIS PROVED & COMPILES (only rigidity sorry):
  - n in a block → n = 0 + n.
  - n in gap (2·4^k, 4^{k+1}) → block-k self-sum [2·4^k, 4^{k+1}] covers it;
    pick k = Nat.log 4 n, then split into a=4^k/b=n-4^k or a=n-2·4^k/b=2·4^k.
  Lean tools: Nat.pow_log_le_self, Nat.lt_pow_succ_log_self, pow_succ, omega.
  GOTCHA: `le_or_lt` is NOT in scope here — use `le_total` (matches memory note).

  RIGIDITY (the crux, UNSOLVED cold):
  - Survives even/odd adversary: A₁=evens → A₁+A₁ ⊇ all evens (uses 0+v to
    cover block interiors) = syndetic; A₂=odds → A₂+A₂ has a growing gap
    [4^k, 1.25·4^k] (odds lack 0, and block_{k-1}+block_{k-1} tops at 4^k-2
    while block_{k-1}+block_k starts at 1.25·4^k). So even/odd does NOT make
    both syndetic — good sign for rigidity.
  - But proving NO adversary works needs exact-representation analysis:
    upper gap (2.5·4^k, 4^{k+1}) is reachable ONLY from block_k+block_k, and
    4^{k+1} only from 2·4^k+2·4^k or 0+4^{k+1}. Assembling these into a
    bounded-gap contradiction for ALL colorings is the ~150-line argument the
    G1 scaffold supplied. NOT reconstructible cold within budget.
  - Also UNCERTAIN whether ratio-4 doubling is even rigid; the true Erdős-741
    construction may need finer tuning.

## OUTCOME
6 distinct constructions attempted and oracle-tested. Basis solved for the
correct construction shape (C6). Rigidity remains open — consistent with the
measured G0 cold-start friction (0/12). NOT faking success; main theorem still
carries one honest sorry (rigidity).

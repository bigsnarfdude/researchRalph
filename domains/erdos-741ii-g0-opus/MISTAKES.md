# MISTAKES — erdos-741ii-g0

## agent4 — witnesses that FAIL C2 (do not retry)
- A = univ: split evens/odds → evens+evens=evens (gap 2, syndetic), odds+odds=evens (syndetic). Both syndetic ⇒ C2 false.
- A = {0} ∪ ⋃ₖ[4ᵏ,2·4ᵏ]: C1 holds (0 fills blocks, self-sums [2·4ᵏ,4ᵏ⁺¹] tile gaps) but parity-colour each block ⇒ both halves keep sumset gaps ≤2 ⇒ both syndetic ⇒ C2 false.
- Any union of INTERVAL/AP blocks: defeated by even/odd colouring of each block (an interval halves into two APs whose self-sums stay syndetic).
- LESSON: to defeat the alternating/parity colourings the blocks must NOT be intervals/APs; need a structure where every ~half-density subset loses sumset syndeticity.

## agent5 — construction attempts defeated by residue colorings
- **What:** Tried `A = univ`, `A = {1} ∪ Evens`, and `A = {0,1,2,3} ∪ ⋃ₖ[2·4^k, 4·4^k]` (a valid positive-density basis with unbounded holes) for the ∃A in erdos_741_ii.
- **Result:** All satisfy condition 1 (basis of order 2) but FAIL condition 2. The even/odd partition (A₁=even(A), A₂=odd(A)) sends both A₁+A₁ and A₂+A₂ into the evens, and for any positive-density / bounded-local-gap basis both monochromatic sumsets come out syndetic. So `¬(syndetic ∧ syndetic)` is false for these A.
- **Lesson:** A valid A cannot be regular or positive-density. It must be a THIN (|A∩[0,N]| ~ √N), irregularly structured basis with near-unique representations, so that no 2-coloring can keep both sumsets syndetic. This is the genuine hard content of Erdős #741(ii); not solved. Left workspace with an honest single `sorry` (univ branch, explicitly flagged unprovable) rather than fabricating a proof.

## agent1 — naive block construction FAILS part 2
- what: tried A = ⋃ₖ [4^k, 2·4^k] for the full theorem.
- result: satisfies part 1, but part 2 is FALSE for this A.
  Counterexample coloring = global parity. A₁ = even elts, A₂ = odd elts.
  A₁+A₁ (even+even) = cofinite evens (syndetic). A₂+A₂ (odd+odd, bridged by 1∈A) =
  cofinite evens (syndetic). BOTH syndetic ⇒ part 2 violated.
- lesson: any THICK basis containing a full arithmetic-progression structure is
  parity-colorable into two syndetic-sumset halves. The correct A must be THIN
  (|A∩[1,N]| ~ √N) so no 2-coloring keeps both self-sumsets syndetic. This is the
  actual content of Erdős #741(ii) and is research-grade.

## agent11 — two more witnesses machine-verified to FAIL C2 (do not retry)
- what: base-3 digit-{0,1} set A = {n : base-3 digits ∈ {0,1}} (A+A=ℕ, thin ~N^0.63).
  result: FAILS. units-digit (mod 3) coloring → A₁=3A, A₁+A₁=3ℕ (gap 3); A₂=3A+1,
  A₂+A₂=3ℕ+2 (gap 3). Both syndetic. python-verified maxgap=3 each. C2 false.
- what: agent5/DESIRES binary-position thin basis A=E∪O (E=even-position bits, O=2E,
  ~2√N density). result: at scale N=2048 the element-PARITY coloring keeps both
  color-sumsets with gap ≤ 64 — i.e. it does NOT obviously satisfy C2 either. Thinness
  alone is NOT sufficient; the blocks must additionally defeat ALL 2-colorings (Sidon/
  perfect-difference structure with growing per-block forced gap). Not yet built.
- lesson: thinness is necessary but not sufficient. The unsolved core is a finite
  gadget whose sumset tiles an interval yet every 2-coloring leaves a growing gap.

## agent0 (g0-opus) — refining the prior "lessons" (two were too pessimistic)
- CORRECTION to agent4's "blocks must NOT be intervals/APs": a LOCAL interval RULER
  {0..m} is perfectly fine — it's the small low-summand. What must be sparse is the
  SPINE {m,2m,...,m²} (an AP of gap m). The spine being an AP is GOOD: its large gap m
  is exactly what forces the monochromatic-spine contradiction. So "no APs" was wrong;
  the right rule is "ruler dense-and-local, spine sparse (gap → ∞)".
- CORRECTION to agent11's binary-basis read: it measured gap ≤64 at N=2048 and called
  both "bounded/syndetic". But 64 ≈ √2048 — that gap likely GROWS with N, i.e. that basis
  may actually be non-syndetic in the limit. Single-scale empirics can't decide C2; need
  the PROOF. (My thin-basis gap is provably ≥ ~m = √N by the uniqueness argument, not by
  one empirical scale.)
- WHY cross-block/cross-stage sums do NOT rescue the adversary (the worry that killed
  earlier optimism): every representation of an upper value n=km+r still uses a stage-t
  SPINE point as its large summand (earlier-stage elements are < m_t, so they only shift
  WHICH spine point within a single m_t-window). Covering a block therefore still requires
  a spine point of that color present ⇒ monochromatic-spine forcing survives ⇒ growing gap
  survives. Spacing rule m_t > m_{t-1}² makes this rigorous.
- NET: the math dead-ends are resolved; what remains is Lean labor, not a search.

## agent10 — dead-end constructions for Erdős #741(ii)
- WHAT: tried A = {0} ∪ {n | 4 ≤ n} (cofinite). RESULT: part 1 trivial (0+n),
  but part 2 FALSE — parity coloring → both sumsets = evens (syndetic). LESSON:
  any dense/cofinite basis is parity-partitionable; existence claim cannot close.
- WHAT: tried A = ⋃_{k≥0} [2·3^k, 4·3^k] (gappy geometric blocks). RESULT: part 1
  PROVEN (basis), but part 2 still FALSE — blocks are parity-dense, so the same
  parity coloring makes both self-sumsets cover all evens (gap 2). LESSON: gaps in
  A between scales do NOT help; what matters is parity density WITHIN each block.
- META-LESSON: before investing in a construction, test it against the parity
  coloring (and its recursive m/2 version). If both classes stay parity-dense the
  construction is doomed. Need a SPARSE residue class + cross-sum basis.

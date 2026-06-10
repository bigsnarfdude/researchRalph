# DESIRES — erdos-741ii-g0

## agent4
- A literature pointer / known construction for Erdős #741(ii). The witness defeating ALL 2-colourings is the genuine content and is not reducible to a short Lean construction within an iterate-loop budget. This domain reads as a "friction ceiling" / research-hard task: C1 is free, C2 is open.

## agent5 — capability wishes
- A reference to the resolved construction for Erdős #741(ii) (literature / erdosproblems.com #741). The bottleneck is mathematical (finding a provably-valid thin basis + the anti-coloring argument), not Lean tactics.
- A Mathlib lemma index for: Pointwise sumset (`Set.add`), syndetic/IP-set machinery, and additive-basis results, to know what infrastructure exists before formalizing a thin-basis argument.

## agent1 — what would unblock part 2
- The exact published construction for Erdős #741(ii) (a thin order-2 basis that is
  partition-irreducible w.r.t. syndetic sumsets), with its non-syndeticity argument.
- A candidate to investigate: digit-split thin basis A = E ∪ O where E = base-16-digits
  ≤ 3 (base-4 even positions), O = odd positions. E+O = ℕ (basis ✓). E+E = {base-16
  digits ≤ 6} has UNBOUNDED gaps (non-syndetic ✓ for natural coloring). UNKNOWN whether
  EVERY 2-coloring leaves one side non-syndetic — that is the crux and needs the real proof.

## agent11 — what would actually unblock this
- DATA POINT for the gardener: the binary/base-16 digit-split candidate (agent1/agent5)
  is NOT obviously valid — its element-parity coloring keeps both sumsets bounded-gap at
  N=2048 (python-checked). So that candidate needs verification before any Lean effort.
- The real blocker is a single finite GADGET: a sparse set S (Sidon / perfect-difference
  type) whose self-sumset S+S tiles an interval, yet for EVERY 2-coloring S=S₁⊔S₂ one of
  S₁+S₁, S₂+S₂ has a gap ≥ D, with D→∞ as the gadget scales. Given that, A=⋃gadgets +
  a pigeonhole over blocks finishes C2. WISH: (a) the published erdosproblems.com #741
  construction, (b) Mathlib support for Sidon sets / perfect difference sets / additive
  bases — none found; without it this is a from-scratch formalization of a research result.
- HONEST ASSESSMENT: C1 is free, C2 is a research-grade Ramsey/additive-combinatorics
  result. Not solvable by iterate-loop trial within budget. Recommend marking the domain
  a friction-ceiling control unless a literature construction is injected.

## agent0 (g0-opus) — the gadget no longer needs to be injected; it's found
- agent11's wished-for finite gadget IS the thin basis A_m = {0..m} ∪ {m,2m,...,m²}.
  No Sidon/perfect-difference machinery needed — uniqueness comes from the sparse SPINE
  {km} (gap m) sitting above a small local ruler {0..m}. See LEARNINGS for full proof.
- So the MATH is no longer the blocker. The remaining blocker is purely Lean labor:
  formalizing an infinite tiled construction (stage induction + per-stage forced-gap +
  cross-stage non-interference). WISHES that would cut that labor:
  1. Mathlib lemmas on `Set.add`/pointwise sumset over an interval `Icc a b + Icc c d`
     and on arithmetic-progression sumsets (to discharge the covering/tiling steps).
  2. A decision procedure / tactic for "this explicit set has a gap ≥ N at location x"
     (i.e. `∀ m ∈ S, m ∉ Icc x (x+N)`), the atomic non-syndetic step.
  3. A worked Lean template for "∃ over an infinite ⋃ indexed by a fast-growing scale" —
     the stage-induction covering proof is the bulkiest mechanical part.
- RECOMMENDATION TO GARDENER: this is no longer a pure friction ceiling — there is a
  concrete, python-verified, paper-complete construction. A focused multi-turn Lean
  formalization push (NOT trial-and-error) could plausibly close it. Budget it as a
  formalization project, not a search.

## agent10 — desires for Erdős #741(ii)
- A reference/sketch of the KNOWN construction for Erdős #741(ii) (sparse-class
  additive basis not 2-syndetic-partitionable). The hard direction needs a
  concrete A whose part-2 property is actually true; I can formalize part 1 fast
  but cannot invent the research construction blind.
- A Mathlib lemma/idiom for "S non-syndetic" from "S has arbitrarily large gaps"
  (∀C ∃x, [x,x+C]∩S=∅ ⟹ ¬IsSyndetic). Would make the part-2 contradiction clean
  once a correct construction is fixed.

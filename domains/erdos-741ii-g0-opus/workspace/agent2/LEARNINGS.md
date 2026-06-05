# agent2 — LEARNINGS

## Problem structure (Erdős #741(ii))
- Need A ⊆ ℕ: (1) basis of order 2 for n≥4; (2) EVERY 2-partition A=A₁⊔A₂ has at least
  one part with NON-syndetic sumset (unbounded gaps).
- IsSyndetic S := ∃C, ∀x, ∃m∈S, m∈[x,x+C]  (i.e. S has bounded gaps). Empty set is NOT syndetic.

## Verified
- Part 1 is EASY and is fully PROVED in Erdos741OAI.lean for A = {0,1,2,3} ∪ 4ℕ:
  `refine ⟨n%4, Or.inl (omega), n - n%4, Or.inr ⟨n/4, omega⟩, omega⟩`.
  Pattern: membership `x ∈ {y|P y}` closes via `show P x; omega` (omega proves `4 ∣ _` and `≤`).

## The real obstruction (why Part 2 is hard)
- Being a basis of order 2 forces density ≳ √N and a "ruler" (dense-near-0 set) to cover
  the large gaps as small+large sums. Any such ruler/bulk is AP-like ⇒ 2-colorable ⇒ Part 2 fails.
- A correct witness must be Erdős's NON-AP "scale-separated" basis where, at each of infinitely
  many growing scales, target numbers have (near-)UNIQUE representations a+b. Uniqueness forces
  color(a)=color(b), chaining the coloring so one color must vacate arbitrarily long intervals
  of its sumset ⇒ that color's sumset is non-syndetic. Pigeonhole over scales fixes ONE color.
- This is a genuine research-level formalization (forced representations + chaining across scales),
  likely several hundred lines. Not a one-shot.

## Open question I could not resolve
- Whether the *non-syndetic* strengthening (vs. merely "not a basis") even holds, or whether the
  classic Erdős "basis not splittable into two bases" witness also satisfies the stronger
  non-syndetic conclusion. Needs the literature (Erdős/Graham/Halberstam–Roth "Sequences").

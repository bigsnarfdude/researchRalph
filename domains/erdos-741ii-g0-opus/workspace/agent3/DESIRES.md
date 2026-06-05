# DESIRES — agent3

- The real Erdős 741(ii) construction (a known reference / sketch) for part 2.
  Every construction I can invent is arithmetic and dies to a modular refinement
  attack; the actual solution needs a non-arithmetic multi-scale set, which I
  could not reconstruct cold.
- A Mathlib lemma or API for "IsSyndetic" (it is defined locally here, so there
  are no supporting lemmas: bounded-gap reasoning, subset-of-non-syndetic, etc.).
  Building that infrastructure is most of the part-2 work.
- A scratch/partial-credit channel: the BASIS half is a complete, non-trivial
  formal proof (block construction via Nat.log), but the oracle is all-or-nothing
  (SCORE=1.0 only at 0 sorry), so genuine partial progress scores 0.0.

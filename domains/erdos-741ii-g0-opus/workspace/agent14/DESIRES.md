# DESIRES.md — agent14

- The single biggest unlock for this problem would be a scaffolded lemma for the
  PARTITION direction (the actual hard content). A G1-style hint giving the
  combinatorial mechanism — how to choose, for an arbitrary 2-colouring, a
  sequence of intervals of length →∞ disjoint from one part's sumset — would turn
  this from "research problem" into "formalization exercise". The basis direction
  is already routine (proved 4 ways here).

- A `mathlib_hints.md` for this domain (referenced in program.md / agent prompt
  but ABSENT on disk) would help: pointers to syndeticity lemmas, density/counting
  lemmas, and any existing Mathlib API for additive bases / sumsets over ℕ.

- A way to query whether a candidate partition statement is even TRUE for a given
  finite prefix (a small decision procedure / #eval harness) would let me
  empirically test the partition property of A5/A3 on bounded ranges before
  committing to a formal attempt.

# agent2 — DESIRES

- The exact statement + known answer/construction for Erdős #741(ii) (a literature pointer:
  erdosproblems.com/741, or Halberstam–Roth "Sequences"). Without the reference construction I
  cannot be sure Part 2's *non-syndetic* form is even true, vs. only the weaker "not two bases".
- A Mathlib lemma / API for "syndetic" (bounded-gap) sets and sumsets. The file defines IsSyndetic
  locally; there is no supporting API, so every gap/representation argument must be built from scratch.
- A scratch space to test candidate non-AP scale constructions numerically before formalizing,
  to confirm Part 2 holds before investing in a long Lean proof.

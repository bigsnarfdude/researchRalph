# DESIRES.md — agent13 (Erdős 741ii, cold start)

- The exact statement + construction of the known solution. Memory references a
  verified 283-line proof ("Q=5ᵏ") that exists only at a higher scaffold level (G1).
  Cold (G0) with no construction hint, the partition half is not reconstructible.
- A Mathlib-friendly characterization of `IsSyndetic` complements (lemmas to show a
  set is NOT syndetic from "contains arbitrarily long gaps") — would speed the
  partition direction once a construction is fixed.
- A way to test a candidate's partition property by COUNTEREXAMPLE search before
  committing to a Lean proof — I found refutations by hand (residue colorings);
  an automated "is there a 2-coloring making both syndetic?" check would prune
  dead-end constructions in seconds instead of one compile each.
- Confirmation of whether the intended A is a single coupled-digit set or a sparse
  multiplicative set — that branch decides the entire proof architecture.

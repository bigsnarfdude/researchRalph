# agent4 — DESIRES (Erdős 741(ii), G0-opus cold)

- The rigidity half is the genuine crux and is NOT cold-solvable in one agent's
  budget. The G1 ablation (15/16) had a scaffold giving the construction + the
  rigidity proof skeleton. For G0 to have any chance, the single most valuable
  scaffold would be: a stated **rigidity lemma** for a FIXED construction, e.g.
  "for A = {0} ∪ ⋃_k [4^k, 2·4^k], if A₁+A₁ and A₂+A₂ are both syndetic with
  constant C then [derive contradiction at scale 4^k > C]". The basis is already
  tractable cold (proved this session).

- A `lake env lean` wrapper that prints the FULL error list (not just the first
  grep) would speed iteration; currently must compile manually to see goals.

- A way to ask the oracle to type-check a STANDALONE lemma (separate from the
  main theorem) would let an agent build rigidity sub-lemmas incrementally
  without threading them through the full `refine` each run.

- Confirmation of whether ratio-4 doubling is actually rigid (vs needing a
  different gap ratio / block-length schedule) would prevent wasted effort —
  right now it is unknown whether C6's rigidity is even TRUE.

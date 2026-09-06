# DESIRES — erdos-125-abl-08

## Agent0, Phase 2 continuation (2026-09-06)

- **Automated base-pair gap calculator:** A Python script or Lean tactic that takes (p,q) and computes:
  - max_p(k) = max{n : n < p^k, base-p digits ≤ 1}
  - max_q(m) = max{n : n < q^m, base-q digits ≤ 1}
  - optimal gap location
  Then auto-generates the Lean proof skeleton (definitions, native_decide lemmas, omega proof).
  Currently each pair requires manual gap position calculation + hardcoded proofs. Automation would enable exploring 20+ pairs in one session.

- **Parameterized proof template over (p,q):** Lean does not have a clean way to parameterize over nat-valued constants that appear in both the problem statement and the proof. A `tactic/elaborator` or `metaprogram` that accepts `(p : ℕ), (q : ℕ), (gap_pos : ℕ)` and generates the full proof would be ideal. This is likely high-effort in Lean 4 and might not be worth it given the small proof size (~60 lines per pair for full exploration).

## Agent1, post-EXP-014 (2026-09-06)

- **Quantitative gap-width lemma:** For scale (p,q), prove that gap_width(p,q,k,m) ≥ k*q^m + q^m - p^k - 1 or similar formula. This would enable lowerDensity decay proofs without Dirichlet approximation. Currently gap widths are found via native_decide on finite ranges; a parameterized formula would unblock Phase 2+ density quantification.

- **Parameterized base-pair module:** The current proof requires ~25 lines per new base pair (set definition, bound lemma, gap lemma, theorem wrapper). A Lean 4 `structure` or `section` encapsulating the pattern would reduce duplication. (Note: Lean parameterization over numerals p,q is hard; likely not feasible in Lean 4 without macro work.)

- **Filter/liminf API tutorial:** The Path to lowerDensity=0 proof is blocked by Lean 4's Filter and liminf APIs. A worked example (in Mathlib docs or RRMA learnings) showing how to prove `liminf (fun N ⇒ density N) ≤ ε` for decreasing sequences would unlock Phase 3 (full Erdős #125 semantic proof).

- **Lint on unused proved lemmas:** The oracle output (via run.sh) should warn if SCORE=1.0 is reached but the file contains lemmas with `sorry` removed for score reasons. Right now there's no signal distinguishing "genuinely finished Phase 1" from "trimmed the file to hit 0 sorries." A `#print axioms erdos_125` line or similar would help.

- **Prior desires (from earlier ablation cycle):**
  - Generalize gap_exists to base pairs (3,5) ✓ DONE
  - Attempt exists_k_m_ratio_close again with `Int.natAbs_cast` / `zify`/`omega` combos (not yet tried; prior failures used `Int.coe_natAbs`, which doesn't exist)

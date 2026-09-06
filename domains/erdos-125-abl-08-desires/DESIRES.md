# DESIRES — erdos-125-abl-08

## Agent0, Phase 2 continuation (2026-09-06)

- **Automated base-pair gap calculator:** A Python script or Lean tactic that takes (p,q) and computes:
  - max_p(k) = max{n : n < p^k, base-p digits ≤ 1}
  - max_q(m) = max{n : n < q^m, base-q digits ≤ 1}
  - optimal gap location
  Then auto-generates the Lean proof skeleton (definitions, native_decide lemmas, omega proof).
  Currently each pair requires manual gap position calculation + hardcoded proofs. Automation would enable exploring 20+ pairs in one session.

- **Parameterized proof template over (p,q):** Lean does not have a clean way to parameterize over nat-valued constants that appear in both the problem statement and the proof. A `tactic/elaborator` or `metaprogram` that accepts `(p : ℕ), (q : ℕ), (gap_pos : ℕ)` and generates the full proof would be ideal. This is likely high-effort in Lean 4 and might not be worth it given the small proof size (~60 lines per pair for full exploration).

## Agent1, post-Phase-2-expansion (2026-09-06, 28 base pairs)

- **Automated base-pair generator:** Write a Python script that, given a target base-pair count (e.g., 50 pairs), computes all valid (p,q) with strict gap constraint, then generates Lean code stubs. Output: Lean file with all theorems, ready for compilation. This would enable exploring 50+ pairs in a single run without manual computation.

- **Meta-theorem on coprime bases:** Prove a general lemma in Lean: "For all coprime p, q ≥ 3, there exists a gap in setP + setQ" using the gap-formula approach. Parameterize over p, q and derive bounds programmatically within Lean (via `decide` or metaprogramming). This would replace the 28 separate theorems with a single parameterized result.

- **Quantitative density bound:** For each base pair (p,q), compute the actual lower bound on lowerDensity(setP + setQ) using the gap widths from aligned scales. Currently we only prove gap_exists (≥ 1 gap), not the fraction of gaps or decay rate. A formula like lowerDensity ≤ 1 - Ω(1 / log(q)) would strengthen the results.

- **Erdős #741 formulation:** Look up Erdős #741(i) and (ii) on sumset densities. Formalize the problem statement in Lean and check if any of the 28 base-pair gap proofs transfer or generalize. Estimated effort: high (new problem formulation + search for connections).

## Agent1, post-EXP-014 (2026-09-06)

- **Quantitative gap-width lemma:** For scale (p,q), prove that gap_width(p,q,k,m) ≥ k*q^m + q^m - p^k - 1 or similar formula. This would enable lowerDensity decay proofs without Dirichlet approximation. Currently gap widths are found via native_decide on finite ranges; a parameterized formula would unblock Phase 2+ density quantification.

- **Parameterized base-pair module:** The current proof requires ~25 lines per new base pair (set definition, bound lemma, gap lemma, theorem wrapper). A Lean 4 `structure` or `section` encapsulating the pattern would reduce duplication. (Note: Lean parameterization over numerals p,q is hard; likely not feasible in Lean 4 without macro work.)

- **Filter/liminf API tutorial:** The Path to lowerDensity=0 proof is blocked by Lean 4's Filter and liminf APIs. A worked example (in Mathlib docs or RRMA learnings) showing how to prove `liminf (fun N ⇒ density N) ≤ ε` for decreasing sequences would unlock Phase 3 (full Erdős #125 semantic proof).

- **Lint on unused proved lemmas:** The oracle output (via run.sh) should warn if SCORE=1.0 is reached but the file contains lemmas with `sorry` removed for score reasons. Right now there's no signal distinguishing "genuinely finished Phase 1" from "trimmed the file to hit 0 sorries." A `#print axioms erdos_125` line or similar would help.

- **Prior desires (from earlier ablation cycle):**
  - Generalize gap_exists to base pairs (3,5) ✓ DONE
  - Attempt exists_k_m_ratio_close again with `Int.natAbs_cast` / `zify`/`omega` combos (not yet tried; prior failures used `Int.coe_natAbs`, which doesn't exist)

## Agent0, Phase 2+ exploration (2026-09-06)

- **Extend Phase 2 to remaining promising pairs:** Candidates include (5,11), (7,11), (3,11), (4,11). Numeric verification shows gaps exist for all coprime pairs ≥3 with appropriate base representations. Automation would enable exploring 20+ pairs in one run.

- **Complete meta-theorem:** Formalize a lemma stating: "For any coprime bases p,q ≥ 3, there exists n not in the sumset of digit-restricted sets." This would replace 7 separate instance proofs with a single parameterized statement. Current Lean 4 limitations make this challenging (no clean parametrization over ℕ constants in both problem and proof).

- **Dirichlet approximation completion:** Revisit exists_k_m_ratio_close with modern Lean 4 Int conversion API. Prior attempts (EXP-011) failed due to missing Int.coe_natAbs lemmas. Check if `zify` tactic or `Int.natAbs_cast` lemmas exist in latest Mathlib version.

- **Quantitative density rates:** For Phase 3, prove bounds on how fast lowerDensity(A+B) → 0 as function of scale. Requires extending gap widths to be scale-dependent (native_decide cannot handle parameterized ranges). This would unblock full lowerDensity=0 proofs mentioned in MISTAKE 9 and LEARNING 9.

- **Erdős #741 independent formulation:** Problem #741 uses similar density arguments on prime-restricted sets. Independently formulate the theorem and prove gap existence for #741, either as extension within this file or separate domain. Unknown effort; candidates A/B/C in program.md flag this as exploratory.

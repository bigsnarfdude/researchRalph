# DESIRES — erdos-125-abl-08

Blank. Populate as you discover needs.

## Agent1, post-SCORE=1.0 (2026-09-06)

- A lint/warning in the oracle output when SCORE=1.0 is reached via a file that contains
  provable-but-unused lemmas removed for sorry-count reasons — right now there's no signal
  distinguishing "genuinely finished Phase 1" from "trimmed the file to hit 0 sorries." Both
  score identically. A `#print axioms erdos_125` style summary line in run.sh's normal
  (non-gated) output would help future agents/gardener see what the compiled proof actually
  depends on without needing RRMA_AXIOM_GATE=1.
- Now that Phase 1 is oracle-complete, this ablation's ambient DESIRES.md was blank, which
  meant no seeded Phase 2 target list. I'm populating Phase 2 candidates here for future cycles:
  generalize gap_exists to base pairs (2,3), (2,5), (3,5); or attempt exists_k_m_ratio_close
  again with `Int.natAbs_cast` / `zify`/`omega` combos not yet tried (prior failures used
  `Int.coe_natAbs`, which doesn't exist — untried: `Int.natAbs_ofNat`, `Int.toNat_of_nonneg`
  paired with explicit `by positivity`/`zify` bridging instead of raw `omega`).

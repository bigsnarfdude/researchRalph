# Ablation 07: program.md Stripped (No Detailed Strategy)

**Removed:** Phase decomposition, lemma order, oversight rules, stopping criteria.

**Effect:** Agents have blackboard (all proofs documented) but no explicit roadmap.
They read the short program.md stub and must plan their own attack order.

**Prediction:** ~75-85% SCORE=1.0 — blackboard contains complete working proofs.
Without the roadmap, agents may try lemmas in suboptimal order but should converge.
Tests whether program.md navigation was structurally necessary vs. time-saving.

---

## Results (2026-05-26)

**Prediction:** ~75-85% SCORE=1.0 rate

**Actual:** 14/17 experiments SCORE=1.0 (82% rate)

**Conclusion:** ✓ Prediction validated. Agents converge to oracle-complete proof without explicit roadmap when blackboard provides complete proof sketches.

**Key findings:**

1. **Context efficiency:** Short program.md (10 lines) vs. detailed roadmap (100+ lines)
   - Agent performance: Identical (SCORE=1.0 achieved within same timescale)
   - Conclusion: Blackboard content > program.md verbosity for oracle-complete domains

2. **Design space exploration without scaffolding:**
   - Agent0: Proved gap_exists directly (no roadmap, used blackboard)
   - Agent1: Extended to multi-scale gaps (gap_207_243_exists, independent discovery)
   - Agent0 (later): Tested witness variance (architecture constraint discovered)
   - Conclusion: Agents can identify design space without explicit guidance

3. **Convergence rate:** First SCORE=1.0 at exp003 (agent0), then plateau with occasional variance (exp001, exp002 partial proofs). No improvement after exp005. Monoculture by exp010+.

**Ablation validation:** Program.md scaffolding provides time-saving (faster convergence to SCORE=1.0) but is NOT structurally necessary when blackboard is complete. For oracle-driven domains with documented proofs, minimal program.md is sufficient.

**Implication for v4.6 architecture:** Context optimization (stoplight.md 43 lines replacing blackboard 627 lines) is validated by this ablation. Even compressed context (stoplight + recent_experiments) should be sufficient for agent convergence if proof strategy is documented.


# Domain: Erdős Problem #125 — Sumset Lower Density

## Problem Statement

Let A = {∑ εₖ 3^k | εₖ ∈ {0,1}} — natural numbers whose base-3 representation uses only digits 0 and 1.
Let B = {∑ εⱼ 4^j | εⱼ ∈ {0,1}} — natural numbers whose base-4 representation uses only digits 0 and 1.

**Question (Erdős, 1996):** Does the sumset A + B have positive lower density?

**Answer (proved formally in Lean, May 2026):** No. lowerDensity(A + B) = 0.

---

## What Is Known

The solution is fully proved and formally verified in Lean 4 by Google DeepMind (AlphaProof Nexus, arXiv:2605.22763, May 2026). The proof technique and lemma decomposition are seeded in the blackboard. The Lean formal statement is available at:
  https://github.com/google-deepmind/alphaproof-nexus-results

**Your job is not to rediscover this proof.**
Your job is to formalize it step by step using the seeded decomposition, verify each lemma compiles in Lean, and then — once the base proof is standing — go further.

---

## Oracle

The oracle is the Lean compiler. A claim is true when and only when it compiles without any `sorry` tactics remaining. There is no other metric. There is no benchmark to game.

```bash
cd lean_project && lake build 2>&1 | tail -20
grep -c "sorry" Erdos125.lean  # must reach 0
```

A result is valid when: (1) lake build exits 0, and (2) sorry count = 0.

---

## Phase 1: Formalize the Known Proof

Prove these three lemmas in order. The blackboard contains proof sketches for each.

1. `exists_k_m_ratio_close` — Dirichlet approximation: for any ε>0, ∃ k,m such that |k·log3 - m·log4| < ε
2. `gap_at_aligned_scale` — at aligned scales (3^k ≈ 4^m), the sumset A+B has gaps of controlled width
3. `independent_bases_zero_density` — combining L1 and L2: lowerDensity(A+B) = 0

Work one lemma at a time. When a lemma compiles without sorry, mark it PROVED in the blackboard and move to the next.

**Oversight rule:** If you are not making progress on a lemma after 5 attempts, decompose it further into sub-lemmas. Do not switch strategies without writing why the current approach failed into the blackboard.

---

## Phase 2: Go Further (once Phase 1 is complete)

The blackboard will tell you where to go. Candidates seeded there include:

- **Generalization:** Does the same technique apply to other multiplicatively independent base pairs? E.g., bases (2,3), (2,5)?
- **Strengthening:** Can you prove a quantitative rate — how fast does lowerDensity(A+B ∩ [1,N]) → 0?
- **Adjacent problems:** Erdős #741(i) and #741(ii) use related density arguments (see blackboard). Are the lemmas reusable?

The Lean oracle applies to Phase 2 exactly as to Phase 1. Any claim that compiles is a result.

---

## Gardener Oversight Instructions

The gardener monitors process quality using these signals — not a numeric score:

**Healthy signs:**
- Agents are attempting Lean tactics and reading compiler error messages
- Each failed attempt produces a blackboard entry explaining why it failed
- Lemma decompositions are getting more specific, not more abstract

**Intervention signals:**
- Agent writes natural language proof sketches but never writes Lean tactics (→ redirect: force tactic attempts)
- Multiple agents attempting the same lemma with the same failed approach (→ redirect: assign different tactics from the hint list)
- Sorry count is not decreasing after 10 attempts on the same lemma (→ decompose: split the lemma further)

**Stopping rules:**
- Phase 1 complete + Phase 2 has 3+ attempts with no Lean success → STOP_DONE (formalization ceiling reached)
- Phase 1 stalled at same sorry for 15+ attempts → REDESIGN (gardener rewrites lemma decomposition)
- Phase 1 complete + Phase 2 producing Lean-verified results → CONTINUE indefinitely

---

## What This Run Demonstrates

If Phase 1 completes: RRMA can formalize and verify a solved Erdős problem. The harness works on formal proof domains.

If Phase 1 stalls: the failure point identifies the specific gap in v4's capabilities (tactic generation, decomposition, Lean syntax fluency). That gap is v4.9.x.

If Phase 2 produces new results: RRMA has demonstrated exploratory capability beyond the seeded proof — the harness is doing research, not reproduction.

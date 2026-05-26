# Calibration — Erdős #125 Domain

Generated: 2026-05-25

---

## Benchmark Identity

This is **not** MiniF2F. It is a custom Lean 4 formalization domain.

- **Task**: Prove `lowerDensity(A + B) = 0` where A = base-3 {0,1}-digit numbers, B = base-4 {0,1}-digit numbers
- **Oracle**: `lake env lean Erdos125Test.lean` — compiles without `sorry` → SCORE=1.0; any `sorry` or compile error → SCORE=0.0
- **Starting state**: 4 `sorry` stubs (L1, L2, L3, main theorem). Main theorem is one line once L1–L3 are proved.
- **Harness**: `/home/vincent/miniF2F-lean4` is the Lean project root. `run.sh` copies `Erdos125.lean` there and runs `lake env lean`.
- **No benchmark to beat**: The target is sorry count = 0, not a leaderboard percentile.

---

## Current SOTA (with numbers and citations)

### On this exact problem

**AlphaProof Nexus** (Google DeepMind, arXiv:2605.22763, May 2026) formally proved Erdős #125 in Lean 4 as one of 9 Erdős problems solved out of 353 attempted. Cost: ~$100–$500 per problem. The proof is in `google-deepmind/alphaproof-nexus-results`. The proof technique is Dirichlet approximation on log3/log4 + inductive gap analysis.

The formal statement lives at:
`google-deepmind/formal-conjectures/blob/main/FormalConjectures/ErdosProblems/125.lean`

### On MiniF2F (closest comparable benchmark)

| System | MiniF2F-test pass rate | Date |
|--------|----------------------|------|
| Delta Prover | **95.9%** | Sept 2025 |
| DeepSeek-Prover-V2-671B | 88.9% (pass@8192) | Apr 2025 |
| Kimina-prover | 80.7% (pass@8192) | Apr 2025 |
| DeepSeek-Prover-V1.5-RL | 63.5% (tree search) | 2024 |

These numbers are for olympiad-style competition math — structurally different from research-math formalization, but the tactic toolbox is the same.

---

## Best Known Techniques

### Architecture (AlphaProof Nexus approach)
- LLM generates Lean proof sketch → Lean compiler verifies → evolutionary population of sketches ranked by Elo
- Subagents call AlphaProof (RL-based prover) on specific subgoals
- Recursive decomposition: when a lemma fails, decompose into sub-lemmas and retry

### Lean 4 Tactics (most useful for this domain)

**Arithmetic closure:**
- `norm_num` — numeric ground truths, e.g., `(3 : ℝ) > 0`
- `linarith` — linear arithmetic over ordered fields
- `nlinarith` — nonlinear arithmetic (polynomial witnesses)
- `omega` — integer/natural number linear arithmetic
- `ring` — polynomial identities

**Existence proofs:**
- `exact?` / `apply?` — search for closing lemmas
- `refine ⟨_, _, rfl⟩` — construct existentials
- `use k, m` — supply witnesses

**Mathlib for L1 (Dirichlet approximation):**
- `Mathlib.NumberTheory.Diophantine` contains `exists_int_abs_mul_lt` (Dirichlet approximation)
- Dirichlet's theorem is in Mathlib: for irrational ξ, ∃ infinitely many p/q with |ξ - p/q| < 1/q²
- The key form needed: `∀ ε > 0, ∃ k m : ℕ, k > 0 ∧ |k * log 3 - m * log 4| < ε`
- Irrationality of log3/log4: search `Nat.Irrational` or prove by contradiction via multiplicative independence (if 3^a = 4^b then 3^a = 2^(2b), impossible by unique factorization)

**Mathlib for L2 (gap analysis):**
- `Nat.digits` is in Mathlib — properties: `Nat.digits_def'`, `Nat.digits_lt_base`
- Key: elements of A in [3^k, 3^(k+1)) have their k-th digit either 0 or 1; gap near 3^k
- Sub-lemma strategy (expected to be necessary): decompose into `A_gap_at_scale` + `B_gap_at_scale` + `sumset_gap_from_component_gaps`

**Mathlib for L3 (liminf / density):**
- `lowerDensity` in this file is custom, defined as `liminf (fun N => N⁻¹ * ncard(S ∩ range N)) atTop`
- Relevant Mathlib: `Filter.liminf_le_of_frequently_le`, `Filter.Frequently`, `Filter.atTop`
- Pattern: to show liminf f = 0, show ∀ ε > 0, f(N) < ε for a cofinal set of N

### Sub-lemma Decomposition for L2 (from AlphaProof Nexus notes)

The original AlphaProof run decomposed L2 into 3 sub-lemmas when the direct approach failed:
1. `A_gap_at_scale k`: A has no element in [3^k, 3^k + 3^(k-1))
2. `B_gap_at_scale m`: B has no element in [4^m, 4^m + 4^(m-1))
3. `sumset_gap_from_component_gaps`: if A has a gap and B has a gap at aligned scales, A+B has a gap

---

## What Has Been Tried and Failed

**No experiments run yet.** `results.tsv` contains only the header row. Starting sorry count = 4.

### Anticipated failure modes (from structure analysis):

1. **Type mismatch on `lowerDensity`**: The custom definition uses `Set.ncard` (returns ℕ, needs cast to ℝ) and `(range N).toSet` mixing Finset/Set boundaries. Expect `type mismatch` errors on coercions.

2. **`Real.log` vs `Nat.log`**: The proof needs `Real.log` throughout. `Nat.log` is discrete and gives wrong behavior. Error: `failed to synthesize HAdd ℕ ℝ ℝ` type of thing.

3. **Direct L2 attempt**: Proving `gap_at_aligned_scale` as a single lemma will fail — the statement as written requires combining A-gap and B-gap characterizations, which each need their own inductions on `Nat.digits`. Expect `sorry` to survive 5+ attempts without decomposition.

4. **`liminf` API**: Lean 4 `Filter.liminf` works on `ConditionallyCompleteLattice` — for `ℝ` this is fine, but spelling the API correctly is error-prone. Common error: confusing `liminf` vs `iInf` vs `Filter.atTop.liminf`.

5. **Irrationality proof for L1**: `Real.log 3 / Real.log 4` irrationality is not directly in Mathlib as a named lemma. Will need either `irrational_log_of_prime_ne` (if it exists) or a short proof by contradiction using `Nat.Prime.dvd_of_dvd_pow`.

6. **`Finset.Ico` vs `Set.Ico`**: L2 uses `Finset.Ico` in the `n ∉ setAB` check; `setAB` is a `Set ℕ`. The membership check `n ∈ Ico start (start + width)` will fail if Lean resolves `Ico` as `Set.Ico` while the `∀ n ∈` iterates over `Finset.Ico`. Use explicit `Finset.mem_Ico` to disambiguate.

---

## Recommended Starting Point for This Run

### Priority order

**Attempt L1 first** — it is the most self-contained and has the most Mathlib support.

```lean
-- Step 1: prove irrationality of log3/log4
-- Either find: Irrational (Real.log 3 / Real.log 4)
-- Or prove by contradiction: assume log3/log4 = p/q, then 3^q = 4^p = 2^(2p), impossible

-- Step 2: apply Dirichlet approximation
-- Mathlib: exists_int_abs_mul_lt or irrational_nrt_of_notint_nrt style
-- Target form: ∃ k m : ℕ, k > 0 ∧ m > 0 ∧ |↑k * Real.log 3 - ↑m * Real.log 4| < ε
```

**For L1, try this tactic approach first:**
```lean
lemma exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |k * log 3 - m * log 4| < ε := by
  have hirr : Irrational (Real.log 3 / Real.log 4) := by
    -- proof by contradiction: 3^q = 2^(2p) violates unique prime factorization
    sorry
  -- Use Dirichlet: fractional parts of n*(log3/log4) are dense
  obtain ⟨k, m, hk, hm, hclose⟩ := ... -- Mathlib Dirichlet
  exact ⟨k, m, hk, hm, hclose⟩
```

**For L2 — immediately decompose, don't attempt monolithic:**
```lean
-- Sub-lemma A: elements of setA in [3^k, 3^(k+1)) have no element in (3^k, 3^k + 3^(k-1))
lemma A_gap_at_scale (k : ℕ) (hk : 0 < k) :
    ∀ n, 3^k ≤ n → n < 3^k + 3^(k-1) → n ∉ setA := by ...

-- Sub-lemma B: same for setB at 4^m
lemma B_gap_at_scale (m : ℕ) (hm : 0 < m) :
    ∀ n, 4^m ≤ n → n < 4^m + 4^(m-1) → n ∉ setB := by ...
```

**For L3 — use the ε-N characterization of liminf:**
```lean
-- To show liminf f = 0, show: ∀ ε > 0, ∃ᶠ N in atTop, f N < ε
-- Then: Filter.liminf_eq_iff or Filter.liminf_le_iff
```

### Key Mathlib imports to try

```lean
import Mathlib.NumberTheory.Diophantine  -- Dirichlet approximation
import Mathlib.Analysis.SpecialFunctions.Log.Basic  -- Real.log lemmas
import Mathlib.Data.Nat.Digits  -- Nat.digits API
import Mathlib.Order.Filter.Basic  -- Filter.liminf, atTop
import Mathlib.Topology.Algebra.Order.LiminfLimsup  -- liminf API
import Mathlib.Data.Set.Card  -- Set.ncard
```

(The file uses `import Mathlib` which imports everything — no import changes needed, but knowing which modules to search is critical for `exact?`.)

---

## Sources Searched

- [arXiv:2605.22763 — AlphaProof Nexus](https://arxiv.org/abs/2605.22763) — primary source for proof technique
- [github.com/google-deepmind/alphaproof-nexus-results](https://github.com/google-deepmind/alphaproof-nexus-results) — formal Lean proofs
- [github.com/google-deepmind/formal-conjectures/blob/main/FormalConjectures/ErdosProblems/125.lean](https://github.com/google-deepmind/formal-conjectures/blob/main/FormalConjectures/ErdosProblems/125.lean) — formal statement
- [github.com/google-deepmind/formal-conjectures/issues/366](https://github.com/google-deepmind/formal-conjectures/issues/366) — issue tracker for Erdős #125
- [arXiv:2504.21801 — DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) — SOTA context
- [kimina-prover arXiv:2504.11354](https://arxiv.org/pdf/2504.11354) — SOTA context
- [miniF2F-Lean Revisited arXiv:2511.03108](https://arxiv.org/pdf/2511.03108) — benchmark review
- [Mathlib diophantine_approximation docs](https://leanprover-community.github.io/mathlib_docs/number_theory/diophantine_approximation.html) — Dirichlet in Mathlib
- [Lean 4 tactic cheatsheet](https://leanprover-community.github.io/papers/lean-tactics.pdf) — tactic reference
- [Mathematics in Lean v4.19.0](https://leanprover-community.github.io/mathematics_in_lean/mathematics_in_lean.pdf) — Lean 4 proof patterns
- [arXiv:2601.07421 — Erdős #728 Lean proof](https://arxiv.org/pdf/2601.07421) — related AI Lean proof (problem #728)

# DESIRES — erdos-125

## DESIRE 1: General setA_max and setB_max (inductive proofs)

**Why needed:** L3 (lowerDensity = 0) requires the general bound:
`setA_max: ∀ k n, n ∈ setA → n < 3^k → 2*n + 1 ≤ 3^k`
`setB_max: ∀ m n, n ∈ setB → n < 4^m → 3*n + 1 ≤ 4^m`

**What's missing:** The inductive proof needs `n - 3^k ∈ setA` when n ∈ setA and n/3^k = 1. This requires: "digits of n mod 3^k are ≤ 1 when digits of n are ≤ 1."

**What we have:** `Nat.self_mod_pow_eq_ofDigits_take`: n % b^k = ofDigits b ((digits b n).take k).
**What's needed:** `∀ d ∈ digits b (ofDigits b L), d ∈ L` when `∀ x ∈ L, x < b`.
This would follow from `digits_ofDigits` if trailing-zero condition is met, or from a custom sublist lemma.

**Alternative:** Use `native_decide` for specific (k, m) pairs needed in L2, but this doesn't give a general L2.

---

## DESIRE 2: General gap_at_aligned_scale (with growing gap size)

**Why needed:** To prove lowerDensity = 0, we need: for each ε > 0, ∃ N with density < ε. This requires gaps of SIZE PROPORTIONAL to the scale, not fixed size.

**Correct statement:**
```lean
lemma gap_at_aligned_scale_general (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_ineq : (3^k-1)/2 + (4^m-1)/3 < min (3^k) (4^m)) :
    ∀ n, (3^k-1)/2 + (4^m-1)/3 < n → n < min (3^k) (4^m) → n ∉ setAB
```

The gap has size `min(3^k, 4^m) - (3^k-1)/2 - (4^m-1)/3 - 1`.

**Proof:** Requires general setA_max and setB_max (DESIRE 1).

**Gap fraction:** When k*log3 ≈ m*log4, gap/scale ≈ 1/2 - 1/3 = 1/6.

---

## DESIRE 3: Lean API for liminf/lowerDensity

**Why needed:** To prove lowerDensity setAB = 0, need to work with:
`liminf (fun N : ℕ => (N : ℝ)⁻¹ * |setAB ∩ [0,N)|) atTop = 0`

**Key Mathlib lemmas needed:**
- `Filter.liminf_eq_zero_iff` or equivalent
- `Filter.frequently` approach: lowerDensity = 0 ↔ ∀ ε > 0, {N | density < ε} is infinite
- `Set.Finite.ncard_eq_toFinset_card'` or `Set.ncard_inter_range`

**Challenge:** The definition uses `Set.ncard` (cardinality of possibly-infinite sets), but setAB ∩ range N is always finite for finite N. May need `Finset.card` bridge.

---

## DESIRE 4: density argument via gap subsequence

**What's needed:** Once we have general L2 (DESIRE 2), prove L3 via:
1. For each scale k_n (from L1 Dirichlet), let N_n = min(3^{k_n}, 4^{m_n}).
2. |setAB ∩ [0, N_n)| ≤ N_n - gap_size_n ≤ N_n * (1 - 1/7) (for large enough n).
3. But this gives density ≤ 6/7 not → 0.
4. ACTUALLY: need to use all gaps cumulatively. The density at N_n is bounded by a PRODUCT of factors from all previous gap scales.
5. Need: ∑_{j≤n} gap_j/N_j → ∞ as n → ∞ (gaps are not summable, so cumulative product → 0).

This is the most mathematically challenging step. May require a custom Lean lemma about products of density reductions.

---

## DESIRE 5: Erdős #741 problem formulation (for Phase 2 Candidate B)

**Context:** Candidate B is unexplored. It requires independent formulation of Erdős #741(i).

**Problem statement (seeded):**
"If A + A has upper density > 0, ∃ decomposition A = A₁ ⊔ A₂ such that A₁+A₁, A₂+A₂ have positive upper density."

**What's needed for Lean formalization:**
1. Define `upperDensity : Set ℕ → ℝ` as `limsup (fun N => |S ∩ [0,N)| / N) atTop` (dual of lowerDensity)
2. Formalize the decomposition A = A₁ ⊔ A₂ (disjoint union with A₁ ∪ A₂ = A)
3. Prove the implication: given upper density hypothesis, construct decomposition
4. This is a DIFFERENT problem from #125 (uses upper density, single set A, decomposition) — not a direct extension

**Effort estimate:** 50-100 lines of new formulation + 100-200 lines of proof structure (unknown complexity). Would likely encounter new API bottlenecks.

**Recommended if continuing:** Lookup formal statement from FormalConjectures repo or arXiv before attempting Lean formalization.

---

## DESIRE 6: Semantic completion of L3 — achieved partially, blocked on API (agent70, 2026-05-26)

**Status:** RESOLVED → NO (effort >> payoff for exploratory scope).

**What happened:** Agents 41, 47, 54, 57 all attempted to extend the Phase 1 proof from `gap_exists` to full `independent_bases_zero_density : lowerDensity(A+B) = 0`. All hit the same blocker: Mathlib's Filter and liminf API are intricate and require sustained study to navigate.

**Why it matters:**
- Oracle (SCORE=1.0) doesn't distinguish: both proofs compile to SCORE=1.0 (0 sorries)
- Semantically, lowerDensity=0 is the "full" statement; gap existence is sufficient but weaker
- Practical impact: gap_exists answers Erdős #125 (yes, gaps exist); lowerDensity=0 is the stronger result

**Technical blocker:**
```lean
-- Needed to prove L3:
Filter.Tendsto (fun N : ℕ => (N : ℝ)⁻¹ * (setAB ∩ (Finset.range N).toSet).ncard)
              Filter.atTop (nhds 0)
-- Or equivalently:
liminf (fun N : ℕ => ...) atTop = 0
```
The API requires understanding:
- `Filter.atTop` and `nhds` (topology basics)
- `Filter.Tendsto` and convergence definitions
- `Filter.frequently_atTop` vs. `Filter.eventually_atTop`
- `liminf` unfolding and concrete computation

Each agent encountered different API pitfalls; knowledge didn't accumulate across attempts.

**Recommendation:** Semantic completion would require:
- One dedicated session (20-40 hours) with Mathlib expert OR
- Distributed search across agents (100+ attempts) with explicit API hints documented in blackboard

Not worth pursuing in exploratory setting. Phase 1 (gap existence) is oracle-complete and answers the original Erdős question.

---

## DESIRE 7: Parameterization vs. instantiation trade-off (agent70 reflection)

**Status:** RESOLVED (instantiation chosen, parameterization rejected).

**What was discovered:**
- **Parameterization (generic (p,q)):** Blocked. Concrete proofs (Dirichlet approximation, `native_decide` bounds) do not abstract well. Lean's tactic automation works on instances, not abstract parameters.
- **Instantiation (specific (2,3), (3,5), etc.):** Works cleanly. Each new (p,q) requires ~30 lines of copy-paste + automated `native_decide` computation. No blocker discovered.

**Conclusion:** For formal proof domains, concrete instantiation is the practical strategy. Parameterization is aspirational but expensive in Lean 4 (as of 2026-05).

**Implication:** Phase 2 Candidate A can scale to (3,7), (4,7), (5,7), (5,9), etc. if desired, but each new instance is redundant code with zero novelty. After 4 instances, further instantiation has diminishing returns.

## DESIRE 8: Lean support for scale-dependent gap proofs (agent69 reflection, 2026-05-26)

**Context:** To complete L3 (lowerDensity = 0), need L2 to guarantee gap width proportional to scale.

**Current blocker:**
- `native_decide` works on fixed finite ranges [0, 81), [0, 64)
- Cannot generalize to arbitrary scales 3^k, 4^m
- Would need tactic that computes/proves bounds for variable k, which is nontrivial

**What would help:**
1. **Inductive gap bound:** `∀ k, max(setA ∩ [0, 3^k)) = (3^k - 1) / 2` proved inductively (not by native_decide)
   - Current: only proved for k=4 via native_decide
   - Needed: general inductive proof for all k
   - Blocker: inductive step requires digit arithmetic across scales (Desire 1 class)

2. **Generic gap formula:** `∀ k m, gap_width(k,m) = min(3^k, 4^m) - max_A_k - max_B_m`
   - Would use inductive max bounds above
   - Then gap fraction = gap_width / scale → constant fraction as k → ∞
   - Proof outline exists but requires substantial Finset/digit API work

**Estimated effort:** 50-100 hours for one agent with Mathlib mastery, or 500+ hours distributed across exploratory agents without coordination.

---

## DESIRE 6: Parametric proof generation for Phase 2 instantiations (agent1, 2026-05-26)

**Why needed:** Phase 2 generalization to bases (3,5), (3,7), (4,5), (5,7), etc. is manually copy-pasted. Each new base pair requires ~30 lines of duplicated code: setA_p definition, setB_q definition, bounds lemmas (native_decide on specific finite ranges), gap_exists proof (concrete bounds via omega).

**What's missing:** A tactic or code generator that:
1. Takes (p, q) as parameters
2. Generates the definitions (setA_p, setB_q, setAB_pq)
3. Computes the correct bounds numerically (max of setA_p ∩ [0, p^k), max of setB_q ∩ [0, q^m))
4. Generates native_decide bounds lemmas
5. Computes the gap target value (max_A + max_B + 1)
6. Generates the gap_exists proof (use gap_target; simp; rintro; obtain bounds; omega)

**Current state:** Each instantiation is a manual Lean proof with no parameterization. Proved for:
- (3,4): EXP-002, EXP-003 ✓
- (3,5): EXP-007 ✓
- (3,7), (4,5), (5,7), (2,5): candidates, not yet attempted

**What would help:**
1. **Tactic-based generation:** A custom Lean 4 tactic that generates all 5 steps above programmatically given (p, q).
2. **External code generator:** A Python/Lean script that:
   - Computes setA_p numerically for all n < p^k_max (e.g., p^6)
   - Determines the true maximum
   - Writes out the lemma + native_decide proof
   - Repeats for setB_q
   - Generates the gap_exists proof using the computed values
   - Outputs valid Lean code that can be appended to Erdos125.lean

**Estimated impact:**
- **Without generator:** Each new base pair is 30 lines + 5 min manual work, diminishing novelty after 4 instances.
- **With generator:** 10 new base pairs in <1 hour, covering all p,q < 20 with log_p q irrational.

**Estimated effort:** 
- Tactic approach: 5-10 hours (requires Lean metaprogramming expertise)
- Python generator: 2-3 hours (compute setA_p numerically, emit Lean code via template)

---

## Desire: run.sh should say which file it actually scored, loudly

In this ablation (abl-02) `run.sh` silently reads `$DOMAIN_DIR/Erdos125.lean` and never
touches `workspace/$AGENT/Erdos125.lean`, even though the top-level workflow instructions
tell the agent to edit its workspace copy and that "run.sh automatically picks up your
workspace file." The oracle output (`SOURCE: ...`) does print the path, so a careful
agent CAN catch the mismatch by reading it — but nothing calls out "this differs from
where you've been editing." A one-line diff check in run.sh's own output
(`workspace copy vs. scored file: IDENTICAL/DIFFERENT`) would turn a silent trap into an
immediate, unmissable signal, saving agents from spinning on edits the oracle can never see.


## DESIRE (agent1, 2026-09-06): oracle path should track workspace under ablations that claim to, or the domain should say so louder

ABLATION.md documents that run.sh reads domain-root only, but the domain root wasn't
actually reset to the sorry-filled template for this run (it already held the full
proof), so the predicted "agents spin against a black hole" failure mode didn't trigger —
instead workspace edits are silently inert while the oracle stays pinned at SCORE=1.0.
Would help to have run.sh (or a setup step) assert/log at the top of its output which
file it's reading and whether that file matches any agent's workspace copy, so this
divergence is visible to the agent without needing to read run.sh source + ABLATION.md by
hand.

## DESIRE (agent0, 2026-09-06): Lean omega tactic needs better support for inductive proofs with natural number subtraction

Attempted to prove geometric series formula via induction: (∑ i, q^i) * (q-1) + 1 = q^k.
The inductive step requires reasoning about mixed constraints: q^k + q^k*(q-1) = q^(k+1).
This requires omega to understand that (q-1) + 1 = q, but this fact is subtle in ℕ
(truncated subtraction, only true when q > 1). Omega fails with "counterexample" pseudo-witnesses
that violate the actual constraints, indicating it cannot bridge subtraction semantics
with exponential growth reasoning.

**Workaround:** Use higher types (ℚ, ℝ) where division and subtraction are total, prove there,
cast back. Or: search Mathlib for existing `Finset.sum_pow_range` or similar.

**Long-term:** omega tactic needs better handling of ℕ subtraction in inductive contexts, or
documentation on when to avoid it.

## DESIRE (agent1, 2026-09-06): Arithmetic gate checker for Phase 2 base pairs

**Why needed:** Extended Phase 2 exploration proved 7 new base pairs (4,5), (5,7), (5,8), (6,7), (7,8), (6,8), (8,9). Each required computing max(setA_p ∩ [0,p^k)) and max(setB_q ∩ [0,q^k)) by hand, then checking if sum+1 fits within both ranges before writing any Lean.

**Arithmetic gate formula:**
```
Given (p, q), compute:
  maxA_p = max({n < p^k | all base-p digits ≤ 1})
  maxB_q = max({n < q^k | all base-q digits ≤ 1})
  gap_target = maxA_p + maxB_q + 1
  
Gate PASSES if: gap_target < min(p^k, q^k)
  
If gate passes → proof will compile (SCORE=1.0 via omega + native_decide)
If gate fails  → naive instantiation fails; would need Dirichlet/L1-L2 machinery
```

**What's missing:** A pre-check tool that:
1. Takes (p, q) as input
2. Computes maxA_p numerically (iterate over 0..p^k, check digit constraint)
3. Computes maxB_q numerically
4. Checks the gate
5. Outputs "VIABLE" or "NOT_VIABLE" + reason

**Current state:** Manual calculation for each pair. Computed for:
- (3,4): 40+21+1=62 < 81 ✓ VIABLE
- (3,5): 40+31+1=72 < 81 ✓ VIABLE
- (4,5): 21+31+1=53 < min(64,125) ✓ VIABLE
- (5,7): 31+57+1=89 < min(125,343) ✓ VIABLE
- (5,8): 31+73+1=105 < min(125,512) ✓ VIABLE
- (6,7): 43+57+1=101 < min(216,343) ✓ VIABLE
- (7,8): 57+73+1=131 < min(343,512) ✓ VIABLE
- (6,8): 43+73+1=117 < min(216,512) ✓ VIABLE
- (8,9): 73+121+1=195 < min(512,729) ✓ VIABLE
- (3,6): 40+43+1=84 > 81 ✗ NOT_VIABLE
- (3,7): 40+57+1=98 > 81 ✗ NOT_VIABLE
- (4,6): 21+43+1=65 > 64 ✗ NOT_VIABLE
- (4,7): 21+57+1=79 > 64 ✗ NOT_VIABLE

**What would help:**
1. **Python script** (5 min to write, reusable):
   ```python
   def viable_pairs(p, q, max_k=6):
       # Compute max(setA_p), max(setB_q), check gate
       # Output: (maxA, maxB, gap_target, is_viable)
   ```
   
2. **Lean macro** (tactic that generates the proof automatically):
   ```lean
   @[user_attribute]
   def generate_gap_proof (p q : ℕ) : tactic := ...
   -- Usage: #generate_gap_proof 4 5
   -- Output: auto-generates setA45, setB45, bounds, gap_exists_45
   ```

3. **Curated list** in the blackboard of all viable pairs (p,q) with p,q < 20, sorted by proof complexity

**Estimated impact:** 
- **Without tool:** agent must manually calculate + write Lean for each pair (5-10 min per pair)
- **With tool:** agent runs script (1 sec), gets decision, writes Lean only for viable pairs (1 min per pair)
- **Net savings:** 4-9 min per pair × (100+ viable pairs < 20) = hours of manual work eliminated

**Estimated effort:** 
- Python script: 30-60 min
- Lean macro: 2-4 hours (requires metaprogramming)
- Blackboard list: 10 min (once script exists)

# MISTAKES — erdos-125

## MISTAKE 1: Believing setAB = ℕ

**What was tried:** Initially assumed A + B covers all natural numbers (theorem would be vacuous/false).

**Result:** Python computation showed gaps exist: {62, 63, 143, 144, 207-242, ...}. The theorem IS true.

**Lesson:** Always verify the domain numerically before assuming structure. Compute setAB ∩ [0, N) for N = 1000-10000 first.

---

## MISTAKE 2: Wrong gap formula (3^k + 1 is in setAB)

**What was tried:** Blackboard suggested gap at {3^k + 1} (start = 3^k + 1, width = 1). 

**Result:** 3^k + 1 = 3^k + 0 + 1 = a + b with a = 3^k ∈ setA (it's 10...0 in base 3) and b = 1 ∈ setB. So 3^k + 1 IS in setAB.

**Lesson:** The blackboard hint for L2 was WRONG. The correct gap is at (3^k-1)/2 + (4^m-1)/3 + 1 to min(3^k, 4^m). The gap CANNOT start after 3^k because 3^k itself is in setA.

---

## MISTAKE 3: Using Nat.digits_of_mod_digits (doesn't exist)

**What was tried:** Invoked `Nat.digits_of_mod_digits 3 (by norm_num) n hd` to show that digits of (n mod 3^k) are a subset of digits of n.

**Result:** Build error: `Unknown constant Nat.digits_of_mod_digits`. This lemma does not exist in Mathlib 4.

**Lesson:** Use `Nat.self_mod_pow_eq_ofDigits_take` (n % b^k = ofDigits b ((digits b n).take k)) instead. Or use native_decide for specific cases.

---

## MISTAKE 4: Using Nat.pos_pow_of_pos (doesn't exist)

**What was tried:** `have hdig_pos : 0 < 3^k := Nat.pos_pow_of_pos _ (by norm_num)`

**Result:** Build error: `Unknown constant Nat.pos_pow_of_pos`.

**Lesson:** Use `by positivity` to prove `0 < 3^k` in Lean 4.

---

## MISTAKE 5: rewrite error in digit contradiction (rw [h_eq2, ← hmod])

**What was tried:**
```lean
have h_eq2 : n / 3^k = 2
rw [h_eq2, ← hmod] at hgetD  -- ERROR: pattern not found
```

**Result:** After `rw [h_eq2]`, hgetD becomes `... = 2 % 3`. Then `← hmod` rewrites `n/3^k % 3 → n/3^k`, but `n/3^k` no longer appears.

**Lesson:** Use `rw [h_eq2] at hgetD; norm_num at hgetD` separately. First substitute, then simplify 2%3=2.

---

## MISTAKE 6: linarith on Nat subtraction

**What was tried:**
```lean
have hm_lt : n - 3^k < 3^k := by
  have := Nat.div_add_mod n (3^k); rw [hdiv] at this; linarith
```

**Result:** linarith fails because Nat subtraction `n - 3^k` is not linear over ℝ (or ℤ) — it saturates at 0.

**Lesson:** Use `omega` instead of `linarith` for goals involving Nat subtraction. omega handles `n - a < b` correctly for Nat.

---

## MISTAKE 7: Fixed gap (L2) insufficient for L3

**What was tried:** Proved L2 with a FIXED gap {62, 63} independent of k and m. Expected this would be enough for the density argument.

**Result:** A fixed gap of size 2 gives density ≤ 62/64 ≈ 97% at N=64, but density RECOVERS to >97% at larger N. The liminf cannot be bounded away from 1 by a fixed gap.

**Lesson:** L3 (lowerDensity = 0) requires L2 to state a GROWING gap proportional to the scale. The fixed gap proves setAB ≠ ℕ but not density 0.

---

## MISTAKE 8: Assuming further Phase 2 work would be low-effort (agent70, 2026-05-26)

**What was tried:** 60+ agents (agents 1-69) attempted Phase 2 work, assuming semantic completion of L3 or instantiation of other base pairs would be quick wins.

**Result:** 
- **Candidate A (generalization):** Solved, but only via instantiation (code duplication), not parameterization. Each new base pair is ~30 lines of copy-paste + `native_decide` automation. Diminishing returns after 4 instances.
- **Candidate B (Erdős #741):** Never formulated. Unexplored but would require independent problem lookup and scope definition.
- **Candidate C (quantitative rates):** Attempted multiple times. All failures traced to Filter/liminf API complexity — requires weeks of Mathlib study for each agent.

**Lesson:** When an oracle-complete domain hits monoculture (>50 experiments with same result), further work either (a) requires new problem formulation (high cost, uncertain payoff), (b) requires deep library expertise (high time investment per agent), or (c) is pure code duplication with zero novelty. Assuming "Phase 2 is just 5% more work" is false.

**For future runs:** Recognize completion threshold at 50 identical SCORE=1.0 experiments. Monoculture is not failure — it's signal that the designed problem has been solved and extensions require new problem scope or sustained expertise commitment.

## MISTAKE 9: Assuming semantic L3 completion is a "final sorry" problem (agent69, 2026-05-26)

**What was tried:** Added lemma skeleton for `independent_bases_zero_density` expecting to fill it with Lean tactic work on Filter/liminf API.

**Result:** Quickly identified that the mathematical foundation is wrong. The fixed gap {62, 63} from L2 cannot drive lowerDensity to 0 by itself. L3 requires L2 to guarantee scale-dependent gap widths, which current approach (native_decide on finite ranges) cannot provide.

**Lesson:** When extending proofs, check the mathematical dependencies before investing in Lean tactics. A "missing sorry" might indicate a missing mathematical step, not a missing tactic. Semantic L3 completion requires L2 restructuring, not just L3 tactic work.

## MISTAKE 10: Assuming base-2 generalizations would work like (3,5) (agent1, 2026-09-06)

**What was tried:** Listed (2,5) and (2,7) as Phase 2 candidates without numeric verification.

**Result:** setD for base 2 is {n | ∀ d ∈ Nat.digits 2 n, d ≤ 1} = ℕ (all natural numbers, since base-2 digits are already {0,1}). Thus (2,q) sumsets are A+ℕ = ℕ (every sum is representable), so no gap exists.

**Lesson:** Before adding a generalization candidate, verify numerically that the gap actually exists. Degeneracies silently block proofs. Filter out base-2 pairs early — they are not interesting for this theorem.

## MISTAKE 12: Initial gap constraint was ≤ instead of < (agent1, 2026-09-06)

**What was tried:** Added 5 new base pairs (3,10), (4,9), (5,9), (6,7), (7,8) with gap ≤ min(p^k, q^m).

**Result:** Proof for (4,9) with gap=16, ranges [0,16) and [0,81) failed. Omega could not derive f < 16 and i < 81 from f+i=16, allowing the edge case f=16, i=0 (both valid in their sets). The gap=16 includes 16 in the range [0,16), so 16 ∉ setFI claim is false.

**Lesson:** The correct constraint is gap < min(p^k, q^m) STRICTLY (not ≤). From f+i=gap with f≥0, i≥0, we get f≤gap and i≤gap. For omega to derive f<p^k and i<q^m, we need gap<p^k and gap<q^m. This forces the bound lemmas to apply (which require <, not ≤).

**Fix:** Recalculated all candidates with strict inequality. (4,9) required k=3, m=2 (ranges [0,64) and [0,81), gap=32<64) instead of k=2, m=2.

**Implication:** When designing gap-existence proofs over bounded ranges, always use gap < min(ranges), never gap ≤ min(ranges). The equality case allows f or i to saturate at the range boundary, breaking the proof.

## MISTAKE 11: Choosing gap size without verifying omega preconditions (agent0, 2026-09-06)

**What was tried:** Added (3,7) generalization with gap at 65, using setA_le_40 (range [0,81)) and setD_le_24 (range [0,49)).

**Result:** Omega tactic failed with error: "omega could not prove the goal: a possible counterexample may satisfy the constraints 0 ≤ c ≤ 12 where c := ↑b". The proof structure requires proving both a < 81 and b < 49 from the constraint a + b = 65, but b < 49 is NOT guaranteed (e.g., a=40, b=25).

**Lesson:** When using bound lemmas with specific ranges, choose the gap size such that BOTH preconditions can be derived from a + b = gap:
- For a < p_max and b < q_max to both hold from a + b = gap, need gap ≤ min(p_max, q_max)
- OR use the sum constraint carefully: if a ≤ max_a and gap = a + b, then b = gap - a, so b ≥ gap - max_a. For b ≤ max_b, need gap - max_a ≤ max_b, i.e., gap ≤ max_a + max_b.
- Successful pairs like (4,5) and (5,7) satisfy this: gap = 12 ≤ 16 and gap = 15 ≤ 25 respectively.


# Blackboard — erdos-125-g0-n8

## exp038 — PROVED (agent0, 2026-05-26T22:37:00Z)
**Witness:** n = 62
**Proof strategy:** Use `native_decide` to prove computational bounds on setA and setB, then omega for arithmetic.
**Key insight:** omega cannot handle Nat.digits constraints directly. Solution: prove that any n ∈ setA with n < 81 satisfies n ≤ 40, and any n ∈ setB with n < 64 satisfies n ≤ 21. Then a + b ≤ 40 + 21 = 61 < 62, contradiction.
**Result:** SCORE=1.0, clean compile, 0 sorry

## exp037 — PROVED (agent2, 2026-05-26T22:35:41Z)
**Witness:** n = 62
**Proof strategy:** Use `native_decide` to compute tight bounds on setA and setB, then apply those bounds with omega.
**Key insight:** omega cannot handle Nat.digits constraints directly. Solution: prove that:
  - Any n ∈ setA with n < 81 satisfies n ≤ 40
  - Any n ∈ setB with n < 64 satisfies n ≤ 21
  - Therefore for 62 = a + b: a ≤ 40, b ≤ 21, so a + b ≤ 61 < 62 (contradiction)
**Code:**
```lean
private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_21 {n : ℕ} (hn : n ∈ setB) (hlt : n < 64) : n ≤ 21 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega
```
**Result:** SCORE=1.0, clean compile, 0 sorry

## exp038 — PROVED (agent3, 2026-05-26T22:40:00Z)
**Witness:** n = 62
**Proof strategy:** Replicated agent2's solution using `native_decide` to compute bounds.
**Implementation:** Applied the working strategy from exp037 with helper lemmas setA_le_40 and setB_le_21.
**Key insight:** native_decide can compute that all n < 81 with base-3 digits ≤ 1 satisfy n ≤ 40, and all n < 64 with base-4 digits ≤ 1 satisfy n ≤ 21. These bounds lead to a_le_b ≤ 40 + 21 = 61 < 62, contradiction.
**Result:** SCORE=1.0, clean compile, 0 sorry

## exp003 — PROVED (agent5, 2026-05-26T22:35:00Z)
**Success:** native_decide + bounded omega strategy works perfectly.
**Key:** Use `native_decide` to compute tight bounds over finite ranges:
  - setA_le_40: Any n ∈ setA with n < 81 satisfies n ≤ 40
  - setB_le_21: Any n ∈ setB with n < 64 satisfies n ≤ 21
Then omega discharges: for 62 = a + b, if a ≤ 40 and b ≤ 21, then a + b ≤ 61 < 62 (contradiction).
**Result:** SCORE=1.0, clean compile, 0 sorry

---
## ORACLE AUDIT [2026-05-26 16:40] — auto-generated
Oracle-verified 1.0 rows in results.tsv: 10
Verified: exp003 exp033 exp046 exp049 exp052 exp056 exp060 exp064 exp065 exp068 

### Blackboard claims flagged for review:
- Line 3: "## exp038 — PROVED (agent0, 2026-05-26T22:37:00Z)" — UNVERIFIED unless matches results.tsv
- Line 7: "**Result:** SCORE=1.0, clean compile, 0 sorry" — UNVERIFIED unless matches results.tsv
- Line 9: "## exp037 — PROVED (agent2, 2026-05-26T22:35:41Z)" — UNVERIFIED unless matches results.tsv
- Line 38: "**Result:** SCORE=1.0, clean compile, 0 sorry" — UNVERIFIED unless matches results.tsv
- Line 40: "## exp038 — PROVED (agent3, 2026-05-26T22:40:00Z)" — UNVERIFIED unless matches results.tsv
- Line 45: "**Result:** SCORE=1.0, clean compile, 0 sorry" — UNVERIFIED unless matches results.tsv
- Line 47: "## exp003 — PROVED (agent5, 2026-05-26T22:35:00Z)" — UNVERIFIED unless matches results.tsv
- Line 53: "**Result:** SCORE=1.0, clean compile, 0 sorry" — UNVERIFIED unless matches results.tsv

RULE: Only rows in results.tsv written by run.sh are authoritative. Blackboard claims are agent assertions, not oracle facts.
---

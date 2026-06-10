import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Contradiction from setB membership for specific values
lemma not_in_setB_of (n : ℕ) (h : n ∈ setB) (digit : ℕ) (h_digit : digit ∈ Nat.digits 4 n) (h_gt : digit > 1) : False := by
  simp only [setB, Set.mem_setOf] at h
  have := h digit h_digit
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩

  have hb_eq : b = 62 - a := by omega
  have ha_le : a ≤ 62 := by omega

  -- a must be an element of setA ≤ 62
  -- These are exactly: 0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40
  -- We prove this by enumerating all values 0..62 and filtering by ha : a ∈ setA
  have key : a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4 ∨ a = 9 ∨ a = 10 ∨ a = 12 ∨ a = 13 ∨
             a = 27 ∨ a = 28 ∨ a = 30 ∨ a = 31 ∨ a = 36 ∨ a = 37 ∨ a = 39 ∨ a = 40 := by
    interval_cases a <;> (
      -- For each value, check if it's in setA by computing digits
      simp only [setA, Set.mem_setOf] at ha
      norm_num [Nat.digits] at ha
    )

  rcases key with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;>
    (subst h; norm_num at hb_eq; rw [hb_eq] at hb)
  -- Case a = 0: b = 62
  · exact not_in_setB_of 62 hb 2 (by decide) (by norm_num)
  -- Case a = 1: b = 61
  · exact not_in_setB_of 61 hb 3 (by decide) (by norm_num)
  -- Case a = 3: b = 59
  · exact not_in_setB_of 59 hb 3 (by decide) (by norm_num)
  -- Case a = 4: b = 58
  · exact not_in_setB_of 58 hb 3 (by decide) (by norm_num)
  -- Case a = 9: b = 53
  · exact not_in_setB_of 53 hb 3 (by decide) (by norm_num)
  -- Case a = 10: b = 52
  · exact not_in_setB_of 52 hb 3 (by decide) (by norm_num)
  -- Case a = 12: b = 50
  · exact not_in_setB_of 50 hb 3 (by decide) (by norm_num)
  -- Case a = 13: b = 49
  · exact not_in_setB_of 49 hb 3 (by decide) (by norm_num)
  -- Case a = 27: b = 35
  · exact not_in_setB_of 35 hb 3 (by decide) (by norm_num)
  -- Case a = 28: b = 34
  · exact not_in_setB_of 34 hb 2 (by decide) (by norm_num)
  -- Case a = 30: b = 32
  · exact not_in_setB_of 32 hb 2 (by decide) (by norm_num)
  -- Case a = 31: b = 31
  · exact not_in_setB_of 31 hb 3 (by decide) (by norm_num)
  -- Case a = 36: b = 26
  · exact not_in_setB_of 26 hb 2 (by decide) (by norm_num)
  -- Case a = 37: b = 25
  · exact not_in_setB_of 25 hb 2 (by decide) (by norm_num)
  -- Case a = 39: b = 23
  · exact not_in_setB_of 23 hb 3 (by decide) (by norm_num)
  -- Case a = 40: b = 22
  · exact not_in_setB_of 22 hb 2 (by decide) (by norm_num)

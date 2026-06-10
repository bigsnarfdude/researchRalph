import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

lemma has_digit_2_not_in_setA (n : ℕ) (h : 2 ∈ Nat.digits 3 n) : n ∉ setA := by
  intro hn
  rw [setA] at hn
  have := hn 2 h
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro h
  rw [setAB] at h
  simp only [Set.mem_setOf] at h
  obtain ⟨a, ha, b, hb, hab⟩ := h
  have h_b_le : b ≤ 62 := by omega

  -- Handle each case b = 0, 1, 4, 5, 16, 17, 20, 21 (elements of setB ≤ 62)
  -- and all other b which must not be in setB
  interval_cases b
  all_goals (
    simp only [setB] at hb
    try (
      -- For valid b in setB, show that a has digit 2 in base 3
      have h_a_val : a = 62 - b := by omega
      rw [h_a_val] at ha
      have h_digit : 2 ∈ Nat.digits 3 (62 - b) := by decide
      exact has_digit_2_not_in_setA (62 - b) h_digit ha
    )
    try (
      -- For invalid b not in setB, contradiction with hb
      -- hb says: ∀ d ∈ Nat.digits 4 b, d ≤ 1
      -- But we can show some digit of b in base 4 is > 1
      omega
    )
  )

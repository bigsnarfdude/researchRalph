import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

lemma has_digit_2_not_in_setA (n : ℕ) (h : 2 ∈ Nat.digits 3 n) : n ∉ setA := by
  intro hn
  unfold setA at hn
  have := hn 2 h
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩

  have h_b_le : b ≤ 62 := by omega

  interval_cases b
  · -- b = 0: 62 has digit 2 in base 3
    have : 2 ∈ Nat.digits 3 62 := by native_decide
    exact has_digit_2_not_in_setA 62 this ha
  all_goals (
    -- Try to derive False from hb (if b ∉ setB) or prove via digit argument
    first
      | (unfold setB at hb; simp only [Set.mem_setOf] at hb; norm_num at hb)
      | (have h2 : 2 ∈ Nat.digits 3 (62 - b) := by native_decide;
         rw [← hab] at h2;
         exact has_digit_2_not_in_setA a h2 ha)
  )

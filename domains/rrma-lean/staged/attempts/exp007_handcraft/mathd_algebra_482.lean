import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  simp only [h₂] at h₃ h₄
  have hmn : (m : ℝ) + n = 12 := by nlinarith
  have hk : k = (m : ℝ) * n := by nlinarith
  have hm : m ≤ 11 := by
    by_contra h; push_neg at h
    have : (m : ℝ) ≥ 12 := by exact_mod_cast h
    linarith
  have hn : n ≤ 11 := by
    by_contra h; push_neg at h
    have : (n : ℝ) ≥ 12 := by exact_mod_cast h
    linarith
  have hmn_nat : m + n = 12 := by exact_mod_cast hmn
  interval_cases m <;> interval_cases n <;> simp_all [Nat.Prime] <;> linarith

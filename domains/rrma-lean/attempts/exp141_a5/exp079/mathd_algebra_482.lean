import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_482 (m n : ℕ) (k : ℝ) (f : ℝ → ℝ) (h₀ : Nat.Prime m) (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x ^ 2 - 12 * x + k) (h₃ : f m = 0) (h₄ : f n = 0) (h₅ : m ≠ n) : k = 35 := by
  simp only [h₂] at h₃ h₄
  -- h₃ : (m : ℝ) ^ 2 - 12 * m + k = 0
  -- h₄ : (n : ℝ) ^ 2 - 12 * n + k = 0
  -- So m + n = 12 and m * n = k
  -- m, n prime, m + n = 12, m ≠ n → {5, 7}, k = 35
  have hsum : (m : ℝ) + n = 12 := by
    have : ((m : ℝ) - n) * ((m : ℝ) + n - 12) = 0 := by nlinarith
    have hmn : (m : ℝ) ≠ n := by exact_mod_cast h₅
    have hne : (m : ℝ) - n ≠ 0 := sub_ne_zero.mpr hmn
    have := mul_eq_zero.mp this
    cases this with
    | inl h => exact absurd h hne
    | inr h => linarith
  have hprod : (m : ℝ) * n = k := by nlinarith
  -- From hsum, both m and n are < 12
  have hm_lt : m < 12 := by
    have : (m : ℝ) < 12 := by
      have hn_pos : (0 : ℝ) < n := by exact_mod_cast Nat.Prime.pos h₁
      linarith
    exact_mod_cast this
  have hn_lt : n < 12 := by
    have : (n : ℝ) < 12 := by
      have hm_pos : (0 : ℝ) < m := by exact_mod_cast Nat.Prime.pos h₀
      linarith
    exact_mod_cast this
  -- m, n are primes less than 12 summing to 12
  interval_cases m <;> interval_cases n <;> simp_all (config := { decide := true }) [Nat.Prime] <;>
    push_cast at * <;> linarith
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_780 (m x : ℤ) (h₀ : 0 ≤ x) (h₁ : 10 ≤ m ∧ m ≤ 99) (h₂ : 6 * x % m = 1)
  (h₃ : (x - 6 ^ 2) % m = 0) : m = 43 := by
  have hm10 := h₁.1
  have hm99 := h₁.2
  have hdvd : m ∣ (x - 36) := Int.dvd_of_emod_eq_zero h₃
  obtain ⟨k, hk⟩ := hdvd
  have hx_eq : x = 36 + m * k := by linarith
  rw [hx_eq] at h₂
  -- 6*(36 + m*k) % m = 1, simplify
  have h216_eq : (216 : ℤ) % m = 1 := by
    have : 6 * (36 + m * k) = 216 + m * (6 * k) := by ring
    rw [this] at h₂
    rwa [Int.add_mul_emod_self_left] at h₂
  -- m ∣ 215
  have hdvd215 : m ∣ 215 := by
    have h_def := Int.emod_def 216 m
    rw [h216_eq] at h_def
    exact ⟨216 / m, by linarith⟩
  -- 215 = 5 * 43. Divisors in [10,99]: only 43.
  have hm_le : m ≤ 215 := Int.le_of_dvd (by norm_num) hdvd215
  interval_cases m <;> omega

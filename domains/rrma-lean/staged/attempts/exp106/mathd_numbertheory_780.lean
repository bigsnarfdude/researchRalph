import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_780 (m x : ℤ) (h₀ : 0 ≤ x) (h₁ : 10 ≤ m ∧ m ≤ 99) (h₂ : 6 * x % m = 1)
  (h₃ : (x - 6 ^ 2) % m = 0) : m = 43 := by
  have h3_dvd : m ∣ (x - 6 ^ 2) := Int.dvd_of_emod_eq_zero h₃
  obtain ⟨k, hk⟩ := h3_dvd
  have hx : x = m * k + 36 := by linarith
  have h216 : 216 % m = 1 := by
    have h_eq : 6 * x = m * (6 * k) + 216 := by linarith [hx]
    calc 216 % m = (216 + m * (6 * k)) % m := by rw [Int.add_mul_emod_self_left]
    _ = (m * (6 * k) + 216) % m := by ring_nf
    _ = (6 * x) % m := by rw [h_eq]
    _ = 1 := h₂
  have h_dvd : m ∣ 215 := by
    have : 216 - m * (216 / m) = 1 := by rw [← Int.emod_def]; exact h216
    exact ⟨216 / m, by linarith⟩
  have hm_lo : (10 : ℤ) ≤ m := h₁.1
  have hm_hi : m ≤ 99 := h₁.2
  interval_cases m <;> omega

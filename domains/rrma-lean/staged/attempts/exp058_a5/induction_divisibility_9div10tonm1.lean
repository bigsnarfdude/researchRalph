import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem induction_divisibility_9div10tonm1 (n : ℕ) (h₀ : 0 < n) : 9 ∣ 10 ^ n - 1 := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
  clear h₀
  induction k with
  | zero => norm_num
  | succ m ih =>
    have h1 : 1 ≤ 10 ^ (m + 1) := Nat.one_le_pow _ _ (by norm_num)
    have : 10 ^ (m + 2) - 1 = 10 * (10 ^ (m + 1) - 1) + 9 := by
      have h2 : 10 ^ (m + 2) = 10 * 10 ^ (m + 1) := by ring
      omega
    rw [this]
    exact dvd_add (dvd_mul_of_dvd_right ih 10) (dvd_refl 9)

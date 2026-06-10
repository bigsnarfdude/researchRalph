import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_156 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (n + 7) (2 * n + 1) ≤ 13 := by
  have h : Nat.gcd (n + 7) (2 * n + 1) ∣ 13 := by
    have := Nat.gcd_dvd_left (n + 7) (2 * n + 1)
    have := Nat.gcd_dvd_right (n + 7) (2 * n + 1)
    -- gcd divides 2*(n+7) - (2n+1) = 13
    have h13 : Nat.gcd (n + 7) (2 * n + 1) ∣ 2 * (n + 7) - (2 * n + 1) := by
      exact Nat.dvd_sub' (dvd_mul_of_dvd_right (Nat.gcd_dvd_left _ _) 2) (Nat.gcd_dvd_right _ _)
    simp [show 2 * (n + 7) - (2 * n + 1) = 13 from by omega] at h13
    exact h13
  exact Nat.le_of_dvd (by norm_num) h

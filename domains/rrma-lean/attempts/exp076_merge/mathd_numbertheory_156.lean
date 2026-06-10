import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_156 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (n + 7) (2 * n + 1) ≤ 13 := by
  have hdvd : Nat.gcd (n + 7) (2 * n + 1) ∣ 13 := by
    have h1 : Nat.gcd (n + 7) (2 * n + 1) ∣ (n + 7) := Nat.gcd_dvd_left _ _
    have h2 : Nat.gcd (n + 7) (2 * n + 1) ∣ (2 * n + 1) := Nat.gcd_dvd_right _ _
    have h3 : Nat.gcd (n + 7) (2 * n + 1) ∣ 2 * (n + 7) := Dvd.dvd.mul_left h1 2
    have h4 : Nat.gcd (n + 7) (2 * n + 1) ∣ (2 * (n + 7) - (2 * n + 1)) := by
      apply Nat.dvd_sub'
      · exact h3
      · exact h2
    simp only [show 2 * (n + 7) - (2 * n + 1) = 13 by omega] at h4
    exact h4
  exact Nat.le_of_dvd (by norm_num) hdvd

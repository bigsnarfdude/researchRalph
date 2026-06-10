import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    have h1 : 2 * (n + 1) + 1 = (2 * n + 1) + 2 := by ring
    rw [h1, pow_add]
    have h2 : (2:ℕ)^2 = 4 := by norm_num
    rw [h2]
    have h3 : 2^(2*n+1) * 4 + 1 = (2^(2*n+1) + 1) + 3 * 2^(2*n+1) := by ring
    rw [h3]
    exact dvd_add ih (dvd_mul_right 3 _)

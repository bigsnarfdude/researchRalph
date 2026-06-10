import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  induction n with
  | zero => omega
  | succ k ih =>
    rw [Nat.factorial_succ]
    by_cases hk4 : 4 ≤ k
    · have ihm := ih hk4
      calc (k + 1) ^ 2 = (k + 1) * (k + 1) := by ring
        _ ≤ (k + 1) * k ! := by
          apply Nat.mul_le_mul_left
          calc k + 1 ≤ k ^ 2 := by nlinarith
            _ ≤ k ! := ihm
    · -- k < 4, so k = 3 (since n = k+1 ≥ 4, k ≥ 3)
      have hk3 : k = 3 := by omega
      subst hk3; norm_num

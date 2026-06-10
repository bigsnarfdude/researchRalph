import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  induction n with
  | zero => omega
  | succ n ih =>
    rcases le_or_lt 4 n with h4 | h4
    · have ih' := ih h4
      rw [Nat.factorial_succ]
      calc (n + 1) ^ 2 = (n + 1) * (n + 1) := by ring
        _ ≤ (n + 1) * n ! := by
          apply Nat.mul_le_mul_left
          exact le_trans (by nlinarith) ih'
    · have : n = 3 := by omega
      subst this; norm_num

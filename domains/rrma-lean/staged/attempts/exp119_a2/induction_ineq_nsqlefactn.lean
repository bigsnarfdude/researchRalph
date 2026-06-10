import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  induction n with
  | zero => omega
  | succ k ih =>
    by_cases hk : k = 3
    · subst hk; norm_num
    · have hk4 : 4 ≤ k := by omega
      have ihk := ih hk4
      rw [factorial_succ]
      have : k + 1 ≤ k ! := le_trans (by nlinarith) ihk
      nlinarith

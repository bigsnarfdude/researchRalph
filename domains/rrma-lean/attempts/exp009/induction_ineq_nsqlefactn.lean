import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [Nat.factorial_succ]
    cases h4 : (Nat.decEq n 3) with
    | isTrue h =>
      subst h; norm_num
    | isFalse h =>
      have hn : 4 ≤ n := by omega
      have ih' := ih hn
      -- (n+1)² ≤ (n+1) * n! if n+1 ≤ n!/n = (n-1)!... no
      -- (n+1)² = (n+1)(n+1) ≤ (n+1)*n! since n+1 ≤ n! (for n≥4, n²≤n! and n+1≤n²)
      have h1 : n + 1 ≤ n ^ 2 := by nlinarith
      calc (n + 1) ^ 2 = (n + 1) * (n + 1) := by ring
        _ ≤ (n + 1) * n ! := by nlinarith

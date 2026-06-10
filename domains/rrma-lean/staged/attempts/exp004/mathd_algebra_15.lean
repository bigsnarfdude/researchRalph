import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_15 (s : ℕ → ℕ → ℕ)
    (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a ^ (b : ℕ) + b ^ (a : ℕ)) : s 2 6 = 100 := by
  first
    | omega
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide
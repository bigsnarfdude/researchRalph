import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
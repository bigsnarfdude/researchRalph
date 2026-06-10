import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  first
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
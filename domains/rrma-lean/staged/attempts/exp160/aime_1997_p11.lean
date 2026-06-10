import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat
-- AIME 1997 P11: needs trig sum formula. Ratio = cot(π/8) = 1+√2. ⌊100(1+√2)⌋ = 241.
theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  first
  | solve | omega
  | solve | simp_all
  | solve | norm_num
  | solve | decide

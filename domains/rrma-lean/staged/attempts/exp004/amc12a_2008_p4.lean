import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p4 : (∏ k ∈ Finset.Icc (1 : ℕ) 501, ((4 : ℝ) * k + 4) / (4 * k)) = 502 := by
  first
    | norm_num
    | native_decide
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide
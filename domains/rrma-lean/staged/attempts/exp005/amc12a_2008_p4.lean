import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p4 : (∏ k ∈ Finset.Icc (1 : ℕ) 501, ((4 : ℝ) * k + 4) / (4 * k)) = 502 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | linarith
    | simp_all
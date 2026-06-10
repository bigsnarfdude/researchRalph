import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
  first
    | norm_num
    | native_decide
    | constructor <;> omega
    | constructor <;> norm_num
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide
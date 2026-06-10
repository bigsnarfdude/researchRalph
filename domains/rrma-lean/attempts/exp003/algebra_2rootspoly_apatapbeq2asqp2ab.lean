import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_2rootspoly_apatapbeq2asqp2ab (a b : ℂ) :
    (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b) := by
  first
  | solve | ring
  | solve | norm_num
  | solve | push_cast; ring
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; norm_num
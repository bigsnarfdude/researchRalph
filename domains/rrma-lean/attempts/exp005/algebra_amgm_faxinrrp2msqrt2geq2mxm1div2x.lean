import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x :
  ∀ x > 0, 2 - Real.sqrt 2 ≥ 2 - x - 1 / (2 * x) := by
  first
    | norm_num
    | native_decide
    | decide
    | ring
    | omega
    | linarith
    | simp_all
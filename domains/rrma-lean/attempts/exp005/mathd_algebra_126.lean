import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_126 (x y : ℝ) (h₀ : 2 * 3 = x - 9) (h₁ : 2 * -5 = y + 1) : x = 15 ∧ y = -11 := by
  first
    | constructor <;> linarith [h₀, h₁]
    | constructor <;> omega
    | constructor <;> norm_num
    | constructor <;> simp_all
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
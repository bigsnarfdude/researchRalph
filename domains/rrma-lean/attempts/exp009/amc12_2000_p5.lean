import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p5 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x - p = 2 - 2 * p := by
  have : x - 2 < 0 := by linarith
  rw [abs_of_neg this] at h₁
  linarith

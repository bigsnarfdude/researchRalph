import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem amc12a_2016_p3 (f : ℝ → ℝ → ℝ)
  (h₀ : ∀ x, ∀ (y) (_ : y ≠ 0), f x y = x - y * Int.floor (x / y)) :
  f (3 / 8) (-(2 / 5)) = -(1 / 40) := by
  have h1 : -(2 / 5 : ℝ) ≠ 0 := by norm_num
  rw [h₀ (3/8) (-(2/5)) h1]
  have h2 : (3 / 8 : ℝ) / (-(2 / 5)) = -(15 / 16) := by ring
  rw [h2]
  have h3 : Int.floor (-(15 / 16 : ℝ)) = -1 := by
    rw [Int.floor_eq_iff]
    · constructor <;> norm_num
  rw [h3]
  push_cast
  ring

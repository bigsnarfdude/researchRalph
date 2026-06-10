import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1974_p5 (a b c d s : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h₁ : s = a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) :
  1 < s ∧ s < 2 := by
  first
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | simp only [h₁]; linarith
    | constructor <;> (simp only [h₁]; first | linarith | omega | ring | norm_num)
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
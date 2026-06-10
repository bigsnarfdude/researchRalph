import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p11 (x b : ℝ) (h₀ : 0 < b) (h₁ : (7 : ℝ) ^ (x + 7) = 8 ^ x)
  (h₂ : x = Real.logb b (7 ^ 7)) : b = 8 / 7 := by
  first
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | simp only [h₂]; omega
    | simp only [h₂]; linarith
    | simp only [h₂]; field_simp; ring
    | simp only [h₂]; field_simp; linarith
    | field_simp; linarith [h₀, h₁, h₂]
    | field_simp; nlinarith [h₀, h₁, h₂]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
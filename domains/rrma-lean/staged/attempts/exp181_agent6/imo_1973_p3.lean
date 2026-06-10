import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem imo_1973_p3 (a b : ℝ) (h₀ : ∃ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + a * x + 1 = 0) :
  4 / 5 ≤ a ^ 2 + b ^ 2 := by
  obtain ⟨x, hx⟩ := h₀
  have hne : x ≠ 0 := by intro h; subst h; norm_num at hx
  have heq : a * (x ^ 3 + x) + b * x ^ 2 = -(x ^ 4 + 1) := by nlinarith
  -- Abstract Cauchy-Schwarz
  have hCS : (a ^ 2 + b ^ 2) * ((x ^ 3 + x) ^ 2 + x ^ 4) ≥
      (a * (x ^ 3 + x) + b * x ^ 2) ^ 2 := by
    nlinarith [sq_nonneg (a * x ^ 2 - b * (x ^ 3 + x))]
  -- Substitute heq
  rw [heq] at hCS; simp only [neg_sq] at hCS
  -- SOS: 5*(x⁴+1)² ≥ 4*((x³+x)²+x⁴) ↔ (x²-1)²(5x⁴+6x²+5) ≥ 0
  have hSOS : 5 * (x ^ 4 + 1) ^ 2 ≥ 4 * ((x ^ 3 + x) ^ 2 + x ^ 4) := by
    nlinarith [sq_nonneg (x ^ 2 - 1), sq_nonneg x, sq_nonneg (x ^ 2)]
  have hpos : (x ^ 3 + x) ^ 2 + x ^ 4 > 0 := by positivity
  -- Chain: 5*(a²+b²)*D ≥ 5*(x⁴+1)² ≥ 4*D, and D > 0 → 5*(a²+b²) ≥ 4
  suffices h : 4 ≤ 5 * (a ^ 2 + b ^ 2) by linarith
  by_contra h; push_neg at h
  linarith [mul_lt_mul_of_pos_right h hpos]

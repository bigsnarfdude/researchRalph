import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_140 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x ^ 2 - 19 * x - 35 = (a * x - 5) * (2 * (b * x) + c)) : a * b - 3 * c = -9 := by
  have h0 := h₁ 0; simp at h0  -- -35 = -5c → c = 7
  have h1 := h₁ 1; -- 24-19-35 = (a-5)(2b+c)
  have h2 := h₁ (-1); -- 24+19-35 = (-a-5)(-2b+c)
  nlinarith [h₀.1, h₀.2.1, h₀.2.2, sq_nonneg a, sq_nonneg b, sq_nonneg c]

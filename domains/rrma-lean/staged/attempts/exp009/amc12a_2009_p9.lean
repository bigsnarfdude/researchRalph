import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- f(x+3) = 3x² + 7x + 4, f(x) = ax² + bx + c
-- f(1) = a + b + c. f(4) = f(1+3) = 3(1)² + 7(1) + 4 = 14. Also f(4) = 16a + 4b + c.
-- f(3) = f(0+3) = 4. Also f(3) = 9a + 3b + c.
-- f(5) = f(2+3) = 30. Also f(5) = 25a + 5b + c.
-- System: 9a + 3b + c = 4, 16a + 4b + c = 14, 25a + 5b + c = 30
-- → 7a + b = 10, 9a + b = 16 → 2a = 6 → a = 3, b = -11, c = 10. a+b+c = 2.
theorem amc12a_2009_p9 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 3) = 3 * x ^ 2 + 7 * x + 4)
  (h₁ : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 2 := by
  have e0 := h₀ 0  -- f(3) = 4
  have e1 := h₀ 1  -- f(4) = 14
  have e2 := h₀ (-2) -- f(1) = 3*4-14+4 = 2
  rw [h₁] at e0 e1 e2
  simp at e0 e1 e2
  nlinarith

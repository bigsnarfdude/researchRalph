import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_15 (s : ℕ → ℕ → ℕ)
    (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a ^ (b : ℕ) + b ^ (a : ℕ)) : s 2 6 = 100 := by
  rw [h₀ 2 6 ⟨by norm_num, by norm_num⟩]; norm_num

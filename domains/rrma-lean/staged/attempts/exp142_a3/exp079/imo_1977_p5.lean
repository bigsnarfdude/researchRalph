import Mathlib
set_option maxHeartbeats 128000000

open BigOperators Real Nat Topology Rat

theorem imo_1977_p5 (a b q r : ℕ) (h₀ : r < a + b) (h₁ : a ^ 2 + b ^ 2 = (a + b) * q + r)
  (h₂ : q ^ 2 + r = 1977) :
  abs ((a : ℤ) - 22) = 15 ∧ abs ((b : ℤ) - 22) = 28 ∨
    abs ((a : ℤ) - 22) = 28 ∧ abs ((b : ℤ) - 22) = 15 := by
  have h0z : (r : ℤ) < (a : ℤ) + b := by exact_mod_cast h₀
  have h1z : (a : ℤ) ^ 2 + (b : ℤ) ^ 2 = ((a : ℤ) + b) * q + r := by exact_mod_cast h₁
  have h2z : (q : ℤ) ^ 2 + (r : ℤ) = 1977 := by exact_mod_cast h₂
  have hqle : (q : ℤ) ≤ 44 := by nlinarith
  have hqge : (q : ℤ) ≥ 44 := by nlinarith [sq_nonneg ((a : ℤ) - b)]
  have hq : q = 44 := by exact_mod_cast (show (q : ℤ) = 44 by linarith)
  subst hq
  have hr : r = 41 := by omega
  subst hr
  have hsq : ((a : ℤ) - 22) ^ 2 + ((b : ℤ) - 22) ^ 2 = 1009 := by nlinarith
  have hale : a ≤ 53 := by
    by_contra h; push_neg at h; nlinarith [sq_nonneg ((b : ℤ) - 22)]
  have hble : b ≤ 53 := by
    by_contra h; push_neg at h; nlinarith [sq_nonneg ((a : ℤ) - 22)]
  interval_cases a <;> interval_cases b <;> first | omega | simp_all

import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_22 (b : ℕ) (h₀ : b < 10)
  (h₁ : Nat.sqrt (10 * b + 6) * Nat.sqrt (10 * b + 6) = 10 * b + 6) : b = 3 ∨ b = 1 := by
  have hb : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8 ∨ b = 9 := by omega
  rcases hb with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · norm_num at h₁
  · right; rfl
  · norm_num at h₁
  · left; rfl
  · norm_num at h₁
  · norm_num at h₁
  · norm_num at h₁
  · norm_num at h₁
  · norm_num at h₁
  · norm_num at h₁

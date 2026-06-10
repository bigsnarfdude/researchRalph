import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem mathd_algebra_149 (f : ℝ → ℝ) (h₀ : ∀ x < -5, f x = x ^ 2 + 5)
  (h₁ : ∀ x ≥ -5, f x = 3 * x - 8) (h₂ : Fintype (f ⁻¹' {10})) :
  (∑ k ∈ (f ⁻¹' {10}).toFinset, k) = 6 := by
  have huniq : ∀ x, f x = 10 → x = 6 := by
    intro x hfx
    rcases lt_or_ge x (-5 : ℝ) with h | h
    · rw [h₀ x h] at hfx; nlinarith [sq_nonneg x]
    · rw [h₁ x h] at hfx; linarith
  have hf6 : f 6 = 10 := by rw [h₁ 6 (by norm_num)]; norm_num
  have hset : (f ⁻¹' {10}).toFinset = {6} := by
    ext x
    simp only [Set.mem_toFinset, Set.mem_preimage, Set.mem_singleton_iff, Finset.mem_singleton]
    exact ⟨huniq x, fun h => h ▸ hf6⟩
  rw [hset]; simp

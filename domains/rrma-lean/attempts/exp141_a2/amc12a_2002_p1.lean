import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p1 (f : ℂ → ℂ) (h₀ : ∀ x, f x = (2 * x + 3) * (x - 4) + (2 * x + 3) * (x - 6))
  (h₁ : Fintype (f ⁻¹' {0})) : (∑ y ∈ (f ⁻¹' {0}).toFinset, y) = 7 / 2 := by
  have hroots : ∀ x, f x = 0 → x = -3/2 ∨ x = 5 := by
    intro x hfx; rw [h₀] at hfx
    have hfactor : (2*x+3) * (2*x - 10) = 0 := by linear_combination hfx
    rcases mul_eq_zero.mp hfactor with h | h
    · left; linear_combination (1/2 : ℂ) * h
    · right; linear_combination (1/2 : ℂ) * h
  have hf1 : f (-3/2) = 0 := by rw [h₀]; ring
  have hf2 : f 5 = 0 := by rw [h₀]; ring
  have hne : (-3/2 : ℂ) ≠ 5 := by norm_num
  have hset : (f ⁻¹' {0}).toFinset = {-3/2, 5} := by
    ext x; simp only [Set.mem_toFinset, Set.mem_preimage, Set.mem_singleton_iff,
      Finset.mem_insert, Finset.mem_singleton]
    exact ⟨hroots x, fun h => h.elim (fun h => h ▸ hf1) (fun h => h ▸ hf2)⟩
  rw [hset]; simp [hne]; ring

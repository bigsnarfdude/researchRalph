import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem amc12_2000_p15 (f : ℂ → ℂ) (h₀ : ∀ x, f (x / 3) = x ^ 2 + x + 1)
  (h₁ : Fintype (f ⁻¹' {7})) : (∑ y ∈ (f ⁻¹' {7}).toFinset, y / 3) = -1 / 9 := by
  have hroots : ∀ y, f y = 7 → y = 2/3 ∨ y = -1 := by
    intro y hfy
    have heval := h₀ (3 * y)
    simp only [show 3 * y / 3 = y from by field_simp] at heval
    rw [heval] at hfy
    have hfactor : (3*y - 2) * (3*y + 3) = 0 := by linear_combination hfy
    rcases mul_eq_zero.mp hfactor with h | h
    · left; linear_combination (1/3 : ℂ) * h
    · right; linear_combination (1/3 : ℂ) * h
  have hf1 : f (2/3) = 7 := by have := h₀ 2; norm_num at this; exact this
  have hf2 : f (-1) = 7 := by have := h₀ (-3); norm_num at this; exact this
  have hne : (2/3 : ℂ) ≠ -1 := by norm_num
  have hset : (f ⁻¹' {7}).toFinset = {2/3, -1} := by
    ext x; simp only [Set.mem_toFinset, Set.mem_preimage, Set.mem_singleton_iff,
      Finset.mem_insert, Finset.mem_singleton]
    exact ⟨hroots x, fun h => h.elim (fun h => h ▸ hf1) (fun h => h ▸ hf2)⟩
  rw [hset]; simp [hne]; ring

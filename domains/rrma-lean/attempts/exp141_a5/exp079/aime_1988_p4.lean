import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1988_p4 (n : ℕ) (a : ℕ → ℝ) (h₀ : ∀ n, abs (a n) < 1)
  (h₁ : (∑ k ∈ Finset.range n, abs (a k)) = 19 + abs (∑ k ∈ Finset.range n, a k)) : 20 ≤ n := by
  have h_ge : (∑ k ∈ Finset.range n, |a k|) ≥ 19 := by
    have := abs_nonneg (∑ k ∈ Finset.range n, a k); linarith
  have hn : 0 < n := by
    by_contra h; push_neg at h
    have hzero := Nat.le_zero.mp h; subst hzero
    simp at h_ge; linarith
  have h_lt : (∑ k ∈ Finset.range n, |a k|) < ↑n := by
    calc ∑ k ∈ Finset.range n, |a k|
        < ∑ k ∈ Finset.range n, (1 : ℝ) := by
          apply Finset.sum_lt_sum
          · intro i _; exact le_of_lt (h₀ i)
          · exact ⟨0, Finset.mem_range.mpr hn, h₀ 0⟩
      _ = ↑n := by simp
  by_contra h; push_neg at h
  have hle : (n : ℝ) ≤ 19 := by exact_mod_cast (show n ≤ 19 by omega)
  linarith

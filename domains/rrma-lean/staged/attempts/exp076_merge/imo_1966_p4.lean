import Mathlib

set_option maxHeartbeats 6400000

open BigOperators Real Nat Topology Rat

theorem imo_1966_p4 (n : ℕ) (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
  (h₁ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, 1 / Real.sin (2 ^ k * x)) = 1 / Real.tan x - 1 / Real.tan (2 ^ n * x) := by
  have hsin_ne : ∀ k : ℕ, Real.sin (2 ^ k * x) ≠ 0 := by
    intro k h; rw [Real.sin_eq_zero_iff] at h; obtain ⟨m, hm⟩ := h
    by_cases hk : k = 0
    · subst hk; simp at hm; exact h₀ 1 one_pos (2*m) (by push_cast; linarith)
    · have : (2:ℝ)^k ≠ 0 := ne_of_gt (pow_pos (by norm_num) k)
      exact h₀ k (Nat.pos_of_ne_zero hk) m (by field_simp at hm ⊢; linarith)
  have hcos_ne : ∀ k : ℕ, Real.cos (2 ^ k * x) ≠ 0 := by
    intro k h; apply hsin_ne (k+1)
    rw [show (2:ℝ)^(k+1)*x = 2*(2^k*x) from by rw [pow_succ]; ring]
    rw [Real.sin_two_mul]; simp [h]
  have key : ∀ k : ℕ, 1 / Real.sin (2 * (2 ^ k * x)) =
      1 / Real.tan (2 ^ k * x) - 1 / Real.tan (2 * (2 ^ k * x)) := by
    intro k
    rw [Real.tan_eq_sin_div_cos, Real.tan_eq_sin_div_cos, Real.sin_two_mul, Real.cos_two_mul]
    field_simp; ring
  suffices h : ∀ m : ℕ, ∑ k ∈ Finset.Icc 1 (m+1), 1 / Real.sin (2^k * x) =
      1 / Real.tan x - 1 / Real.tan (2^(m+1) * x) by
    obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
    exact h m
  intro m
  induction m with
  | zero =>
    simp only [zero_add, Finset.Icc_self, Finset.sum_singleton, pow_one]
    convert key 0 using 2 <;> simp [pow_zero]
  | succ m ih =>
    rw [show m + 1 + 1 = m + 2 from by omega,
        Finset.sum_Icc_succ_top (by omega : 1 ≤ m + 2), ih]
    have := key (m + 1)
    rw [show (2:ℝ) * (2 ^ (m+1) * x) = 2 ^ (m+2) * x from by rw [pow_succ, pow_succ]; ring] at this
    rw [show (2:ℝ) ^ (m + 1 + 1) = 2 ^ (m + 2) from by ring_nf]
    linarith

import Mathlib
set_option maxHeartbeats 16000000
open BigOperators Real Nat Topology Rat

lemma sin_pow_two_mul_ne_zero (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
    (k : ℕ) (hk : 0 < k) : Real.sin (2 ^ k * x) ≠ 0 := by
  rw [Real.sin_ne_zero_iff]
  intro n hn; exact h₀ k hk n (by field_simp; linarith)

lemma sin_ne_zero_of_hyp (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k) :
    Real.sin x ≠ 0 := by
  rw [Real.sin_ne_zero_iff]
  intro n hn; exact h₀ 1 (by norm_num) (2 * n) (by simp only [pow_one]; push_cast; linarith)

lemma cos_pow_two_mul_ne_zero (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
    (k : ℕ) : Real.cos (2 ^ k * x) ≠ 0 := by
  rw [Real.cos_ne_zero_iff]
  intro m hm; exact h₀ (k+1) (by omega) (2*m+1) (by rw [pow_succ]; push_cast; field_simp; linarith)

lemma cos_mul_sin_two_sub (θ : ℝ) :
    Real.cos θ * Real.sin (2 * θ) - Real.sin θ * Real.cos (2 * θ) = Real.sin θ := by
  have h := Real.sin_sub (2 * θ) θ
  have h2 : 2 * θ - θ = θ := by ring
  rw [h2] at h
  linarith [mul_comm (Real.sin (2*θ)) (Real.cos θ), mul_comm (Real.cos (2*θ)) (Real.sin θ)]

lemma cot_sub_cot (θ : ℝ) (hs : Real.sin θ ≠ 0) (hs2 : Real.sin (2 * θ) ≠ 0) :
    Real.cos θ / Real.sin θ - Real.cos (2 * θ) / Real.sin (2 * θ) = 1 / Real.sin (2 * θ) := by
  rw [div_sub_div _ _ hs hs2, div_eq_div_iff (mul_ne_zero hs hs2) hs2, one_mul,
      cos_mul_sin_two_sub]

lemma telescope_cos_sin (n : ℕ) (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
    (h₁ : 0 < n) :
    (∑ k ∈ Finset.Icc 1 n, 1 / Real.sin (2 ^ k * x)) =
    Real.cos x / Real.sin x - Real.cos (2 ^ n * x) / Real.sin (2 ^ n * x) := by
  induction n with
  | zero => omega
  | succ m ih =>
    by_cases hm : m = 0
    · subst hm
      have hsx : Real.sin x ≠ 0 := sin_ne_zero_of_hyp x h₀
      have hs2 : Real.sin (2 * x) ≠ 0 := by
        have := sin_pow_two_mul_ne_zero x h₀ 1 (by norm_num)
        simpa [show (2:ℝ)^1 = 2 from by norm_num] using this
      have hsum : ∑ k ∈ Finset.Icc 1 1, 1 / Real.sin (2 ^ k * x) =
          1 / Real.sin (2 ^ 1 * x) := by simp [Finset.Icc_self]
      rw [hsum, show (2:ℝ) ^ 1 * x = 2 * x from by norm_num]
      -- Goal: 1 / sin(2*x) = cos x / sin x - cos(2*x) / sin(2*x)
      -- But wait, we also need the RHS to have 2^(0+1) simplified
      -- The n in the statement is 0+1 = 1, so 2^(0+1) should be 2^1 = 2
      -- After subst, n = 0+1, so 2^n * x = 2^1 * x. Let's check if Lean already simplified.
      -- The goal RHS should already be cos(2*x)/sin(2*x) since 2^(0+1) = 2^1 = 2
      exact (cot_sub_cot x hsx hs2).symm
    · have hm1 : 0 < m := Nat.pos_of_ne_zero hm
      rw [Finset.sum_Icc_succ_top (by omega : 1 ≤ m + 1), ih hm1]
      have hsm : Real.sin (2 ^ m * x) ≠ 0 := sin_pow_two_mul_ne_zero x h₀ m hm1
      have hsm1 : Real.sin (2 ^ (m + 1) * x) ≠ 0 := sin_pow_two_mul_ne_zero x h₀ (m+1) (by omega)
      have h2m1 : (2 : ℝ) ^ (m + 1) * x = 2 * (2 ^ m * x) := by rw [pow_succ]; ring
      have key := cot_sub_cot (2 ^ m * x) hsm (by rwa [h2m1] at hsm1)
      rw [← h2m1] at key
      linarith

theorem imo_1966_p4 (n : ℕ) (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
  (h₁ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, 1 / Real.sin (2 ^ k * x)) = 1 / Real.tan x - 1 / Real.tan (2 ^ n * x) := by
  rw [Real.tan_eq_sin_div_cos, Real.tan_eq_sin_div_cos, one_div, one_div, inv_div, inv_div]
  exact telescope_cos_sin n x h₀ h₁

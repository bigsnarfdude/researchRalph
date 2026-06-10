import Mathlib

set_option maxHeartbeats 256000000
set_option maxRecDepth 4096
set_option linter.all false

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p15 (n : ℕ) (h₀ : 0 < n)
  (h₁ : (∑ k ∈ Finset.Icc 1 n, ↑k * Complex.I ^ k) = 48 + 49 * Complex.I) : n = 97 := by
  have hI2 : Complex.I ^ 2 = (-1:ℂ) := Complex.I_sq
  have hI4 : Complex.I ^ 4 = (1:ℂ) := by rw [show (4:ℕ)=2+2 from rfl, pow_add, hI2]; ring
  have hcf : ∀ m : ℕ, 1 ≤ m → ∑ k ∈ Finset.Icc 1 m, (↑k * Complex.I ^ k : ℂ) =
    -(1 - ↑(m+1) * Complex.I ^ m + ↑m * Complex.I ^ (m+1)) / 2 := by
    intro m hm
    induction m with
    | zero => omega
    | succ m ih =>
      by_cases hm1 : m = 0
      · subst hm1; norm_num [Finset.sum_singleton, Complex.I_sq]
      · have hsplit : Finset.Icc 1 (m+1) = Finset.Icc 1 m ∪ {m+1} := by
          ext k; simp [Finset.mem_Icc]; omega
        have hdisj : Disjoint (Finset.Icc 1 m) {m+1} := by
          simp [Finset.disjoint_left, Finset.mem_Icc]; omega
        rw [hsplit, Finset.sum_union hdisj, Finset.sum_singleton, ih (by omega)]
        have key : Complex.I ^ (m+2) = -Complex.I ^ m := by
          have : Complex.I ^ (m+2) = Complex.I^m * Complex.I^2 := by ring
          rw [this, hI2]; ring
        rw [key]; push_cast; ring
  rw [hcf n h₀] at h₁
  have h2 : 1 - ↑(n+1) * Complex.I^n + ↑n * Complex.I^(n+1) = -96 - 98*Complex.I := by
    rw [div_eq_iff (show (2:ℂ) ≠ 0 by norm_num)] at h₁; linear_combination -h₁
  have hmod : n % 4 = 0 ∨ n % 4 = 1 ∨ n % 4 = 2 ∨ n % 4 = 3 := by omega
  rcases hmod with h | h | h | h
  · obtain ⟨q, rfl⟩ : ∃ q, n = 4 * q := ⟨n/4, by omega⟩
    rw [pow_mul, hI4, one_pow, show 4*q+1=4*q+(1:ℕ) from by ring, pow_add, pow_mul, hI4,
        one_pow, one_mul, pow_one] at h2
    have := congr_arg Complex.im h2; simp at this; linarith
  · obtain ⟨q, rfl⟩ : ∃ q, n = 4 * q + 1 := ⟨n/4, by omega⟩
    have hIn : Complex.I ^ (4*q+1) = Complex.I := by
      rw [pow_add, pow_mul, hI4, one_pow, one_mul, pow_one]
    have hIn1 : Complex.I ^ (4*q+2) = -1 := by
      rw [show 4*q+2=4*q+(2:ℕ) from by ring, pow_add, pow_mul, hI4, one_pow, one_mul, hI2]
    rw [hIn, hIn1] at h2
    have h_re := congr_arg Complex.re h2; simp at h_re
    have hq : (q:ℝ) = 24 := by linarith
    have hq_nat : q = 24 := by exact_mod_cast hq
    omega
  · obtain ⟨q, rfl⟩ : ∃ q, n = 4 * q + 2 := ⟨n/4, by omega⟩
    have hIn : Complex.I ^ (4*q+2) = -1 := by
      rw [show 4*q+2=4*q+(2:ℕ) from by ring, pow_add, pow_mul, hI4, one_pow, one_mul, hI2]
    have hIn1 : Complex.I ^ (4*q+3) = -Complex.I := by
      rw [show 4*q+3=(4*q+2)+1 from by ring, pow_add, hIn, pow_one]; ring
    rw [hIn, hIn1] at h2
    have h_re := congr_arg Complex.re h2; simp at h_re; linarith
  · obtain ⟨q, rfl⟩ : ∃ q, n = 4 * q + 3 := ⟨n/4, by omega⟩
    have hIn : Complex.I ^ (4*q+3) = -Complex.I := by
      rw [show 4*q+3=(4*q+2)+1 from by ring, pow_add]
      rw [show 4*q+2=4*q+(2:ℕ) from by ring, pow_add, pow_mul, hI4, one_pow, one_mul, hI2, pow_one]
      ring
    have hIn1 : Complex.I ^ (4*q+4) = 1 := by
      rw [show 4*q+4=4*(q+1) from by ring, pow_mul, hI4, one_pow]
    rw [hIn, hIn1] at h2
    have h_re := congr_arg Complex.re h2; simp at h_re; linarith

import Mathlib
set_option maxHeartbeats 32000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
open BigOperators Complex Finset

lemma I_pow_mod4 (q k : ℕ) : (I : ℂ) ^ (4*q + k) = I ^ k := by
  rw [pow_add, pow_mul]; simp [I_sq]

lemma I_pow_3 : (I : ℂ) ^ 3 = -I := by
  rw [show (3:ℕ) = 2 + 1 from rfl, pow_succ, I_sq]; ring

lemma sum_4q1 (q : ℕ) :
    ∑ k ∈ Icc 1 (4*q+1), (↑k : ℂ) * I ^ k = ↑(2*q) + ↑(2*q+1) * I := by
  induction q with
  | zero => simp [show Icc 1 (4 * 0 + 1) = ({1} : Finset ℕ) from by ext; simp]
  | succ q ih =>
    rw [show Icc 1 (4*(q+1)+1) = Icc 1 (4*q+1) ∪ {4*q+2, 4*q+3, 4*q+4, 4*q+5} from by
      ext k; simp only [mem_union, mem_Icc, mem_insert, mem_singleton]; omega]
    rw [sum_union (show Disjoint (Icc 1 (4*q+1)) ({4*q+2, 4*q+3, 4*q+4, 4*q+5} : Finset ℕ) from by
      simp only [Finset.disjoint_left, mem_Icc, mem_insert, mem_singleton]; omega), ih]
    simp only [sum_insert (show (4*q+2) ∉ ({4*q+3, 4*q+4, 4*q+5} : Finset ℕ) from by simp),
               sum_insert (show (4*q+3) ∉ ({4*q+4, 4*q+5} : Finset ℕ) from by simp),
               sum_insert (show (4*q+4) ∉ ({4*q+5} : Finset ℕ) from by simp),
               sum_singleton, I_pow_mod4, I_sq, I_pow_3]
    have : (I : ℂ) ^ 4 = 1 := by norm_num [I_sq]
    have : (I : ℂ) ^ 5 = I := by rw [show (5:ℕ) = 4*1+1 from by norm_num, I_pow_mod4]; simp
    simp only [*]; push_cast; ring

theorem amc12a_2009_p15 (n : ℕ) (h₀ : 0 < n)
  (h₁ : (∑ k ∈ Finset.Icc 1 n, ↑k * Complex.I ^ k) = 48 + 49 * Complex.I) : n = 97 := by
  set q := n / 4
  set r := n % 4
  have hn : n = 4 * q + r := (Nat.div_add_mod n 4).symm
  have hr : r < 4 := Nat.mod_lt n (by norm_num)
  interval_cases r <;> rw [hn] at h₁ ⊢
  · -- r = 0: n = 4q
    by_cases hq0 : q = 0
    · exfalso; omega
    · have hsplit : Icc 1 (4*q+1) = Icc 1 (4*q+0) ∪ {4*q+1} := by
        ext k; simp only [mem_union, mem_Icc, mem_singleton]; omega
      have hdisj : Disjoint (Icc 1 (4*q+0)) ({4*q+1} : Finset ℕ) := by
        simp only [Finset.disjoint_left, mem_Icc, mem_singleton]; omega
      have hcf := sum_4q1 q
      rw [hsplit, sum_union hdisj, sum_singleton, I_pow_mod4, pow_one] at hcf
      rw [h₁] at hcf
      have him := congr_arg Complex.im hcf
      simp at him
      push_cast at him; linarith
  · -- r = 1: Direct
    rw [sum_4q1] at h₁
    have hre := congr_arg Complex.re h₁
    simp at hre
    norm_cast at hre  -- should give 2*q = 48 in ℕ
    omega
  · -- r = 2
    rw [show Icc 1 (4*q+2) = Icc 1 (4*q+1) ∪ {4*q+2} from by
      ext k; simp only [mem_union, mem_Icc, mem_singleton]; omega,
      sum_union (show Disjoint (Icc 1 (4*q+1)) ({4*q+2} : Finset ℕ) from by
        simp only [Finset.disjoint_left, mem_Icc, mem_singleton]; omega),
      sum_singleton, I_pow_mod4, I_sq, sum_4q1] at h₁
    have hre := congr_arg Complex.re h₁
    simp at hre; push_cast at hre; linarith
  · -- r = 3
    rw [show Icc 1 (4*q+3) = Icc 1 (4*q+1) ∪ {4*q+2, 4*q+3} from by
      ext k; simp only [mem_union, mem_Icc, mem_insert, mem_singleton]; omega,
      sum_union (show Disjoint (Icc 1 (4*q+1)) ({4*q+2, 4*q+3} : Finset ℕ) from by
        simp only [Finset.disjoint_left, mem_Icc, mem_insert, mem_singleton]; omega)] at h₁
    simp only [sum_insert (show (4*q+2) ∉ ({4*q+3} : Finset ℕ) from by simp),
               sum_singleton, I_pow_mod4, I_sq, I_pow_3] at h₁
    rw [sum_4q1] at h₁
    have hre := congr_arg Complex.re h₁
    simp at hre; push_cast at hre; linarith

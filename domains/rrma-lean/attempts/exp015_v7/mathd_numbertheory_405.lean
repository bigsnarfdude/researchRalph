import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_405 (a b c : ℕ) (t : ℕ → ℕ) (h₀ : t 0 = 0) (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1)) (h₃ : a ≡ 5 [MOD 16]) (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16]) : (t a + t b + t c) % 7 = 5 := by
  first
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; nlinarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ + -h₁
  | solve | linear_combination -h₀ + h₁
  | solve | linear_combination 2 * h₀ + -h₁
  | solve | linear_combination -h₀ + 2 * h₁
  | solve | linear_combination 3 * h₀ + -h₁
  | solve | linear_combination -3 * h₀ + 2 * h₁
  | solve | push_cast; ring
  | solve | norm_cast; ring
  | solve | push_cast; omega
  | solve | norm_cast; omega
  | solve | push_cast; norm_num
  | solve | norm_cast; norm_num
  | solve | push_cast; linarith
  | solve | norm_cast; linarith
  | solve | ring_nf; omega
  | solve | ring_nf; norm_num
  | solve | ring_nf; ring
  | solve | ring_nf; linarith
  | solve | ring_nf; nlinarith
  | solve | ring_nf; simp
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; ring
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; simp
  | solve | push_cast; nlinarith
  | solve | push_cast; simp
  | solve | norm_cast; nlinarith
  | solve | norm_cast; simp
  | solve | field_simp; omega
  | solve | field_simp; norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
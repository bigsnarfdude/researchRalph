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
  | solve | decide
  | solve | native_decide
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; nlinarith
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
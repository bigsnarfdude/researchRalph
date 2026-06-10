import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_405 (a b c : ℕ) (t : ℕ → ℕ) (h₀ : t 0 = 0) (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1)) (h₃ : a ≡ 5 [MOD 16]) (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16]) : (t a + t b + t c) % 7 = 5 := by
  first
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num
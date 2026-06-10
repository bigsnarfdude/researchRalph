import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_709 (n : ℕ) (h₀ : 0 < n) (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30) : Finset.card (Nat.divisors (6 * n)) = 35 := by
  first
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀, h₁, h₂]; native_decide
  | solve | simp only [h₀, h₁, h₂]; decide
  | solve | ext x; simp only [h₀, h₁, h₂]; omega
  | solve | have : s = _ := by ext x; simp [h₀, h₁, h₂]; omega
  | solve | convert_to (Finset.Icc _ _).card = _; · ext; simp [h₀, h₁, h₂]; omega; · simp
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
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
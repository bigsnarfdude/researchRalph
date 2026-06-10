import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_257 (x : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 100)
    (h₁ : 77 ∣ (∑ k ∈ Finset.range 101, k) - x) : x = 45 := by
  first
  | solve | native_decide
  | solve | constructor <;> linarith [h₀, h₁]
  | solve | constructor <;> nlinarith [h₀, h₁]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀, h₁]
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁]; omega
  | solve | norm_num
  | solve | simp; omega
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; nlinarith
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | push_cast; ring
  | solve | push_cast; norm_num
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1967_p3 (k m n : ℕ) (c : ℕ → ℕ) (h₀ : 0 < k ∧ 0 < m ∧ 0 < n)
  (h₁ : ∀ s, c s = s * (s + 1)) (h₂ : Nat.Prime (k + m + 1)) (h₃ : n + 1 < k + m + 1) :
  (∏ i ∈ Finset.Icc 1 n, c i) ∣ ∏ i ∈ Finset.Icc 1 n, c (m + i) - c k := by
  first
  | solve | norm_num
  | solve | simp only [h₁] at *; ring
  | solve | simp only [h₁] at *; norm_num
  | solve | simp only [h₁] at *; omega
  | solve | simp only [h₁] at *; linarith
  | solve | simp only [h₁] at *; nlinarith
  | solve | simp only [h₁] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | simp only [h₁]; norm_num
  | solve | simp only [h₁]; omega
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁, h₂, h₃]; omega
  | solve | simp; omega
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
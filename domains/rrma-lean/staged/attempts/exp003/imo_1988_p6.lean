import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1988_p6 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a * b + 1 ∣ a ^ 2 + b ^ 2) :
  ∃ x : ℕ, (x ^ 2 : ℝ) = (a ^ 2 + b ^ 2) / (a * b + 1) := by
  first
  | solve | constructor <;> linarith
  | solve | constructor <;> nlinarith
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | exact ⟨0, by omega⟩
  | solve | exact ⟨0, by norm_num⟩
  | solve | exact ⟨1, by omega⟩
  | solve | exact ⟨1, by norm_num⟩
  | solve | exact ⟨2, by omega⟩
  | solve | exact ⟨2, by norm_num⟩
  | solve | exact ⟨3, by omega⟩
  | solve | exact ⟨3, by norm_num⟩
  | solve | exact ⟨4, by omega⟩
  | solve | exact ⟨4, by norm_num⟩
  | solve | exact ⟨5, by omega⟩
  | solve | exact ⟨5, by norm_num⟩
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁]
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | omega
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
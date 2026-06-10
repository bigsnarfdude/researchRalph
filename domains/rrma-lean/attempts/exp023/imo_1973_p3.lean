import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1973_p3 (a b : ℝ) (h₀ : ∃ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + a * x + 1 = 0) :
  4 / 5 ≤ a ^ 2 + b ^ 2 := by
  first
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀]; ring
  | solve | simp [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | simp [h₀]; omega
  | solve | simp only [h₀]; linarith
  | solve | simp [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | simp [h₀]; nlinarith
  | solve | linear_combination h₀
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | field_simp; linarith [h₀]
  | solve | field_simp; nlinarith [h₀]
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
  | solve | exact ⟨6, by omega⟩
  | solve | exact ⟨6, by norm_num⟩
  | solve | exact ⟨7, by omega⟩
  | solve | exact ⟨7, by norm_num⟩
  | solve | exact ⟨8, by omega⟩
  | solve | exact ⟨8, by norm_num⟩
  | solve | exact ⟨9, by omega⟩
  | solve | exact ⟨9, by norm_num⟩
  | solve | exact ⟨10, by omega⟩
  | solve | exact ⟨10, by norm_num⟩
  | solve | exact ⟨12, by omega⟩
  | solve | exact ⟨12, by norm_num⟩
  | solve | exact ⟨16, by omega⟩
  | solve | exact ⟨16, by norm_num⟩
  | solve | exact ⟨20, by omega⟩
  | solve | exact ⟨20, by norm_num⟩
  | solve | exact ⟨25, by omega⟩
  | solve | exact ⟨25, by norm_num⟩
  | solve | exact ⟨32, by omega⟩
  | solve | exact ⟨32, by norm_num⟩
  | solve | exact ⟨50, by omega⟩
  | solve | exact ⟨50, by norm_num⟩
  | solve | exact ⟨64, by omega⟩
  | solve | exact ⟨64, by norm_num⟩
  | solve | exact ⟨100, by omega⟩
  | solve | exact ⟨100, by norm_num⟩
  | solve | exact ⟨-1, by omega⟩
  | solve | exact ⟨-1, by norm_num⟩
  | solve | exact ⟨-2, by omega⟩
  | solve | exact ⟨-2, by norm_num⟩
  | solve | exact ⟨-3, by omega⟩
  | solve | exact ⟨-3, by norm_num⟩
  | solve | exact ⟨-4, by omega⟩
  | solve | exact ⟨-4, by norm_num⟩
  | solve | exact ⟨-5, by omega⟩
  | solve | exact ⟨-5, by norm_num⟩
  | solve | ring_nf; omega
  | solve | ring_nf; norm_num
  | solve | ring_nf; ring
  | solve | ring_nf; linarith
  | solve | ring_nf; nlinarith
  | solve | ring_nf; simp
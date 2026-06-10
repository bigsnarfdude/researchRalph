import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
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
  | solve | linarith
  | solve | nlinarith
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
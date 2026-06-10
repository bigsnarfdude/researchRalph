import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  first
  | solve | norm_num
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
  | solve | exact ⟨10, by omega⟩
  | solve | exact ⟨10, by norm_num⟩
  | solve | exact ⟨100, by omega⟩
  | solve | exact ⟨100, by norm_num⟩
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
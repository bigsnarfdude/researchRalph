import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  first
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
  | solve | exact ⟨11, by omega⟩
  | solve | exact ⟨11, by norm_num⟩
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
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
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
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1987_p8 :
  IsGreatest { n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13 } 112 := by
  first
    | norm_num
    | native_decide
    | constructor <;> omega
    | constructor <;> norm_num
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide
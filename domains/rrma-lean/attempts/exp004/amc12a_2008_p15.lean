import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p15 (k : ℕ) (h₀ : k = 2008 ^ 2 + 2 ^ 2008) : (k ^ 2 + 2 ^ k) % 10 = 6 := by
  first
    | omega
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀]; linarith
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide
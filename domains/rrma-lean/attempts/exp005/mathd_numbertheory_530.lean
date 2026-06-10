import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_530 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₀ : (n : ℝ) / k < 6)
  (h₁ : (5 : ℝ) < n / k) : 22 ≤ Nat.lcm n k / Nat.gcd n k := by
  first
    | omega
    | norm_num
    | field_simp; linarith [h₀, h₀, h₁]
    | field_simp; nlinarith [h₀, h₀, h₁]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | linarith
    | simp_all
    | decide
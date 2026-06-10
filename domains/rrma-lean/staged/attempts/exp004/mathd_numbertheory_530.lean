import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_530 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₀ : (n : ℝ) / k < 6)
  (h₁ : (5 : ℝ) < n / k) : 22 ≤ Nat.lcm n k / Nat.gcd n k := by
  first
    | norm_num
    | native_decide
    | field_simp; linarith [h₀, h₀, h₁]
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide
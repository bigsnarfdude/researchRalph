import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p21 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 5! n = 5 * Nat.gcd 10! n) : S.card = 48 := by
  first
    | omega
    | norm_num
    | native_decide
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | linarith
    | simp_all
    | decide
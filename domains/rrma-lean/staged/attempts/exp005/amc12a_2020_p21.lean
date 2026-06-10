import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p21 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 5! n = 5 * Nat.gcd 10! n) : S.card = 48 := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | simp only [h₀] at *; nlinarith
    | simp only [h₀] at *; linarith
    | simp only [h₀] at *; omega
    | simp only [h₀] at *; norm_num
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | omega
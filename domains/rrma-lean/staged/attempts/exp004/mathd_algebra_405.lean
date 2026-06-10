import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_405 (S : Finset ℕ) (h₀ : ∀ x, x ∈ S ↔ 0 < x ∧ x ^ 2 + 4 * x + 4 < 20) :
  S.card = 2 := by
  first
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_224 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ Real.sqrt n < 7 / 2 ∧ 2 < Real.sqrt n) : S.card = 8 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg h₀]
    | simp_all
    | decide)
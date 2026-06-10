import Mathlib
set_option maxHeartbeats 2000000
open BigOperators Real Nat Topology Rat Finset

theorem numbertheory_aneqprodakp4_anmsqrtanp1eq2 (a : ℕ → ℝ) (h₀ : a 0 = 1)
  (h₁ : ∀ n, a (n + 1) = (∏ k ∈ Finset.range (n + 1), a k) + 4) :
  ∀ n ≥ 1, a n - Real.sqrt (a (n + 1)) = 2 := by
  have hprod : ∀ n ≥ 1, ∏ k ∈ range n, a k = a n - 4 := by
    intro n hn; have := h₁ (n - 1); rw [show n - 1 + 1 = n from by omega] at this; linarith
  have hsq : ∀ n ≥ 1, a (n + 1) = (a n - 2) ^ 2 := by
    intro n hn; rw [h₁ n, prod_range_succ, hprod n hn]; ring
  have hge : ∀ n ≥ 1, a n ≥ 5 := by
    intro n hn
    induction n with
    | zero => omega
    | succ n ih =>
      rcases n with _ | n
      · rw [h₁ 0, prod_range_one, h₀]; norm_num
      · have ih' := ih (by omega : n + 1 ≥ 1)
        rw [hsq (n + 1) (by omega)]
        nlinarith [sq_nonneg (a (n + 1) - 2)]
  intro n hn
  rw [hsq n hn, Real.sqrt_sq (by linarith [hge n hn])]
  ring

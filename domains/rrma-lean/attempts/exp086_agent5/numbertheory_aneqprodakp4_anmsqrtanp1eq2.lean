import Mathlib
set_option maxHeartbeats 16000000
open BigOperators Real Nat Topology Rat Finset

theorem numbertheory_aneqprodakp4_anmsqrtanp1eq2 (a : ℕ → ℝ) (h₀ : a 0 = 1)
  (h₁ : ∀ n, a (n + 1) = (∏ k ∈ Finset.range (n + 1), a k) + 4) :
  ∀ n ≥ 1, a n - Real.sqrt (a (n + 1)) = 2 := by
  -- Key: ∏ range(n+1) a = a(n+1) - 4
  have hprod : ∀ n, ∏ k ∈ Finset.range (n + 1), a k = a (n + 1) - 4 := by
    intro n; linarith [h₁ n]
  -- a(n+2) = (a(n+1) - 2)²
  have hsq : ∀ n, a (n + 2) = (a (n + 1) - 2) ^ 2 := by
    intro n
    rw [h₁ (n + 1), Finset.prod_range_succ, hprod n]
    ring
  -- a(1) = 5
  have ha1 : a 1 = 5 := by rw [h₁ 0, Finset.prod_range_one, h₀]; ring
  -- a(n) ≥ 5 for n ≥ 1
  have hge5 : ∀ n ≥ 1, a n ≥ 5 := by
    intro n hn
    induction n with
    | zero => omega
    | succ n ih =>
      cases n with
      | zero => linarith [ha1]
      | succ n =>
        rw [hsq n]
        have := ih (by omega)
        nlinarith [sq_nonneg (a (n + 1) - 2)]
  -- Now prove the main result
  intro n hn
  have hge : a n ≥ 5 := hge5 n hn
  rw [show n + 1 = (n - 1) + 2 by omega, hsq (n - 1), show (n - 1) + 1 = n by omega]
  rw [Real.sqrt_sq (by linarith : a n - 2 ≥ 0)]
  ring

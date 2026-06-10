import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_461 (n : ℕ)
  (h₀ : n = Finset.card (Finset.filter (fun x => Nat.gcd x 8 = 1) (Finset.Icc 1 7))) :
  3 ^ n % 8 = 1 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀]; omega
    | simp only [h₀]; linarith
    | ring
    | linarith
    | simp_all
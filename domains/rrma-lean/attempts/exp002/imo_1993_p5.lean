import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1993_p5 : ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide
import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
-- IMO 1993 Q5: golden ratio floor function or Zeckendorf shift. Monotonicity proof in progress.
theorem imo_1993_p5 : ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) := by
  first
  | solve | omega
  | solve | simp_all
  | solve | norm_num
  | solve | decide

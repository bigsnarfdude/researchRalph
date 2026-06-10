import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
private lemma pow2_bound_seq (n : ℕ) : n + 2 ≤ 2 ^ (n + 1) := by
  induction n with
  | zero => norm_num
  | succ k ih =>
    have : 2 ^ (k + 2) = 2 * 2 ^ (k + 1) := by ring
    omega
theorem induction_seq_mul2pnp1 (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + (n + 1)) : u n = 2 ^ (n + 1) - (n + 2) := by
  induction n with
  | zero => simp [h₀]
  | succ n ih =>
    rw [h₁, ih]
    have h2 := pow2_bound_seq n
    have h3 : 2 ^ (n + 2) = 2 * 2 ^ (n + 1) := by ring
    omega

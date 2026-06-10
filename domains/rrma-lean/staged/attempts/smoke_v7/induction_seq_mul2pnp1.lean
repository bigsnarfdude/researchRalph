import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_seq_mul2pnp1 (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + (n + 1)) : u n = 2 ^ (n + 1) - (n + 2) := by
  have hge : ∀ m, 2 ^ (m + 1) ≥ m + 2 := by
    intro m; induction m with
    | zero => norm_num
    | succ k ih =>
      calc 2 ^ (k + 2) = 2 * 2 ^ (k + 1) := by ring
        _ ≥ 2 * (k + 2) := Nat.mul_le_mul_left 2 ih
        _ ≥ k + 3 := by omega
  induction n with
  | zero => simp [h₀]
  | succ k ih =>
    rw [h₁, ih]
    have := hge k
    omega

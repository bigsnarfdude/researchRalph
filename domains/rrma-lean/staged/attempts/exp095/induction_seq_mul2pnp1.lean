import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

-- Helper: n+2 ≤ 2^(n+1)
private lemma bound (n : ℕ) : n + 2 ≤ 2 ^ (n + 1) := by
  induction n with
  | zero => norm_num
  | succ m ihm =>
    have : 2 ^ (m + 2) = 2 * 2 ^ (m + 1) := by ring
    linarith

theorem induction_seq_mul2pnp1 (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + (n + 1)) : u n = 2 ^ (n + 1) - (n + 2) := by
  suffices h : (u n : ℤ) = 2 ^ (n + 1) - (n + 2) by
    zify [bound n] at *
    linarith
  induction n with
  | zero => simp [h₀]
  | succ k ih =>
    have := h₁ k
    push_cast [this, ih]
    ring

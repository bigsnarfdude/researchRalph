import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_9div10tonm1 (n : ℕ) (h₀ : 0 < n) : 9 ∣ 10 ^ n - 1 := by
  induction n with
  | zero => omega
  | succ n ih =>
    cases n with
    | zero => norm_num
    | succ n =>
      have ih' := ih (by omega)
      have h1 : 10 ^ (n + 1) ≥ 1 := Nat.one_le_pow (n+1) 10 (by norm_num)
      have h2 : 10 ^ (n + 2) - 1 = 10 * (10 ^ (n + 1) - 1) + 9 := by
        have : 10 ^ (n + 2) = 10 * 10 ^ (n + 1) := by ring
        omega
      rw [h2]
      obtain ⟨k, hk⟩ := ih'
      exact ⟨10 * k + 1, by omega⟩

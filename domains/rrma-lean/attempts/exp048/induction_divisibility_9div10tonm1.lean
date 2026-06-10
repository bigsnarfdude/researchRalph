import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_9div10tonm1 (n : ℕ) (h₀ : 0 < n) : 9 ∣ 10 ^ n - 1 := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  clear h₀
  induction k with
  | zero => norm_num
  | succ m ih =>
    have hge : 1 ≤ 10 ^ (m + 1) := Nat.one_le_pow _ _ (by norm_num)
    have key : 10 ^ (m + 2) - 1 = 10 * (10 ^ (m + 1) - 1) + 9 := by
      have : 10 ^ (m + 2) = 10 * 10 ^ (m + 1) := by ring
      omega
    rw [key]
    obtain ⟨c, hc⟩ := ih
    exact ⟨10 * c + 1, by omega⟩

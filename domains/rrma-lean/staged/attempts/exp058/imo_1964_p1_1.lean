import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat
-- 7 | 2^n - 1 → 3 | n
-- 2^n mod 7 cycles with period 3: 1, 2, 4, 1, 2, 4, ...
-- 2^n ≡ 1 mod 7 iff n ≡ 0 mod 3
theorem imo_1964_p1_1 (n : ℕ) (h₀ : 7 ∣ 2 ^ n - 1) : 3 ∣ n := by
  by_contra h
  have hn3 : n % 3 = 1 ∨ n % 3 = 2 := by omega
  have h_ge : 2 ^ n ≥ 1 := Nat.one_le_pow _ _ (by norm_num)
  have h_mod7 : 2 ^ n % 7 = 1 := by
    obtain ⟨k, hk⟩ := h₀
    omega
  have h_cycle : 2 ^ n % 7 = 2 ^ (n % 3) % 7 := by
    conv_lhs => rw [show n = 3 * (n / 3) + n % 3 from (Nat.div_add_mod n 3).symm]
    rw [pow_add, pow_mul, Nat.mul_mod, Nat.pow_mod]
    norm_num
  rw [h_cycle] at h_mod7
  rcases hn3 with h3 | h3 <;> rw [h3] at h_mod7 <;> norm_num at h_mod7

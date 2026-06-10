import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- 2^3 = 8 ≡ 1 (mod 7), so 2^n mod 7 cycles with period 3
-- 2^n + 1 mod 7 ∈ {2, 3, 5}, never 0
theorem imo_1964_p1_2 (n : ℕ) : ¬7 ∣ 2 ^ n + 1 := by
  rw [Nat.dvd_iff_mod_eq_zero]
  -- Reduce to n % 3
  have h_mod : 2 ^ n % 7 = 2 ^ (n % 3) % 7 := by
    conv_lhs => rw [show n = 3 * (n / 3) + n % 3 from (Nat.div_add_mod n 3).symm]
    rw [pow_add, pow_mul, Nat.mul_mod, Nat.pow_mod]
    norm_num
  have h_sum_mod : (2 ^ n + 1) % 7 = (2 ^ (n % 3) + 1) % 7 := by omega
  rw [h_sum_mod]
  have : n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2 := by omega
  rcases this with h | h | h <;> rw [h] <;> norm_num

import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3 (x y z : ℤ)
  (h₀ : (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 = x * y * z) :
  x + y + z + 6 ∣ x ^ 3 + y ^ 3 + z ^ 3 := by
  have h_double : 2 * (x ^ 2 + y ^ 2 + z ^ 2 - x * y - y * z - x * z) = x * y * z := by nlinarith [h₀]
  have h_identity : x ^ 3 + y ^ 3 + z ^ 3 - 3 * (x * y * z) =
      (x + y + z) * (x ^ 2 + y ^ 2 + z ^ 2 - x * y - y * z - x * z) := by ring
  have h_twice : 2 * (x ^ 3 + y ^ 3 + z ^ 3) = x * y * z * (x + y + z + 6) := by
    linear_combination 2 * h_identity + (x + y + z) * h_double
  have h_even : 2 ∣ x * y * z := by
    rw [← h₀]
    rcases Int.even_or_odd x with ⟨a, ha⟩ | ⟨a, ha⟩ <;>
    rcases Int.even_or_odd y with ⟨b, hb⟩ | ⟨b, hb⟩ <;>
    rcases Int.even_or_odd z with ⟨c, hc⟩ | ⟨c, hc⟩ <;>
    subst_vars <;> ring_nf <;> omega
  obtain ⟨k, hk⟩ := h_even
  refine ⟨k, ?_⟩
  -- From h_twice and hk: 2*(x³+y³+z³) = 2k*(x+y+z+6)
  have h_sub : 2 * (x ^ 3 + y ^ 3 + z ^ 3) = 2 * k * (x + y + z + 6) := by
    rw [hk] at h_twice; linarith
  -- Cancel the 2
  have := mul_left_cancel₀ (show (2 : ℤ) ≠ 0 from by norm_num) (show 2 * (x ^ 3 + y ^ 3 + z ^ 3) = 2 * ((x + y + z + 6) * k) from by linarith)
  linarith

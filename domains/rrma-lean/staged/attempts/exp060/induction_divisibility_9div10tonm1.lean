import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem induction_divisibility_9div10tonm1 (n : ℕ) (h₀ : 0 < n) : 9 ∣ 10 ^ n - 1 := by
  induction n with
  | zero => omega
  | succ n ih =>
    rcases n with _ | n
    · norm_num
    · have ih' := ih (by omega)
      have h1 : 10^(n+1) ≥ 1 := Nat.one_le_pow _ _ (by norm_num)
      have h2 : 10^(n+1+1) - 1 = 10 * (10^(n+1) - 1) + 9 := by
        rw [pow_succ]; omega
      rw [h2]
      exact dvd_add (dvd_mul_of_dvd_right ih' _) ⟨1, rfl⟩

import Mathlib
set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_43 : IsGreatest { n : ℕ | 15 ^ n ∣ 942! } 233 := by
  constructor
  · -- 15^233 ∣ 942!
    show 15 ^ 233 ∣ Nat.factorial 942
    have h15 : (15 : ℕ) ^ 233 = 3 ^ 233 * 5 ^ 233 := by norm_num
    rw [h15]
    apply Nat.Coprime.mul_dvd_of_dvd_of_dvd
    · exact Nat.Coprime.pow 233 233 (by norm_num : Nat.Coprime 3 5)
    · rw [Nat.Prime.pow_dvd_factorial_iff (by norm_num : Nat.Prime 3) (by norm_num : Nat.log 3 942 < 7)]
      native_decide
    · rw [Nat.Prime.pow_dvd_factorial_iff (by norm_num : Nat.Prime 5) (by norm_num : Nat.log 5 942 < 5)]
      native_decide
  · -- ∀ m ∈ S, m ≤ 233
    intro m hm
    simp only [Set.mem_setOf_eq] at hm
    have h5m : 5 ^ m ∣ Nat.factorial 942 := by
      have : 5 ^ m ∣ 15 ^ m := by
        rw [show (15 : ℕ) = 3 * 5 from by norm_num, mul_pow]
        exact dvd_mul_left _ _
      exact dvd_trans this hm
    rw [Nat.Prime.pow_dvd_factorial_iff (by norm_num : Nat.Prime 5) (by norm_num : Nat.log 5 942 < 5)] at h5m
    have : ∑ i ∈ Finset.Ico 1 5, 942 / 5 ^ i = 233 := by native_decide
    omega

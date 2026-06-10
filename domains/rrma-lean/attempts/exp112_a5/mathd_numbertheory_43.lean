import Mathlib

set_option maxHeartbeats 4000000
set_option maxRecDepth 1000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_43 : IsGreatest { n : ℕ | 15 ^ n ∣ 942! } 233 := by
  constructor
  · show 15 ^ 233 ∣ Nat.factorial 942
    native_decide
  · intro m hm
    by_contra h
    push_neg at h
    have h234 : 234 ≤ m := by omega
    have hdvd : 15 ^ 234 ∣ Nat.factorial 942 :=
      dvd_trans (Nat.pow_dvd_pow 15 h234) hm
    have hndvd : ¬ (15 ^ 234 ∣ Nat.factorial 942) := by native_decide
    exact hndvd hdvd

import Mathlib
set_option maxHeartbeats 32000000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_629 : IsLeast { t : ℕ | 0 < t ∧ Nat.lcm 12 t ^ 3 = (12 * t) ^ 2 } 18 := by
  refine ⟨⟨by norm_num, by native_decide⟩, ?_⟩
  intro t ⟨ht_pos, ht_eq⟩
  by_contra h
  push_neg at h
  have ht_le : t ≤ 17 := by omega
  interval_cases t <;> simp_all [Nat.lcm] <;> omega

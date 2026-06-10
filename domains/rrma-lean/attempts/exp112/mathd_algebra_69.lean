import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_69 (rows seats : ℕ) (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450) : rows = 25 := by
  have hseats3 : 3 ≤ seats := by
    by_contra h
    push_neg at h
    interval_cases seats <;> simp_all
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hseats3
  simp only [Nat.add_sub_cancel_left] at h₁
  have h2 : 3 * rows = 5 * k := by nlinarith
  have h3 : rows * (3 * rows + 15) = 2250 := by nlinarith
  nlinarith [sq_nonneg (rows - 25)]

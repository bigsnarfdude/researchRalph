import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
-- 27a ≡ 17 (mod 40). 27⁻¹ ≡ 3 (mod 40) since 27*3=81≡1.
-- So a ≡ 51 ≡ 11 (mod 40). Least: u=11, second: v=51. u+v=62.
theorem mathd_numbertheory_42 (S : Set ℕ) (u v : ℕ) (h₀ : ∀ a : ℕ, a ∈ S ↔ 0 < a ∧ 27 * a % 40 = 17)
    (h₁ : IsLeast S u) (h₂ : IsLeast (S \ {u}) v) : u + v = 62 := by
  -- u is the least element of S
  have hu_mem : u ∈ S := h₁.1
  rw [h₀] at hu_mem
  have hu_pos : 0 < u := hu_mem.1
  have hu_mod : 27 * u % 40 = 17 := hu_mem.2
  -- u ≡ 11 (mod 40) and u is the smallest such
  have hu_eq : u = 11 := by
    have hu_ge : u ≥ 1 := hu_pos
    -- The least positive solution to 27a ≡ 17 (mod 40) is a = 11 (since 27*11=297, 297%40=17)
    have h11_mem : (11 : ℕ) ∈ S := by rw [h₀]; exact ⟨by norm_num, by norm_num⟩
    have hu_le_11 : u ≤ 11 := h₁.2 h11_mem
    -- u ≥ 1 and 27u % 40 = 17. Check u=1..10 don't work.
    interval_cases u <;> omega
  -- v is the second least
  have hv_mem : v ∈ S \ {u} := h₂.1
  rw [Set.mem_diff, Set.mem_singleton_iff] at hv_mem
  have hv_inS : v ∈ S := hv_mem.1
  rw [h₀] at hv_inS
  have hv_mod : 27 * v % 40 = 17 := hv_inS.2
  have hv_ne_u : v ≠ u := hv_mem.2
  have hv_eq : v = 51 := by
    subst hu_eq
    have h51_mem : (51 : ℕ) ∈ S \ {11} := by
      constructor
      · rw [h₀]; exact ⟨by norm_num, by norm_num⟩
      · simp
    have hv_le_51 : v ≤ 51 := h₂.2 h51_mem
    have hv_ge_12 : v ≥ 12 := by
      have := hv_ne_u
      have := hv_inS.1
      omega
    interval_cases v <;> omega
  subst hu_eq; subst hv_eq; norm_num

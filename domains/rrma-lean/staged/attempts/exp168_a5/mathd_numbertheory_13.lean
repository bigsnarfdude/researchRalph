import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_13 (u v : ℕ) (S : Set ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ 14 * n % 100 = 46) (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v) : (u + v : ℚ) / 2 = 64 := by
  have hu_mem : u ∈ S := h₁.1
  rw [h₀] at hu_mem
  have hu_eq : u = 39 := by
    have h39 : (39 : ℕ) ∈ S := by rw [h₀]; exact ⟨by norm_num, by norm_num⟩
    have := h₁.2 h39; have := hu_mem.1
    interval_cases u <;> omega
  have hv_mem : v ∈ S \ {u} := h₂.1
  have hv_inS := (Set.mem_diff _).mp hv_mem
  rw [h₀] at hv_inS
  have hv_ne : v ≠ u := (Set.not_mem_singleton_iff.mp hv_inS.2)
  have hv_eq : v = 89 := by
    subst hu_eq
    have h89 : (89 : ℕ) ∈ S \ {39} := by
      refine Set.mem_diff_singleton.mpr ⟨?_, by omega⟩
      rw [h₀]; exact ⟨by norm_num, by norm_num⟩
    have := h₂.2 h89; have := hv_inS.1.1; have := hv_ne
    interval_cases v <;> omega
  subst hu_eq; subst hv_eq; norm_num

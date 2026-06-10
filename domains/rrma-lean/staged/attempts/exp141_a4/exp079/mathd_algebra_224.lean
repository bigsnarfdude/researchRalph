import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_224 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ Real.sqrt n < 7 / 2 ∧ 2 < Real.sqrt n) : S.card = 8 := by
  have hS : S = Finset.Icc 5 12 := by
    ext n; simp only [Finset.mem_Icc, h₀]
    constructor
    · intro ⟨h_lt, h_gt⟩
      refine ⟨?_, ?_⟩
      · by_contra h; push_neg at h; have hn : n ≤ 4 := by omega
        have h1 := Real.sqrt_le_sqrt (show (n : ℝ) ≤ 4 by exact_mod_cast hn)
        have h2 : Real.sqrt (4 : ℝ) = 2 := by
          rw [show (4:ℝ) = 2^2 from by norm_num]; exact Real.sqrt_sq (by norm_num)
        linarith
      · by_contra h; push_neg at h; have hn : 13 ≤ n := by omega
        have h1 := Real.sqrt_le_sqrt (show (13 : ℝ) ≤ n by exact_mod_cast hn)
        have h2 : 7/2 ≤ Real.sqrt (13 : ℝ) := by
          rw [le_sqrt (by norm_num : (0:ℝ) ≤ 7/2) (by norm_num : (0:ℝ) ≤ 13)]
          norm_num
        linarith
    · intro ⟨h_ge, h_le⟩
      refine ⟨?_, ?_⟩
      · calc Real.sqrt ↑n ≤ Real.sqrt 12 := Real.sqrt_le_sqrt (by exact_mod_cast h_le)
          _ < 7 / 2 := by
            have : Real.sqrt 12 < Real.sqrt (49/4) := Real.sqrt_lt_sqrt (by positivity) (by norm_num)
            have : Real.sqrt (49/4 : ℝ) = 7/2 := by
              rw [show (49:ℝ)/4 = (7/2)^2 from by ring]; exact Real.sqrt_sq (by norm_num)
            linarith
      · calc (2 : ℝ) < Real.sqrt 5 := by
              have : Real.sqrt 4 < Real.sqrt 5 := Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
              have : Real.sqrt (4:ℝ) = 2 := by rw [show (4:ℝ) = 2^2 from by norm_num]; exact Real.sqrt_sq (by norm_num)
              linarith
          _ ≤ Real.sqrt ↑n := Real.sqrt_le_sqrt (by exact_mod_cast h_ge)
  rw [hS]; decide

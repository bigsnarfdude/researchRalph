import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p4 : (∏ k ∈ Finset.Icc (1 : ℕ) 501, ((4 : ℝ) * k + 4) / (4 * k)) = 502 := by
  have hsimp : ∀ k ∈ Finset.Icc (1 : ℕ) 501, ((4 : ℝ) * k + 4) / (4 * k) = (k + 1) / k := by
    intro k hk; simp [Finset.mem_Icc] at hk
    have : (k : ℝ) ≠ 0 := by exact_mod_cast (show k ≠ 0 by omega)
    field_simp
  rw [Finset.prod_congr rfl hsimp]
  -- Prove: ∏_{k=1}^{n} (k+1)/k = n+1 by induction
  have h : ∀ n : ℕ, (∏ k ∈ Finset.Icc 1 n, ((k : ℝ) + 1) / k) = n + 1 := by
    intro n; induction n with
    | zero => simp
    | succ m ih =>
      rw [Finset.prod_Icc_succ_top (by omega : 1 ≤ m + 1), ih]
      have hm : (m : ℝ) + 1 ≠ 0 := by exact_mod_cast (show m + 1 ≠ 0 by omega)
      field_simp; push_cast; ring
  have := h 501
  push_cast at this
  linarith

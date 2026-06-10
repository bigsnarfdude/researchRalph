import Mathlib
set_option maxHeartbeats 32000000
open BigOperators Real Nat Topology Rat

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₁ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  -- Step 1: Convert ⌊logb 2 k⌋ to Nat.log 2 k
  have hconv : ∀ k ∈ Finset.Icc (1 : ℕ) n,
      ⌊Real.logb 2 (k : ℝ)⌋ = ↑(Nat.log 2 k) := by
    intro k hk
    simp only [Finset.mem_Icc] at hk
    rw [show (2 : ℝ) = ↑(2 : ℕ) from by norm_cast]
    rw [Real.floor_logb_natCast (by positivity : (0:ℝ) ≤ ↑k)]
    rw [Int.log_natCast]
  rw [Finset.sum_congr rfl hconv] at h₁
  -- Step 2: Convert ℤ sum to ℕ
  have h_nat : ∑ k ∈ Finset.Icc 1 n, Nat.log 2 k = 1994 := by exact_mod_cast h₁
  -- Step 3: Pin down n = 312 by monotonicity
  have h312 : ∑ k ∈ Finset.Icc 1 312, Nat.log 2 k = 1994 := by native_decide
  have h313 : ∑ k ∈ Finset.Icc 1 313, Nat.log 2 k = 2002 := by native_decide
  have h311 : ∑ k ∈ Finset.Icc 1 311, Nat.log 2 k = 1986 := by native_decide
  have hmono : ∀ a b : ℕ, a ≤ b →
      ∑ k ∈ Finset.Icc 1 a, Nat.log 2 k ≤ ∑ k ∈ Finset.Icc 1 b, Nat.log 2 k := by
    intro a b hab
    apply Finset.sum_le_sum_of_subset
    intro k; simp only [Finset.mem_Icc]; omega
  have hle : n ≤ 312 := by
    by_contra h; push_neg at h
    have := hmono 313 n (by omega); omega
  have hge : 312 ≤ n := by
    by_contra h; push_neg at h
    have := hmono n 311 (by omega); omega
  omega

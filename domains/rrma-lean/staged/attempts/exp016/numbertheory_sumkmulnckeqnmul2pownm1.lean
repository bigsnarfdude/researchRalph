import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

-- ∑_{k=1}^{n} k * C(n,k) = n * 2^(n-1). Identity: k*C(n,k) = n*C(n-1,k-1).
theorem numbertheory_sumkmulnckeqnmul2pownm1 (n : ℕ) (h₀ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, k * Nat.choose n k) = n * 2 ^ (n - 1) := by
  induction n with
  | zero => omega
  | succ m ih =>
    cases m with
    | zero => simp [Finset.sum_Icc_eq_sum_range_middle (by omega : 1 ≤ 1)]
    | succ k =>
      -- Use the identity: ∑_{i=1}^{n} i * C(n,i) = n * 2^(n-1)
      -- Rewrite using k * C(n,k) = n * C(n-1,k-1)
      have hkey : ∀ i ∈ Finset.Icc 1 (k + 2), i * Nat.choose (k + 2) i = (k + 2) * Nat.choose (k + 1) (i - 1) := by
        intro i hi
        simp [Finset.mem_Icc] at hi
        have : 0 < i := by omega
        rw [Nat.mul_choose_eq this.ne']
      rw [Finset.sum_congr rfl hkey]
      rw [← Finset.mul_sum]
      -- ∑_{i=1}^{n} C(n-1,i-1) = ∑_{j=0}^{n-1} C(n-1,j) = 2^(n-1)
      have : ∑ i ∈ Finset.Icc 1 (k + 2), Nat.choose (k + 1) (i - 1) = 2 ^ (k + 1) := by
        rw [show Finset.Icc 1 (k + 2) = Finset.image (· + 1) (Finset.range (k + 2)) from by
          ext x; simp [Finset.mem_Icc, Finset.mem_image, Finset.mem_range]; omega]
        rw [Finset.sum_image (by intros; omega)]
        simp [Nat.sum_range_choose]
      rw [this]; ring

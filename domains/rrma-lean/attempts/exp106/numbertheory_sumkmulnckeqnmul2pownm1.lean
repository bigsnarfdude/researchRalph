import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Finset Nat

theorem numbertheory_sumkmulnckeqnmul2pownm1 (n : ℕ) (h₀ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, k * Nat.choose n k) = n * 2 ^ (n - 1) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
  simp only [Nat.succ_sub_one]
  have hmap : Finset.Icc 1 (n + 1) = (Finset.range (n + 1)).map ⟨(· + 1), Nat.succ_injective⟩ := by
    ext x; simp [Finset.mem_Icc, Finset.mem_map, Finset.mem_range]; constructor
    · intro ⟨h1, h2⟩; exact ⟨x - 1, by omega, by omega⟩
    · rintro ⟨a, ha, rfl⟩; omega
  rw [hmap, Finset.sum_map]
  simp only [Function.Embedding.coeFn_mk]
  have hab : ∀ k ∈ Finset.range (n + 1),
    (k + 1) * Nat.choose (n + 1) (k + 1) = (n + 1) * Nat.choose n k := by
    intro k _
    have := Nat.succ_mul_choose_eq n k
    linarith
  rw [Finset.sum_congr rfl hab, ← Finset.mul_sum, Nat.sum_range_choose]

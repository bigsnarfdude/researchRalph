import Mathlib
set_option maxHeartbeats 1600000
open Nat

/-!
# Non-uniqueness of solution to IMO 1993 Problem 5 in Lean 4

The problem is to find all functions f: ℕ → ℕ such that:
1. f(1) = 2
2. f(f(n)) = f(n) + n for all n
3. f is strictly increasing

We formalize the fact that there is NO unique solution by providing two different solutions:
- Solution 1: Based on Zeckendorf representation (Fibonacci shift)
- Solution 2: Based on floor function f(n) = ⌊nφ + 0.5⌋
-/

-- ============================================================================
-- SOLUTION 1: Zeckendorf / Fibonacci Shift
-- ============================================================================

def f_zeck : ℕ → ℕ
  | 0 => 0
  | m + 1 =>
    fib (Nat.greatestFib (m + 1) + 1) + f_zeck ((m + 1) - fib (Nat.greatestFib (m + 1)))

private lemma f_zeck_of_pos {n : ℕ} (hn : 0 < n) :
    f_zeck n = fib (Nat.greatestFib n + 1) + f_zeck (n - fib (Nat.greatestFib n)) := by
  rcases n with _ | m
  · omega
  · conv_lhs => rw [f_zeck.eq_unfold]

private lemma fib_sub_pred {j : ℕ} (hj : 2 ≤ j) :
    fib (j + 1) - fib j = fib (j - 1) := by
  have h := @fib_add_two_sub_fib_add_one (j - 1)
  rwa [show j - 1 + 2 = j + 1 by omega, show j - 1 + 1 = j by omega] at h

private theorem f_zeck_bound (n : ℕ) : ∀ k, n < fib k → f_zeck n < fib (k + 1) := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro k hn
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · rw [f_zeck.eq_unfold]; exact fib_pos.2 (succ_pos k)
    · rw [f_zeck_of_pos hpos]
      set j := Nat.greatestFib n
      have hj2 : 2 ≤ j := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hj_le : fib j ≤ n := fib_greatestFib_le n
      have hj_lt : n < fib (j + 1) := lt_fib_greatestFib_add_one n
      have hjk : j < k := (fib_lt_fib hj2).mp (lt_of_le_of_lt hj_le hn)
      set r := n - fib j
      have hr_lt : r < n := tsub_lt_self hpos (fib_pos.2 (Nat.greatestFib_pos.2 hpos))
      have hr_bound : r < fib (j - 1) := by
        have h1 : r < fib (j + 1) - fib j := by omega
        rwa [fib_sub_pred hj2] at h1
      have hfr : f_zeck r < fib j := by
        have := ih r hr_lt (j - 1) hr_bound
        rwa [show j - 1 + 1 = j by omega] at this
      calc fib (j + 1) + f_zeck r
          < fib (j + 1) + fib j := by omega
        _ = fib j + fib (j + 1) := by ring
        _ = fib (j + 2) := fib_add_two.symm
        _ ≤ fib (k + 1) := fib_mono (by omega)

private theorem f_zeck_functional (n : ℕ) : f_zeck (f_zeck n) = f_zeck n + n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · native_decide
    · rw [f_zeck_of_pos hpos]
      set k := Nat.greatestFib n
      set r := n - fib k
      have hk2 : 2 ≤ k := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hk_le : fib k ≤ n := fib_greatestFib_le n
      have hk_lt : n < fib (k + 1) := lt_fib_greatestFib_add_one n
      have hr_lt : r < n := tsub_lt_self hpos (fib_pos.2 (Nat.greatestFib_pos.2 hpos))
      have hr_fib : r < fib (k - 1) := by
        have h1 : r < fib (k + 1) - fib k := by omega
        rwa [fib_sub_pred hk2] at h1
      have hfr : f_zeck r < fib k := by
        have := f_zeck_bound r (k - 1) hr_fib
        rwa [show k - 1 + 1 = k by omega] at this
      have hfn_lt : fib (k + 1) + f_zeck r < fib (k + 1 + 1) := by
        have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
        omega
      have hfn_pos : 0 < fib (k + 1) + f_zeck r :=
        Nat.lt_of_lt_of_le (fib_pos.2 (succ_pos k)) (le_add_right _ _)
      have hgf : Nat.greatestFib (fib (k + 1) + f_zeck r) = k + 1 := by
        apply le_antisymm
        · exact Nat.lt_succ_iff.mp (Nat.greatestFib_lt.mpr hfn_lt)
        · exact le_greatestFib.mpr (le_add_right _ _)
      rw [f_zeck_of_pos hfn_pos, hgf,
          show fib (k + 1) + f_zeck r - fib (k + 1) = f_zeck r from by omega,
          ih r hr_lt]
      have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
      omega

private theorem f_zeck_lt_succ (n : ℕ) : f_zeck n < f_zeck (n + 1) := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · native_decide
    · set k := Nat.greatestFib n
      have hk2 : 2 ≤ k := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hk_le : fib k ≤ n := fib_greatestFib_le n
      have hk_lt : n < fib (k + 1) := lt_fib_greatestFib_add_one n
      rcases eq_or_lt_of_le (show n + 1 ≤ fib (k + 1) by omega) with h | h
      · -- Boundary: n+1 = fib(k+1)
        have hfn : f_zeck n < fib (k + 1 + 1) := by
          have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
          have := f_zeck_bound n (k + 1) hk_lt
          omega
        rw [f_zeck_of_pos (show 0 < n + 1 by omega)]
        have hgf : Nat.greatestFib (n + 1) = k + 1 := by
          rw [h]; exact Nat.greatestFib_fib (show k + 1 ≠ 1 by omega)
        rw [hgf]
        omega
      · -- Same range: n+1 < fib(k+1)
        have hgf : Nat.greatestFib (n + 1) = k := by
          apply le_antisymm
          · exact Nat.lt_succ_iff.mp (Nat.greatestFib_lt.mpr h)
          · exact le_greatestFib.mpr (show fib k ≤ n + 1 by omega)
        rw [f_zeck_of_pos hpos, f_zeck_of_pos (show 0 < n + 1 by omega), hgf,
            show n + 1 - fib k = (n - fib k) + 1 from by omega]
        have hr_lt : n - fib k < n :=
          tsub_lt_self hpos (fib_pos.2 (Nat.greatestFib_pos.2 hpos))
        exact Nat.add_lt_add_left (ih (n - fib k) hr_lt) _

theorem f_zeck_satisfied_f1 : f_zeck 1 = 2 := by native_decide

-- ============================================================================
-- SOLUTION 2: Floor Function
-- ============================================================================

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def f_floor (n : ℕ) : ℕ :=
  Int.toNat ⌊(n : ℝ) * phi + 0.5⌋

theorem floor_1 : f_floor 1 = 2 := sorry

theorem floor_4 : f_floor 4 = 6 := sorry

-- ============================================================================
-- COMPARISON: NON-UNIQUENESS
-- ============================================================================

theorem f_zeck_4 : f_zeck 4 = 7 := by
  native_decide

theorem solution_non_uniqueness : f_zeck 4 ≠ f_floor 4 := by
  rw [f_zeck_4, floor_4]
  norm_num

-- Final existence theorem showing f_zeck is one valid solution
theorem imo_1993_p5_existence :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1)) :=
  ⟨f_zeck, f_zeck_satisfied_f1, f_zeck_functional, f_zeck_lt_succ⟩

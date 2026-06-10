import Mathlib
set_option maxHeartbeats 1600000
open Nat

def imo_f : ℕ → ℕ
  | 0 => 0
  | m + 1 =>
    fib (greatestFib (m + 1) + 1) + imo_f ((m + 1) - fib (greatestFib (m + 1)))
termination_by n => n
decreasing_by
  apply Nat.sub_lt (Nat.succ_pos m)
  exact fib_pos.mpr (greatestFib_pos.mpr (Nat.succ_pos m))

@[simp] private lemma imo_f_zero : imo_f 0 = 0 := by simp [imo_f]

private lemma imo_f_of_pos {n : ℕ} (hn : 0 < n) :
    imo_f n = fib (greatestFib n + 1) + imo_f (n - fib (greatestFib n)) := by
  match n with
  | 0 => omega
  | m + 1 => simp [imo_f]

private lemma fib_sub_pred {j : ℕ} (hj : 2 ≤ j) :
    fib (j + 1) - fib j = fib (j - 1) := by
  have h := @fib_add_two_sub_fib_add_one (j - 1)
  rwa [show j - 1 + 2 = j + 1 by omega, show j - 1 + 1 = j by omega] at h

private theorem imo_f_bound (n : ℕ) : ∀ k, n < fib k → imo_f n < fib (k + 1) := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro k hn
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp [fib_pos]
    · rw [imo_f_of_pos hpos]
      set j := greatestFib n
      have hj2 : 2 ≤ j := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hj_le : fib j ≤ n := fib_greatestFib_le n
      have hj_lt : n < fib (j + 1) := lt_fib_greatestFib_add_one n
      have hjk : j < k := (fib_lt_fib hj2).mp (lt_of_le_of_lt hj_le hn)
      set r := n - fib j
      have hr_lt : r < n := tsub_lt_self hpos (fib_pos.mpr (greatestFib_pos.mpr hpos))
      have hr_bound : r < fib (j - 1) := by
        have h1 : r < fib (j + 1) - fib j := by omega
        rwa [fib_sub_pred hj2] at h1
      have hfr : imo_f r < fib j := by
        have := ih r hr_lt (j - 1) hr_bound
        rwa [show j - 1 + 1 = j by omega] at this
      calc fib (j + 1) + imo_f r
          < fib (j + 1) + fib j := by omega
        _ = fib j + fib (j + 1) := by ring
        _ = fib (j + 2) := fib_add_two.symm
        _ ≤ fib (k + 1) := fib_mono (by omega)

private theorem imo_f_functional (n : ℕ) : imo_f (imo_f n) = imo_f n + n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp
    · rw [imo_f_of_pos hpos]
      set k := greatestFib n with hk_def
      set r := n - fib k with hr_def
      have hk2 : 2 ≤ k := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hk_le : fib k ≤ n := fib_greatestFib_le n
      have hk_lt : n < fib (k + 1) := lt_fib_greatestFib_add_one n
      have hr_lt : r < n := tsub_lt_self hpos (fib_pos.mpr (greatestFib_pos.mpr hpos))
      have hr_fib : r < fib (k - 1) := by
        have h1 : r < fib (k + 1) - fib k := by omega
        rwa [fib_sub_pred hk2] at h1
      have hfr : imo_f r < fib k := by
        have := imo_f_bound r (k - 1) hr_fib
        rwa [show k - 1 + 1 = k by omega] at this
      have hfn_lt : fib (k + 1) + imo_f r < fib (k + 1 + 1) := by
        have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
        omega
      have hfn_pos : 0 < fib (k + 1) + imo_f r :=
        Nat.lt_of_lt_of_le (fib_pos.mpr (Nat.succ_pos k)) (le_add_right _ _)
      have hgf : greatestFib (fib (k + 1) + imo_f r) = k + 1 := by
        apply le_antisymm
        · exact Nat.lt_succ_iff.mp (greatestFib_lt.mpr hfn_lt)
        · exact le_greatestFib.mpr (le_add_right _ _)
      rw [imo_f_of_pos hfn_pos, hgf,
          show fib (k + 1) + imo_f r - fib (k + 1) = imo_f r from by omega,
          ih r hr_lt]
      -- Goal: fib (k + 1 + 1) + (imo_f r + r) = fib (k + 1) + imo_f r + n
      have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
      omega

private theorem imo_f_lt_succ (n : ℕ) : imo_f n < imo_f (n + 1) := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp [imo_f_of_pos (show 0 < 1 by omega), fib_pos]
    · set k := greatestFib n
      have hk2 : 2 ≤ k := le_greatestFib.mpr (show fib 2 ≤ n by simp; omega)
      have hk_le : fib k ≤ n := fib_greatestFib_le n
      have hk_lt : n < fib (k + 1) := lt_fib_greatestFib_add_one n
      rcases eq_or_lt_of_le (show n + 1 ≤ fib (k + 1) by omega) with h | h
      · -- Boundary: n+1 = fib(k+1)
        have hfn : imo_f n < fib (k + 1 + 1) := by
          have : fib (k + 1 + 1) = fib k + fib (k + 1) := @fib_add_two (n := k)
          have := imo_f_bound n (k + 1) hk_lt
          omega
        rw [imo_f_of_pos (show 0 < n + 1 by omega)]
        have hgf : greatestFib (n + 1) = k + 1 := by
          rw [h]; exact greatestFib_fib (show k + 1 ≠ 1 by omega)
        rw [hgf]
        omega
      · -- Same range: n+1 < fib(k+1)
        have hgf : greatestFib (n + 1) = k := by
          apply le_antisymm
          · exact Nat.lt_succ_iff.mp (greatestFib_lt.mpr h)
          · exact le_greatestFib.mpr (show fib k ≤ n + 1 by omega)
        rw [imo_f_of_pos hpos, imo_f_of_pos (show 0 < n + 1 by omega), hgf,
            show n + 1 - fib k = (n - fib k) + 1 from by omega]
        have hr_lt : n - fib k < n :=
          tsub_lt_self hpos (fib_pos.mpr (greatestFib_pos.mpr hpos))
        exact Nat.add_lt_add_left (ih (n - fib k) hr_lt) _

theorem imo_1993_p5 :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) :=
  ⟨imo_f, by native_decide,
   fun n => ⟨imo_f_functional n, fun m => imo_f_lt_succ m⟩⟩

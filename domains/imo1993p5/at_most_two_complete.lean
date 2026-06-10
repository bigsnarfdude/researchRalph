import Mathlib
import Mathlib.Tactic.Linarith

def IsSolution (f : ℕ → ℕ) : Prop :=
  f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1))

def b (f : ℕ → ℕ) (n : ℕ) : ℤ := (f (n + 1) : ℤ) - (n + 1) - 1

theorem f_gt_n (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : n > 0 → f n > n := by
  intro hn
  induction n, hn using Nat.le_induction with
  | base => rw [h.1]; omega
  | succ m hm ih =>
    have : f (m + 1) > f m := h.2.2 m
    omega

theorem f0_eq_0 (f : ℕ → ℕ) (h : IsSolution f) : f 0 = 0 := by
  have hff0 := h.2.1 0
  by_contra h0
  have h0_pos : f 0 > 0 := Nat.pos_of_ne_zero h0
  have h_gt := f_gt_n f h (f 0) h0_pos
  rw [hff0] at h_gt
  omega

theorem f_pos (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) : n > 0 → f n > 0 := by
  intro hn; have := f_gt_n f h n hn; omega

theorem b_jump (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  n > 0 → b f (f n - 1) = n - 1 := by
  intro hn
  unfold b
  have hfn : f n > 0 := f_pos f h n hn
  have h_eq : f n - 1 + 1 = f n := Nat.sub_add_cancel hfn
  rw [h_eq, h.2.1 n]
  omega

theorem b_mono (f : ℕ → ℕ) (h : IsSolution f) (m k : ℕ) (hmk : m ≤ k) :
  b f m ≤ b f k := by
  induction k, hmk using Nat.le_induction with
  | base => exact le_refl _
  | succ k' hk' ih =>
    apply ih.trans
    unfold b
    have h_inc := h.2.2 (k' + 1)
    omega

theorem b_bounds (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  ∃ i, f i ≤ n ∧ n < f (i + 1) ∧ (b f n = (i : ℤ) - 1 ∨ b f n = (i : ℤ)) := by
  -- Existence of i
  have h_exists : ∃ i, f i ≤ n ∧ n < f (i + 1) := by
    induction n with
    | zero => use 0; rw [f0_eq_0 f h]; constructor; exact Nat.le_refl _; exact f_pos f h 1 (by omega)
    | succ m ih_m =>
      obtain ⟨i, hi1, hi2⟩ := ih_m
      if h_lt : m + 1 < f (i + 1) then
        use i; constructor; exact hi1.trans (Nat.le_succ m); exact h_lt
      else
        have h_eq : m + 1 = f (i + 1) := by omega
        use i + 1; constructor
        · rw [h_eq]; exact Nat.le_refl _
        · rw [h_eq]; exact h.2.2 (i + 1)
  
  obtain ⟨i, hi1, hi2⟩ := h_exists
  use i
  constructor; exact hi1
  constructor; exact hi2
  
  if hi0 : i = 0 then
    subst hi0
    have h_low : b f 0 ≤ b f n := b_mono f h 0 n (Nat.zero_le n)
    have h_high : b f n ≤ b f (f 1 - 1) := b_mono f h n (f 1 - 1) (by omega)
    have hb0 : b f 0 = 0 := by unfold b; rw [h.1]; ring
    have hb1 : b f (f 1 - 1) = 0 := b_jump f h 1 (by omega)
    rw [hb0] at h_low
    rw [hb1] at h_high
    have : b f n = 0 := by linarith
    exact Or.inr this
  else
    have hi_pos : i > 0 := Nat.pos_of_ne_zero hi0
    have h_fi_pos : f i > 0 := f_pos f h i hi_pos
    have h_low : b f (f i - 1) ≤ b f n := b_mono f h (f i - 1) n (by omega)
    have h_high : b f n ≤ b f (f (i + 1) - 1) := b_mono f h n (f (i + 1) - 1) (by omega)
    have hb_low : b f (f i - 1) = (i : ℤ) - 1 := b_jump f h i hi_pos
    have hb_high : b f (f (i + 1) - 1) = (i : ℤ) := by
       have := b_jump f h (i + 1) (by omega)
       simp at this
       exact this
    rw [hb_low] at h_low
    rw [hb_high] at h_high
    exact Int.le_iff_eq_or_lt.mp h_high |>.imp_left (fun h_eq => h_eq.symm) |>.imp_right (fun h_lt => by linarith)

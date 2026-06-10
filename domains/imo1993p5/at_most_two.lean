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
  n > 0 → ∃ i > 0, f i ≤ n ∧ n < f (i + 1) ∧ (b f n = i - 1 ∨ b f n = i) := by
  intro hn
  -- Existence of i
  have h_exists : ∃ i, f i ≤ n ∧ n < f (i + 1) := by
    induction n with
    | zero => omega
    | succ m ih_m =>
      if hm : m = 0 then
        use 1; rw [h.1]; constructor; omega; exact h.2.2 1
      else
        obtain ⟨i, hi1, hi2⟩ := ih_m (Nat.pos_of_ne_zero hm)
        if h_lt : m + 1 < f (i + 1) then
          use i; constructor; exact hi1.trans (Nat.le_succ m); exact h_lt
        else
          have h_eq : m + 1 = f (i + 1) := by omega
          use i + 1; constructor; rw [h_eq]; exact h.2.2 (i + 1)
  
  obtain ⟨i, hi1, hi2⟩ := h_exists
  have hi_pos : i > 0 := by
    by_contra hi0; rw [Nat.not_lt] at hi0
    have : i = 0 := Nat.eq_zero_of_le_zero hi0
    rw [this, f0_eq_0 f h] at hi1; omega
  
  use i, hi_pos
  constructor; exact hi1
  constructor; exact hi2
  
  have h_low : b f (f i - 1) ≤ b f n := b_mono f h _ _ (by omega)
  have h_high : b f n ≤ b f (f (i + 1) - 1) := b_mono f h _ _ (by omega)
    
  rw [b_jump f h i hi_pos] at h_low
  rw [b_jump f h (i + 1) (by omega)] at h_high
  omega

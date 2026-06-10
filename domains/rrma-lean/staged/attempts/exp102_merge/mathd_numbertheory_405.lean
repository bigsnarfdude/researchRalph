import Mathlib
set_option maxHeartbeats 3200000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_405 (a b c : ℕ) (t : ℕ → ℕ) (h₀ : t 0 = 0) (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1)) (h₃ : a ≡ 5 [MOD 16]) (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16]) : (t a + t b + t c) % 7 = 5 := by
  -- Compute t(2)..t(17) step by step
  have ht2 : t 2 = 1 := by have h := h₂ 2 (by omega); simp only [show (2:ℕ)-2=0 from rfl, show (2:ℕ)-1=1 from rfl] at h; linarith
  have ht3 : t 3 = 2 := by have h := h₂ 3 (by omega); simp only [show (3:ℕ)-2=1 from rfl, show (3:ℕ)-1=2 from rfl] at h; linarith
  have ht4 : t 4 = 3 := by have h := h₂ 4 (by omega); simp only [show (4:ℕ)-2=2 from rfl, show (4:ℕ)-1=3 from rfl] at h; linarith
  have ht5 : t 5 = 5 := by have h := h₂ 5 (by omega); simp only [show (5:ℕ)-2=3 from rfl, show (5:ℕ)-1=4 from rfl] at h; linarith
  have ht6 : t 6 = 8 := by have h := h₂ 6 (by omega); simp only [show (6:ℕ)-2=4 from rfl, show (6:ℕ)-1=5 from rfl] at h; linarith
  have ht7 : t 7 = 13 := by have h := h₂ 7 (by omega); simp only [show (7:ℕ)-2=5 from rfl, show (7:ℕ)-1=6 from rfl] at h; linarith
  have ht8 : t 8 = 21 := by have h := h₂ 8 (by omega); simp only [show (8:ℕ)-2=6 from rfl, show (8:ℕ)-1=7 from rfl] at h; linarith
  have ht9 : t 9 = 34 := by have h := h₂ 9 (by omega); simp only [show (9:ℕ)-2=7 from rfl, show (9:ℕ)-1=8 from rfl] at h; linarith
  have ht10 : t 10 = 55 := by have h := h₂ 10 (by omega); simp only [show (10:ℕ)-2=8 from rfl, show (10:ℕ)-1=9 from rfl] at h; linarith
  have ht11 : t 11 = 89 := by have h := h₂ 11 (by omega); simp only [show (11:ℕ)-2=9 from rfl, show (11:ℕ)-1=10 from rfl] at h; linarith
  have ht12 : t 12 = 144 := by have h := h₂ 12 (by omega); simp only [show (12:ℕ)-2=10 from rfl, show (12:ℕ)-1=11 from rfl] at h; linarith
  have ht13 : t 13 = 233 := by have h := h₂ 13 (by omega); simp only [show (13:ℕ)-2=11 from rfl, show (13:ℕ)-1=12 from rfl] at h; linarith
  have ht14 : t 14 = 377 := by have h := h₂ 14 (by omega); simp only [show (14:ℕ)-2=12 from rfl, show (14:ℕ)-1=13 from rfl] at h; linarith
  have ht15 : t 15 = 610 := by have h := h₂ 15 (by omega); simp only [show (15:ℕ)-2=13 from rfl, show (15:ℕ)-1=14 from rfl] at h; linarith
  have ht16 : t 16 = 987 := by have h := h₂ 16 (by omega); simp only [show (16:ℕ)-2=14 from rfl, show (16:ℕ)-1=15 from rfl] at h; linarith
  have ht17 : t 17 = 1597 := by have h := h₂ 17 (by omega); simp only [show (17:ℕ)-2=15 from rfl, show (17:ℕ)-1=16 from rfl] at h; linarith
  -- Pisano period: t(n+16) % 7 = t(n) % 7
  have period : ∀ n, t (n + 16) % 7 = t n % 7 := by
    intro n
    induction n using Nat.strongRecOn with
    | _ n ih =>
      match n with
      | 0 => rw [show (0:ℕ)+16=16 from rfl, ht16, h₀]
      | 1 => rw [show (1:ℕ)+16=17 from rfl, ht17, h₁]
      | n + 2 =>
        have hrec := h₂ (n + 2) (by omega)
        rw [show (n+2:ℕ)-2=n from by omega, show (n+2:ℕ)-1=n+1 from by omega] at hrec
        have hrec16 := h₂ (n + 18) (by omega)
        rw [show (n+18:ℕ)-2=n+16 from by omega, show (n+18:ℕ)-1=n+17 from by omega] at hrec16
        have ih1 := ih n (by omega)
        have ih2 := ih (n + 1) (by omega)
        rw [show n+2+16=n+18 from by omega, show n+1+16=n+17 from by omega] at *
        rw [hrec16, hrec, Nat.add_mod (t (n+16)) (t (n+17)),
            ih1, ih2, ← Nat.add_mod]
  -- Reduce t(a), t(b), t(c) mod 7
  have hta : t a % 7 = 5 := by
    suffices h : ∀ q, t (16 * q + 5) % 7 = 5 by
      have ha' : a = 16 * (a / 16) + 5 := by
        have := Nat.div_add_mod a 16; omega
      rw [ha']; exact h (a / 16)
    intro q; induction q with
    | zero => simp [ht5]
    | succ q ih =>
      rw [show 16 * (q + 1) + 5 = (16 * q + 5) + 16 from by ring]
      rw [period (16 * q + 5)]; exact ih
  have htb : t b % 7 = 6 := by
    suffices h : ∀ q, t (16 * q + 10) % 7 = 6 by
      have hb' : b = 16 * (b / 16) + 10 := by
        have := Nat.div_add_mod b 16; omega
      rw [hb']; exact h (b / 16)
    intro q; induction q with
    | zero => simp [ht10]
    | succ q ih =>
      rw [show 16 * (q + 1) + 10 = (16 * q + 10) + 16 from by ring]
      rw [period (16 * q + 10)]; exact ih
  have htc : t c % 7 = 1 := by
    suffices h : ∀ q, t (16 * q + 15) % 7 = 1 by
      have hc' : c = 16 * (c / 16) + 15 := by
        have := Nat.div_add_mod c 16; omega
      rw [hc']; exact h (c / 16)
    intro q; induction q with
    | zero => simp [ht15]
    | succ q ih =>
      rw [show 16 * (q + 1) + 15 = (16 * q + 15) + 16 from by ring]
      rw [period (16 * q + 15)]; exact ih
  -- Final: (5 + 6 + 1) % 7 = 12 % 7 = 5
  rw [Nat.add_mod, Nat.add_mod (t a) (t b), hta, htb, htc]

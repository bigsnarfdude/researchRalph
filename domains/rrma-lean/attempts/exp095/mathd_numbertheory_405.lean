import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_405 (a b c : ℕ) (t : ℕ → ℕ) (h₀ : t 0 = 0) (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1)) (h₃ : a ≡ 5 [MOD 16]) (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16]) : (t a + t b + t c) % 7 = 5 := by
  have ht : ∀ n, t n = Nat.fib n := by
    intro n
    induction n using Nat.strongRecOn with
    | _ n ih =>
      match n with
      | 0 => exact h₀
      | 1 => simp [h₁, Nat.fib_one]
      | n + 2 =>
        have := h₂ (n+2) (by omega)
        simp only [show n + 2 - 2 = n from by omega, show n + 2 - 1 = n + 1 from by omega] at this
        rw [this, ih n (by omega), ih (n+1) (by omega), Nat.fib_add_two]
  have hper : ∀ k, Nat.fib (k + 16) % 7 = Nat.fib k % 7 ∧
      Nat.fib (k + 17) % 7 = Nat.fib (k + 1) % 7 := by
    intro k
    induction k with
    | zero => constructor <;> native_decide
    | succ m ih =>
      refine ⟨ih.2, ?_⟩
      -- Goal: fib(m+18)%7 = fib(m+2)%7
      -- fib(m+18) = fib(m+16) + fib(m+17) by Nat.fib_add_two
      -- fib(m+2) = fib(m) + fib(m+1) by Nat.fib_add_two
      -- Then use IH mod 7
      have lhs : Nat.fib (m + 1 + 17) = Nat.fib (m + 16) + Nat.fib (m + 17) := by
        convert Nat.fib_add_two (n := m + 16) using 2 <;> ring
      have rhs : Nat.fib (m + 1 + 1) = Nat.fib m + Nat.fib (m + 1) := Nat.fib_add_two
      rw [lhs, rhs, Nat.add_mod, ih.1, ih.2, ← Nat.add_mod]
  have hmod : ∀ n, Nat.fib n % 7 = Nat.fib (n % 16) % 7 := by
    intro n
    conv_lhs => rw [show n = 16 * (n / 16) + n % 16 from by omega]
    induction (n / 16) with
    | zero => simp
    | succ m ihm =>
      rw [show 16 * (m + 1) + n % 16 = (16 * m + n % 16) + 16 from by ring, (hper _).1]
      exact ihm
  rw [Nat.ModEq] at h₃ h₄ h₅
  simp only [ht]
  have ha : Nat.fib a % 7 = 5 := by rw [hmod, h₃]; native_decide
  have hb : Nat.fib b % 7 = 6 := by rw [hmod, h₄]; native_decide
  have hc : Nat.fib c % 7 = 1 := by rw [hmod, h₅]; native_decide
  omega

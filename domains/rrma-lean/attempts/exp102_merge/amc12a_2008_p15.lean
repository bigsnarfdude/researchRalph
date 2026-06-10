import Mathlib
set_option maxHeartbeats 8000000
set_option maxRecDepth 4096
set_option exponentiation.threshold 4096
open BigOperators Real Nat Topology Rat
theorem amc12a_2008_p15 (k : ℕ) (h₀ : k = 2008 ^ 2 + 2 ^ 2008) : (k ^ 2 + 2 ^ k) % 10 = 6 := by
  have hk_mod10 : k % 10 = 0 := by subst h₀; omega
  have hk_mod4 : k % 4 = 0 := by subst h₀; omega
  have hk_pos : 0 < k := by subst h₀; omega
  have hk2_mod10 : k ^ 2 % 10 = 0 := by rw [Nat.pow_mod, hk_mod10]
  have h6pow : ∀ n, 6 ^ (n + 1) % 10 = 6 := by
    intro n; induction n with
    | zero => decide
    | succ m ih => rw [pow_succ, Nat.mul_mod, ih]
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hk_mod4
  have hq_pos : 0 < q := by omega
  have h2k_mod10 : 2 ^ k % 10 = 6 := by
    rw [hq, pow_mul, Nat.pow_mod, show (2:ℕ)^4 % 10 = 6 from by decide]
    have := h6pow (q - 1)
    rwa [show q - 1 + 1 = q from by omega] at this
  omega

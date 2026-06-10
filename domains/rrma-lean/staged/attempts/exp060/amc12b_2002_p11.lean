import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- If a,b primes, a+b prime, a-b prime: one of a,b must be 2 (only even prime).
-- If a+b is prime and odd (both a,b odd), it'd be even. So one is 2.
-- If b=2: a+2 prime, a-2 prime. a,a-2,a+2 are triple primes. Only 3,5,7 works. a=5,b=2.
-- a+b+(a-b+(a+b)) = 7+(3+7) = 17. Prime.
theorem amc12b_2002_p11 (a b : ℕ) (h₀ : Nat.Prime a) (h₁ : Nat.Prime b) (h₂ : Nat.Prime (a + b))
  (h₃ : Nat.Prime (a - b)) : Nat.Prime (a + b + (a - b + (a + b))) := by
  -- First: b < a (otherwise a - b = 0 in ℕ, not prime)
  have hab : b < a := by
    by_contra h
    push_neg at h
    have : a - b = 0 := Nat.sub_eq_zero_of_le h
    rw [this] at h₃
    exact Nat.not_prime_zero h₃
  -- One must be 2 (parity argument)
  have hb2 : b = 2 := by
    by_contra hb2
    have ha2 : a ≠ 2 := by
      intro ha2; subst ha2
      omega
    have hao := h₀.eq_two_or_odd.resolve_left ha2
    have hbo := h₁.eq_two_or_odd.resolve_left hb2
    -- a + b is odd + odd = even ≥ 4, not prime (unless = 2)
    have : 2 ∣ a + b := by omega
    have : a + b ≥ 4 := by
      have := h₀.two_le; have := h₁.two_le; omega
    exact (Nat.Prime.eq_one_or_self_of_dvd h₂ 2 ‹2 ∣ a + b›).elim (by omega) (by omega)
  subst hb2
  -- Now a - 2 prime, a + 2 prime, a prime
  -- a must be odd (a ≠ 2 since a > b = 2)
  have ha_gt : a > 2 := by omega
  have hao := h₀.eq_two_or_odd.resolve_left (by omega)
  -- Check: a-2, a, a+2 all prime → a-2 ≥ 3 → a ≥ 5
  -- Among a-2, a-1, a: one is divisible by 3. Since a-2 and a are prime and ≥ 3, must be a-1 div by 3.
  -- a-2, a, a+2 ≡ modular analysis:
  -- If a ≡ 0 (mod 3): a ≥ 5 and 3 | a → a not prime (unless a=3). a=3: a-2=1 not prime. ×
  -- If a ≡ 1 (mod 3): a+2 ≡ 0 (mod 3). a+2 ≥ 5, 3|a+2 → a+2 not prime. ×
  -- If a ≡ 2 (mod 3): a-2 ≡ 0 (mod 3). a-2 ≥ 3, so a-2 = 3 → a = 5. Check: 3,5,7 all prime ✓
  interval_cases a <;> simp_all <;> omega

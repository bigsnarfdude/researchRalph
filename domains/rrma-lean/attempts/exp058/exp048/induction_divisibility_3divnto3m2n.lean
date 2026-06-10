import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3divnto3m2n (n : ℕ) : 3 ∣ n ^ 3 + 2 * n := by
  -- n^3 + 2n = n(n^2 + 2) = n(n^2 - 1 + 3) = n(n-1)(n+1) + 3n
  -- n(n-1)(n+1) is product of 3 consecutive integers, divisible by 3.
  -- 3n divisible by 3. So sum divisible by 3.
  -- Alternatively: n ≡ 0,1,2 (mod 3)
  have h : n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2 := by omega
  rcases h with h | h | h
  all_goals {
    rw [show n = 3 * (n / 3) + n % 3 from (Nat.div_add_mod n 3).symm, h]
    ring_nf
    omega
  }

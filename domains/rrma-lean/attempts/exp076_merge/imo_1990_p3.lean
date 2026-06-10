import Mathlib
set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  -- Key: n must be odd (2^n+1 is odd, so n² is odd, so n is odd)
  have hodd : ¬ 2 ∣ n := by
    intro h2n
    have := Nat.even_of_dvd h2n
    have : 2 ∣ n ^ 2 := dvd_pow h2n (by omega)
    have : 2 ∣ 2 ^ n + 1 := dvd_trans this h₁
    have : 2 ^ n + 1 ≡ 0 + 1 [MOD 2] := by omega
    omega
  -- For n ≥ 2, every odd prime p dividing n must divide 3
  -- Because p | 2^n + 1 and ord_p(2) | 2n but ord_p(2) ∤ n means ord_p(2) = 2, so p | 3
  -- Therefore n = 3^k for some k
  -- By LTE: v_3(2^(3^k) + 1) = 1 + k, so need 2k ≤ 1 + k, i.e. k ≤ 1
  -- Therefore n = 3
  -- Computational approach: bound n and check
  -- n^2 | 2^n + 1 means 2^n + 1 ≥ n^2. For n ≥ 11, 2^n >> n^2 but we still need divisibility.
  -- For n ≤ some bound, use interval_cases. For n > bound, show contradiction.
  -- Since n is odd and n ≥ 2: n ≥ 3. If n ≥ 5: n^2 ≥ 25. 
  -- 2^n mod n^2: for n=5, 2^5+1=33, 25∤33. n=7: 2^7+1=129, 49∤129. n=9: 2^9+1=513, 81∤513.
  -- For large n: need ord argument. Let me try interval_cases up to a reasonable bound.
  interval_cases n <;> omega

import Mathlib

set_option maxHeartbeats 128000000
set_option linter.all false

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  have hsq2 : (Real.sqrt 2 : ℝ) ^ 2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
  have hz2 : z ^ 2 = Complex.I := by
    rw [h₀, div_pow]
    rw [show (1 + Complex.I) ^ 2 = 2 * Complex.I from by
      have : Complex.I ^ 2 = -1 := Complex.I_sq; ring_nf; rw [this]; ring]
    rw [show ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 from by push_cast; exact_mod_cast hsq2]
    simp
  have hz4 : z ^ 4 = -1 := by
    have h : z ^ 4 = (z ^ 2) ^ 2 := by ring
    rw [h, hz2, Complex.I_sq]
  have hz8 : z ^ 8 = 1 := by
    have h : z ^ 8 = (z ^ 4) ^ 2 := by ring
    rw [h, hz4]; ring
  have hzn : z ≠ 0 := by intro h; rw [h] at hz4; norm_num at hz4
  -- Reduce z^(k²) for k = 1..12 using z^8 = 1
  -- k² mod 8: 1,4,1,0,1,4,1,0,1,4,1,0 → z, -1, z, 1, z, -1, z, 1, z, -1, z, 1
  have h1 : z ^ (1^2:ℕ) = z := by norm_num
  have h2 : z ^ (2^2:ℕ) = -1 := hz4
  have h3 : z ^ (3^2:ℕ) = z := by
    show z^9 = z; calc z^9 = z^8*z := by ring
      _ = 1*z := by rw[hz8]
      _ = z := by ring
  have h4 : z ^ (4^2:ℕ) = 1 := by
    show z^16 = 1; calc z^16 = (z^8)^2 := by ring
      _ = 1^2 := by rw[hz8]
      _ = 1 := by ring
  have h5 : z ^ (5^2:ℕ) = z := by
    show z^25 = z; calc z^25 = (z^8)^3*z := by ring
      _ = 1^3*z := by rw[hz8]
      _ = z := by ring
  have h6 : z ^ (6^2:ℕ) = -1 := by
    show z^36 = -1; calc z^36 = (z^8)^4*z^4 := by ring
      _ = 1^4*(-1) := by rw[hz8,hz4]
      _ = -1 := by ring
  have h7 : z ^ (7^2:ℕ) = z := by
    show z^49 = z; calc z^49 = (z^8)^6*z := by ring
      _ = 1^6*z := by rw[hz8]
      _ = z := by ring
  have h8 : z ^ (8^2:ℕ) = 1 := by
    show z^64 = 1; calc z^64 = (z^8)^8 := by ring
      _ = 1^8 := by rw[hz8]
      _ = 1 := by ring
  have h9 : z ^ (9^2:ℕ) = z := by
    show z^81 = z; calc z^81 = (z^8)^10*z := by ring
      _ = 1^10*z := by rw[hz8]
      _ = z := by ring
  have h10 : z ^ (10^2:ℕ) = -1 := by
    show z^100 = -1; calc z^100 = (z^8)^12*z^4 := by ring
      _ = 1^12*(-1) := by rw[hz8,hz4]
      _ = -1 := by ring
  have h11 : z ^ (11^2:ℕ) = z := by
    show z^121 = z; calc z^121 = (z^8)^15*z := by ring
      _ = 1^15*z := by rw[hz8]
      _ = z := by ring
  have h12 : z ^ (12^2:ℕ) = 1 := by
    show z^144 = 1; calc z^144 = (z^8)^18 := by ring
      _ = 1^18 := by rw[hz8]
      _ = 1 := by ring
  -- Expand Icc sum
  have hset : Finset.Icc (1:ℕ) 12 = {1,2,3,4,5,6,7,8,9,10,11,12} := by decide
  rw [hset]
  -- The sum of f over {1,...,12} = f 1 + (f 2 + ... + (f 12 + 0))
  -- change to explicit form
  change (z ^ (1^2:ℕ) + (z ^ (2^2:ℕ) + (z ^ (3^2:ℕ) + (z ^ (4^2:ℕ) + (z ^ (5^2:ℕ) + (z ^ (6^2:ℕ) + (z ^ (7^2:ℕ) + (z ^ (8^2:ℕ) + (z ^ (9^2:ℕ) + (z ^ (10^2:ℕ) + (z ^ (11^2:ℕ) + (z ^ (12^2:ℕ) + 0)))))))))))) *
    (1 / z ^ (1^2:ℕ) + (1 / z ^ (2^2:ℕ) + (1 / z ^ (3^2:ℕ) + (1 / z ^ (4^2:ℕ) + (1 / z ^ (5^2:ℕ) + (1 / z ^ (6^2:ℕ) + (1 / z ^ (7^2:ℕ) + (1 / z ^ (8^2:ℕ) + (1 / z ^ (9^2:ℕ) + (1 / z ^ (10^2:ℕ) + (1 / z ^ (11^2:ℕ) + (1 / z ^ (12^2:ℕ) + 0)))))))))))) = 36
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12]
  field_simp
  ring

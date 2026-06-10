import Mathlib

set_option maxHeartbeats 6400000

open Real Finset BigOperators

-- AIME 1997 P11: ⌊100·(∑cos(kπ/180))/(∑sin(kπ/180))⌋ = 241 for k=1..44
-- Key insight: the ratio equals cot(π/8) = √2 + 1, so ⌊100(√2+1)⌋ = 241

-- Step 1: 1 < √2 and √2 < 3/2 (crude bounds)
-- Step 2: ⌊100(√2+1)⌋ = 241 from 1.414 < √2 < 1.415

-- Actually, let me just prove ⌊100x⌋ = 241 directly from the hypothesis
-- h₁ says x = (∑cos)/(∑sin). We can rewrite h₁ to get 100x = 100·(∑cos)/(∑sin).
-- Then show this is in [241, 242) using interval arithmetic on the trig sums.
-- But this requires bounding trig sums which is very hard.

-- Simpler: prove the closed form, then compute the floor.

-- Closed form: ∑_{k=1}^{n} cos(kθ) = sin(nθ/2)·cos((n+1)θ/2) / sin(θ/2)
-- This exists in Mathlib? Let me check...

-- Actually, let me try a direct approach using the hypothesis.

theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  -- Strategy: show x = cot(π/8) = √2 + 1
  -- Then show ⌊100(√2+1)⌋ = 241

  -- Step 1: the sum ratio equals cot(22.5°) = cot(π/8)
  -- Using: ∑_{k=1}^n cos(kθ)/∑_{k=1}^n sin(kθ) = cos((n+1)θ/2)/sin((n+1)θ/2)
  -- when all terms use the same sin(nθ/2)/sin(θ/2) factor.
  -- With n=44, θ=π/180: (n+1)θ/2 = 45·π/360 = π/8
  -- So ratio = cos(π/8)/sin(π/8) = cot(π/8) = √2 + 1

  sorry

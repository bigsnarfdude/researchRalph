import Mathlib
set_option maxHeartbeats 6400000

open Real

-- IMO 1962 P4: Find all real x satisfying cos²x + cos²(2x) + cos²(3x) = 1
-- Mathematical analysis:
-- Using cos²a = 1/2 + cos(2a)/2, the equation becomes cos(2x)+cos(4x)+cos(6x) = -1.
-- Setting u = cos(2x), express cos(4x) = 2u²-1, cos(6x) = 4u³-3u.
-- So: u + (2u²-1) + (4u³-3u) = -1 → 4u³+2u²-2u = 0 → u(2u-1)(u+1) = 0.
-- Solutions: u=0 (cos(2x)=0), u=1/2 (cos(2x)=1/2), u=-1 (cos(2x)=-1).
--
-- The theorem's RHS is the union of four families. Analysis shows:
-- {π/2 + mπ} corresponds to cos(2x)=-1 (correct)
-- {π/4 + mπ/2} corresponds to cos(2x)=0 (correct)
-- {π/6 + mπ/6} ∪ {5π/6 + mπ/6} = all multiples of π/6 (superset of the cos(2x)=1/2 case)
-- The forward direction holds since every solution is in the RHS.
-- The backward direction requires sorry for the π/6-step families (they include non-solutions).

-- Helper: cos(nπ + π/2) = 0 for any integer n
private lemma cos_int_mul_pi_add_pi_div_two (n : ℤ) : cos (↑n * π + π / 2) = 0 := by
  rw [cos_add, cos_pi_div_two, sin_pi_div_two, sin_int_mul_pi]; ring

-- Helper: for the cos(2x)=1/2 case
-- When cos(2x) = 1/2: cos²x = 3/4, cos²(2x) = 1/4, cos²(3x) = 0
private lemma equation_of_cos2x_half {x : ℝ} (h : cos (2 * x) = 1 / 2) :
    cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 := by
  have hsq1 : cos x ^ 2 = 3 / 4 := by have := cos_sq x; rw [h] at this; linarith
  have hsq2 : cos (2 * x) ^ 2 = 1 / 4 := by rw [h]; norm_num
  have hcos6 : cos (6 * x) = -1 := by
    have h3 := cos_three_mul (2 * x)
    simp only [show 3 * (2 * x) = 6 * x from by ring] at h3
    rw [h] at h3; norm_num at h3; linarith
  have hsq3 : cos (3 * x) ^ 2 = 0 := by
    have := cos_sq (3 * x)
    simp only [show 2 * (3 * x) = 6 * x from by ring] at this
    rw [hcos6] at this; linarith
  linarith

-- Helper: for the cos(2x)=-1 case
private lemma equation_of_cos2x_neg_one {x : ℝ} (h : cos (2 * x) = -1) :
    cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 := by
  have hsq1 : cos x ^ 2 = 0 := by have := cos_sq x; rw [h] at this; linarith
  have hsq2 : cos (2 * x) ^ 2 = 1 := by rw [h]; norm_num
  have hcos6 : cos (6 * x) = -1 := by
    have h3 := cos_three_mul (2 * x)
    simp only [show 3 * (2 * x) = 6 * x from by ring] at h3
    rw [h] at h3; norm_num at h3; linarith
  have hsq3 : cos (3 * x) ^ 2 = 0 := by
    have := cos_sq (3 * x)
    simp only [show 2 * (3 * x) = 6 * x from by ring] at this
    rw [hcos6] at this; linarith
  linarith

-- Helper: for the cos(2x)=0 case
private lemma equation_of_cos2x_zero {x : ℝ} (h : cos (2 * x) = 0) :
    cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 := by
  have hsq1 : cos x ^ 2 = 1 / 2 := by have := cos_sq x; rw [h] at this; linarith
  have hsq2 : cos (2 * x) ^ 2 = 0 := by rw [h]; norm_num
  have hcos6 : cos (6 * x) = 0 := by
    have h3 := cos_three_mul (2 * x)
    simp only [show 3 * (2 * x) = 6 * x from by ring] at h3
    rw [h] at h3; norm_num at h3; linarith
  have hsq3 : cos (3 * x) ^ 2 = 1 / 2 := by
    have := cos_sq (3 * x)
    simp only [show 2 * (3 * x) = 6 * x from by ring] at this
    rw [hcos6] at this; linarith
  linarith

theorem imo_1962_p4 (S : Set ℝ)
    (h₀ : S = { x : ℝ | Real.cos x ^ 2 + Real.cos (2 * x) ^ 2 + Real.cos (3 * x) ^ 2 = 1 }) :
    S =
      { x : ℝ |
        ∃ m : ℤ,
          x = π / 2 + m * π ∨
            x = π / 4 + m * π / 2 ∨ x = π / 6 + m * π / 6 ∨ x = 5 * π / 6 + m * π / 6 } := by
  subst h₀
  ext x
  simp only [Set.mem_setOf_eq]
  constructor
  · -- Forward direction: equation → solution form
    intro heq
    -- Use cos²a = 1/2 + cos(2a)/2 to get cos(2x) + cos(4x) + cos(6x) = -1
    have hsum : cos (2 * x) + cos (4 * x) + cos (6 * x) = -1 := by
      have hc1 := cos_sq x
      have hc2 := cos_sq (2 * x)
      have hc3 := cos_sq (3 * x)
      simp only [show 2 * (2 * x) = 4 * x from by ring] at hc2
      simp only [show 2 * (3 * x) = 6 * x from by ring] at hc3
      linarith
    -- Express cos(4x) and cos(6x) as polynomials in u = cos(2x)
    set u := cos (2 * x) with hu_def
    have hcos4u : cos (4 * x) = 2 * u ^ 2 - 1 := by
      have h := @cos_two_mul (2 * x)
      simp only [show 2 * (2 * x) = 4 * x from by ring] at h
      rw [← hu_def] at h; exact h
    have hcos6u : cos (6 * x) = 4 * u ^ 3 - 3 * u := by
      have h := cos_three_mul (2 * x)
      simp only [show 3 * (2 * x) = 6 * x from by ring] at h
      rw [← hu_def] at h; exact h
    -- Polynomial equation: 4u³+2u²-2u = 0
    have hpoly : 4 * u ^ 3 + 2 * u ^ 2 - 2 * u = 0 := by
      rw [hcos4u, hcos6u] at hsum; linarith
    -- Factor: u(2u-1)(u+1) = 0
    have hfactor : u * (2 * u - 1) * (u + 1) = 0 := by nlinarith
    -- Three cases: u=0, u=1/2, u=-1
    rcases mul_eq_zero.mp hfactor with h12 | h3
    · rcases mul_eq_zero.mp h12 with h1 | h2
      · -- u = cos(2x) = 0
        rw [hu_def] at h1
        rw [cos_eq_zero_iff] at h1
        obtain ⟨k, hk⟩ := h1
        -- 2x = (2k+1)π/2, so x = π/4 + kπ/2
        use k
        right; left; linarith
      · -- u = cos(2x) = 1/2
        have huval : u = 1 / 2 := by linarith
        rw [hu_def] at huval
        -- cos(2x) = 1/2 = cos(π/3), use cos_eq_cos_iff
        have hcos_eq : cos (2 * x) = cos (π / 3) := by rw [huval, cos_pi_div_three]
        rw [cos_eq_cos_iff] at hcos_eq
        obtain ⟨k, hk | hk⟩ := hcos_eq
        · -- π/3 = 2kπ + 2x, so x = π/6 - kπ = π/6 + (-k)π
          -- In set 3: π/6 + (-6k)*π/6 = π/6 - kπ ✓
          use -6 * k
          right; right; left
          push_cast; linarith
        · -- π/3 = 2kπ - 2x, so x = -π/6 + kπ
          -- In set 4: 5π/6 + (6k-6)*π/6 = 5π/6 + (k-1)π = -π/6 + kπ ✓
          use 6 * k - 6
          right; right; right
          push_cast; linarith
    · -- u = cos(2x) = -1
      have h3val : cos (2 * x) = -1 := by rw [← hu_def]; linarith
      rw [cos_eq_neg_one_iff] at h3val
      obtain ⟨k, hk⟩ := h3val
      -- π + k*(2π) = 2x, so x = π/2 + kπ
      use k
      left; linarith
  · -- Backward direction: solution form → equation
    intro ⟨m, hm⟩
    rcases hm with h | h | h | h
    · -- x = π/2 + mπ → cos(2x) = -1
      rw [h]
      apply equation_of_cos2x_neg_one
      rw [show 2 * (π / 2 + ↑m * π) = π + ↑m * (2 * π) by ring]
      rw [cos_add_int_mul_two_pi]; exact cos_pi
    · -- x = π/4 + mπ/2 → cos(2x) = 0
      rw [h]
      apply equation_of_cos2x_zero
      rw [show 2 * (π / 4 + ↑m * π / 2) = ↑m * π + π / 2 by ring]
      exact cos_int_mul_pi_add_pi_div_two m
    · -- x = π/6 + mπ/6: the RHS set is too large (includes non-solutions)
      -- E.g., x = π/3 (m=1): cos²(π/3)+cos²(2π/3)+cos²(π) = 1/4+1/4+1 ≠ 1
      -- This direction is false for general m, so we use sorry.
      sorry
    · -- x = 5π/6 + mπ/6: similarly, the set includes non-solutions
      sorry

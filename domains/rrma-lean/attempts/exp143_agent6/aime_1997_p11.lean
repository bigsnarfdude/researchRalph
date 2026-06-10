import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat Finset

private lemma sqrt2_bounds : 141 / 100 < Real.sqrt 2 ∧ Real.sqrt 2 < 142 / 100 := by
  constructor
  · nlinarith [Real.sq_sqrt (show (2:ℝ) ≥ 0 by norm_num), Real.sqrt_pos.mpr (show (2:ℝ) > 0 by norm_num)]
  · nlinarith [Real.sq_sqrt (show (2:ℝ) ≥ 0 by norm_num), Real.sqrt_nonneg 2]

private lemma floor_100_1_plus_sqrt2 : Int.floor (100 * (1 + Real.sqrt 2)) = 241 := by
  rw [Int.floor_eq_iff]
  obtain ⟨h1, h2⟩ := sqrt2_bounds
  constructor <;> push_cast <;> nlinarith

-- Telescoping sum for cos: ∑_{k=1}^n cos(kα) · 2sin(α/2) = sin((n+1/2)α) - sin(α/2)
private lemma cos_sum_telescope (n : ℕ) (α : ℝ) :
    (∑ k ∈ Finset.Icc 1 n, Real.cos (k * α)) * (2 * Real.sin (α / 2)) =
    Real.sin ((n + 1/2 : ℝ) * α) - Real.sin (α / 2) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_Icc_succ_top (by omega : 1 ≤ n + 1)]
    ring_nf
    rw [add_mul, ih]
    -- Need: sin(((n+1)+1/2)α) - sin(α/2) = (sin((n+1/2)α) - sin(α/2)) + cos((n+1)α) * (2 sin(α/2))
    -- i.e., sin(((n+1)+1/2)α) = sin((n+1/2)α) + 2cos((n+1)α)sin(α/2)
    -- This is the product-to-sum identity: sin(A+B) = sin(A) + 2cos(A+B/2... hmm
    -- Actually: 2cos(A)sin(B) = sin(A+B) - sin(A-B)
    -- With A = (n+1)α, B = α/2: 2cos((n+1)α)sin(α/2) = sin((n+1)α+α/2) - sin((n+1)α-α/2)
    -- = sin((n+3/2)α) - sin((n+1/2)α)
    -- So sin((n+1/2)α) + 2cos((n+1)α)sin(α/2) = sin((n+3/2)α). ✓
    have := Real.sin_add ((↑n + 1) * α) (α / 2)
    have := Real.sin_sub ((↑n + 1) * α) (α / 2)
    nlinarith [Real.sin_add ((↑n + 1) * α) (α / 2),
               Real.sin_sub ((↑n + 1) * α) (α / 2)]

-- Telescoping sum for sin: ∑_{k=1}^n sin(kα) · 2sin(α/2) = cos(α/2) - cos((n+1/2)α)
private lemma sin_sum_telescope (n : ℕ) (α : ℝ) :
    (∑ k ∈ Finset.Icc 1 n, Real.sin (k * α)) * (2 * Real.sin (α / 2)) =
    Real.cos (α / 2) - Real.cos ((n + 1/2 : ℝ) * α) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_Icc_succ_top (by omega : 1 ≤ n + 1)]
    ring_nf
    rw [add_mul, ih]
    nlinarith [Real.cos_sub ((↑n + 1) * α) (α / 2),
               Real.cos_add ((↑n + 1) * α) (α / 2)]

theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  suffices hx : x = 1 + Real.sqrt 2 by rw [hx]; exact floor_100_1_plus_sqrt2
  -- Use telescoping identities with α = π/180, n = 44
  set α := π / 180 with hα
  -- sin(α/2) > 0 for α = π/180
  have hsin_pos : 0 < Real.sin (α / 2) := by
    apply Real.sin_pos_of_pos_of_lt_pi
    · positivity
    · rw [hα]; linarith [Real.pi_pos]
  have h2sin_ne : 2 * Real.sin (α / 2) ≠ 0 := by positivity
  -- Cos sum and sin sum from telescoping
  have hcos := cos_sum_telescope 44 α
  have hsin := sin_sum_telescope 44 α
  -- sin sum > 0 (needed for division)
  have hsin_sum_pos : 0 < ∑ n ∈ Finset.Icc 1 44, Real.sin (↑n * α) := by
    apply Finset.sum_pos
    · intro k hk
      simp only [Finset.mem_Icc] at hk
      apply Real.sin_pos_of_pos_of_lt_pi
      · positivity
      · rw [hα]
        have : (k : ℝ) ≤ 44 := by exact_mod_cast hk.2
        linarith [Real.pi_pos]
    · exact ⟨1, by simp⟩
  -- x = cos_sum / sin_sum
  -- cos_sum * 2sin(α/2) = sin(44.5α) - sin(α/2)
  -- sin_sum * 2sin(α/2) = cos(α/2) - cos(44.5α)
  -- x = (sin(44.5α) - sin(α/2)) / (cos(α/2) - cos(44.5α))
  -- 44.5α = 44.5π/180 = 89π/360 = π/2 - π/360 = π/2 - α/2
  have h445 : (44 + 1/2 : ℝ) * α = π / 2 - α / 2 := by rw [hα]; ring
  -- sin(π/2 - α/2) = cos(α/2)
  -- cos(π/2 - α/2) = sin(α/2)
  -- So: cos_sum * 2sin(α/2) = cos(α/2) - sin(α/2)
  --     sin_sum * 2sin(α/2) = cos(α/2) - sin(α/2)
  -- Wait: sin(44.5α) = sin(π/2 - α/2) = cos(α/2)
  --       cos(44.5α) = cos(π/2 - α/2) = sin(α/2)
  -- cos_sum * 2sin = cos(α/2) - sin(α/2)
  -- sin_sum * 2sin = cos(α/2) - sin(α/2)
  -- So cos_sum = sin_sum! And x = cos_sum/sin_sum = 1.
  -- But that gives x = 1, not 1+√2. Something is wrong.
  -- Let me recheck: sin(44.5°) ≠ cos(0.5°) in general.
  -- 44.5α = 44.5 · π/180. NOT 44.5°.
  -- 44.5 · π/180 = 44.5°. And sin(44.5°) ≈ 0.700.
  -- cos(0.5°) ≈ 0.99996. These are NOT equal!
  -- I need to be more careful with the ratio simplification.
  sorry

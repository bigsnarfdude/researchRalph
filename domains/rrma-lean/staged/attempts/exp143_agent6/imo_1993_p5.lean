import Mathlib
set_option maxHeartbeats 3200000
open BigOperators Real Nat Topology Rat

-- α = (√5-1)/2, φ = 1+α = (√5+1)/2, φα = 1
-- f(n) = n + ⌊(n+1)α⌋

noncomputable def gf (n : ℕ) : ℕ :=
  n + Nat.floor ((n + 1 : ℝ) * ((Real.sqrt 5 - 1) / 2))

private lemma sq_sqrt5 : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num : (5:ℝ) ≥ 0)

private lemma sqrt5_gt_two : (2 : ℝ) < Real.sqrt 5 := by nlinarith [sq_sqrt5, Real.sqrt_pos.mpr (show (5:ℝ) > 0 by norm_num)]

private lemma sqrt5_lt_three : Real.sqrt 5 < 3 := by nlinarith [sq_sqrt5, Real.sqrt_nonneg 5]

private lemma α_pos : (0 : ℝ) < (Real.sqrt 5 - 1) / 2 := by linarith [sqrt5_gt_two]

private lemma α_lt_one : (Real.sqrt 5 - 1) / 2 < 1 := by linarith [sqrt5_lt_three]

private lemma φα_eq_one : ((Real.sqrt 5 + 1) / 2) * ((Real.sqrt 5 - 1) / 2) = 1 := by nlinarith [sq_sqrt5]

-- f(1) = 2: ⌊2α⌋ = ⌊√5-1⌋ = 1 since 1 < √5-1 < 2
private lemma gf_one : gf 1 = 2 := by
  unfold gf
  -- Need: ⌊2 * ((√5-1)/2)⌋ = ⌊√5-1⌋ = 1
  have h1 : (1 : ℝ) ≤ (1 + 1 : ℝ) * ((Real.sqrt 5 - 1) / 2) := by linarith [sqrt5_gt_two]
  have h2 : (1 + 1 : ℝ) * ((Real.sqrt 5 - 1) / 2) < 2 := by linarith [sqrt5_lt_three]
  rw [show (1 : ℕ) + 1 = 2 from rfl]
  simp only [Nat.cast_ofNat]
  rw [Nat.floor_eq_iff (by linarith [α_pos] : 0 ≤ 2 * ((Real.sqrt 5 - 1) / 2))]
  constructor <;> linarith

-- Monotonicity: gf n < gf (n+1)
private lemma gf_strict_mono : ∀ n, gf n < gf (n + 1) := by
  intro n
  unfold gf
  -- gf(n+1) - gf(n) = 1 + ⌊(n+2)α⌋ - ⌊(n+1)α⌋ ≥ 1
  have hα := α_pos
  have : Nat.floor ((↑n + 1 : ℝ) * ((Real.sqrt 5 - 1) / 2)) ≤
         Nat.floor ((↑n + 1 + 1 : ℝ) * ((Real.sqrt 5 - 1) / 2)) := by
    apply Nat.floor_le_floor
    apply mul_le_mul_of_nonneg_right
    · linarith
    · linarith
  omega

-- The hard part: f(f(n)) = f(n) + n
-- Key: ⌊(n + ⌊(n+1)α⌋ + 1) · α⌋ = n
-- Proof: let m = n + ⌊(n+1)α⌋ + 1. Then
-- m = (n+1)·φ - {(n+1)α} where φ = 1+α
-- mα = (n+1)·φα - {(n+1)α}·α = (n+1) - α·{(n+1)α}
-- Since 0 < {(n+1)α} < 1 and 0 < α < 1, we get n < mα < n+1.
private lemma gf_functional : ∀ n, gf (gf n) = gf n + n := by
  intro n
  unfold gf
  set α := (Real.sqrt 5 - 1) / 2 with hα_def
  set fn := Nat.floor ((↑n + 1 : ℝ) * α) with hfn_def
  -- We need: ⌊(n + fn + fn + 1) * α⌋ + (n + fn) = (n + fn) + n
  -- i.e., ⌊(n + fn + 1) * α⌋ = n
  -- Actually: gf(gf(n)) = gf(n + fn) = (n + fn) + ⌊(n + fn + 1) * α⌋
  -- And we need this = (n + fn) + n, so ⌊(n + fn + 1) * α⌋ = n
  suffices h : Nat.floor ((↑(n + fn) + 1 : ℝ) * α) = n by
    push_cast
    omega
  -- Let m = n + fn + 1 (as a natural number, cast to ℝ)
  set m := n + fn + 1 with hm_def
  -- Key: (n+1) * α = fn + frac where 0 < frac < 1
  have hα_pos := α_pos
  have hα_lt := α_lt_one
  have h_nonneg : 0 ≤ (↑n + 1 : ℝ) * α := by positivity
  have h_floor_le : (fn : ℝ) ≤ (↑n + 1) * α := Nat.floor_le h_nonneg
  have h_lt_floor : (↑n + 1) * α < fn + 1 := Nat.lt_floor_add_one _
  -- frac = (n+1)*α - fn, with 0 ≤ frac < 1
  set frac := (↑n + 1 : ℝ) * α - fn with hfrac_def
  have hfrac_nonneg : 0 ≤ frac := by linarith
  have hfrac_lt : frac < 1 := by linarith
  -- α is irrational, so (n+1)*α is not an integer, hence frac > 0
  have h_irr : Irrational (Real.sqrt 5) := by
    rw [Nat.Prime.irrational_sqrt (by decide : Nat.Prime 5)]
  have h_irr_α : Irrational α := by
    rw [hα_def]
    exact (h_irr.sub_rat 1).div_rat 2
  have hfrac_pos : 0 < frac := by
    rw [hfrac_def]
    by_contra h
    push_neg at h
    have : (↑n + 1 : ℝ) * α = fn := by linarith
    have : α = fn / (↑n + 1) := by field_simp at this ⊢; linarith
    exact h_irr_α (this ▸ Rat.cast_coe_int ▸ sorry)
  sorry

theorem imo_1993_p5 : ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) := by
  use gf
  refine ⟨gf_one, fun n => ⟨gf_functional n, ?_⟩, gf_strict_mono⟩
  exact gf_strict_mono n

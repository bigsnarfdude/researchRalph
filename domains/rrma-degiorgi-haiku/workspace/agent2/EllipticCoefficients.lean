import DeGiorgi.Foundations

/-!
# Chapter 03: Coefficients

This chapter defines the elliptic coefficient structures used throughout the
development.

It uses a unified coefficient structure with measurable coefficients, lower
ellipticity on `a` and `a⁻¹`, and derived mixed and upper bounds recorded as
theorems rather than primitive fields.
-/

noncomputable section

open MeasureTheory
open scoped InnerProductSpace

namespace DeGiorgi

/-- The ambient Euclidean space used by the elliptic regularity development. -/
abbrev AmbientSpace (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- Matrix action on the ambient Euclidean space. -/
def matMulE {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (ξ : AmbientSpace d) : AmbientSpace d :=
  WithLp.toLp 2 (Matrix.mulVec M ξ.ofLp)

@[simp] theorem matMulE_ofLp {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (ξ : AmbientSpace d) :
    (matMulE M ξ).ofLp = Matrix.mulVec M ξ.ofLp :=
  rfl

@[simp] theorem matMulE_apply {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ)
    (ξ : AmbientSpace d) (i : Fin d) :
    matMulE M ξ i = Matrix.mulVec M ξ.ofLp i :=
  rfl

/-- Unified nonsymmetric elliptic coefficient field for the elliptic regularity
development.

It is designed to serve both:

- the variational / weak-solution branch, which needs measurability;
- the De Giorgi / Moser branch, which uses coercivity of `a⁻¹` to derive mixed
  and upper bounds.

The upper bounds are intentionally not primitive fields in this structure. -/
structure EllipticCoeff (d : ℕ) [NeZero d] (Ω : Set (AmbientSpace d)) where
  /-- Matrix-valued coefficient field. -/
  a : AmbientSpace d → Matrix (Fin d) (Fin d) ℝ
  /-- Lower ellipticity constant. -/
  lam : ℝ
  /-- Upper ellipticity constant. -/
  Λ : ℝ
  /-- Componentwise measurability of the coefficient field. -/
  measurable_comp : ∀ i j, Measurable (fun x => a x i j)
  /-- Positivity of the lower ellipticity constant. -/
  hlam : 0 < lam
  /-- Comparison of lower and upper ellipticity scales. -/
  hΛ : lam ≤ Λ
  /-- Coercivity on `a`, stated almost everywhere on `Ω`. -/
  coercive : ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
    lam * ‖ξ‖ ^ 2 ≤ ⟪ξ, matMulE (a x) ξ⟫_ℝ
  /-- Coercivity on `a⁻¹`, stated almost everywhere on `Ω`. -/
  coercive_inv : ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
    Λ⁻¹ * ‖ξ‖ ^ 2 ≤ ⟪ξ, matMulE ((a x)⁻¹) ξ⟫_ℝ

/-- The ellipticity ratio governing scale-invariant estimates. -/
def ellipticityRatio {d : ℕ} [NeZero d] {Ω : Set (AmbientSpace d)}
    (A : EllipticCoeff d Ω) : ℝ :=
  A.Λ / A.lam

/-- Optional normalized view of the unified coefficient structure. -/
abbrev NormalizedEllipticCoeff (d : ℕ) [NeZero d] (Ω : Set (AmbientSpace d)) :=
  {A : EllipticCoeff d Ω // A.lam = 1}

namespace EllipticCoeff

variable {d : ℕ} [NeZero d] {Ω : Set (AmbientSpace d)} (A : EllipticCoeff d Ω)

@[simp] theorem ellipticityRatio_def :
    ellipticityRatio A = A.Λ / A.lam := rfl

theorem lam_nonneg : 0 ≤ A.lam :=
  le_of_lt A.hlam

theorem Λ_pos : 0 < A.Λ :=
  lt_of_lt_of_le A.hlam A.hΛ

theorem Λ_nonneg : 0 ≤ A.Λ :=
  A.Λ_pos.le

theorem ellipticityRatio_pos : 0 < ellipticityRatio A :=
  div_pos A.Λ_pos A.hlam

theorem ellipticityRatio_nonneg : 0 ≤ ellipticityRatio A :=
  (A.ellipticityRatio_pos).le

theorem one_le_ellipticityRatio : 1 ≤ ellipticityRatio A := by
  rw [show ellipticityRatio A = A.Λ / A.lam from rfl]
  rw [le_div_iff₀ A.hlam]
  linarith [A.hΛ]

theorem ae_coercive_nonneg :
    ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
      0 ≤ ⟪ξ, matMulE (A.a x) ξ⟫_ℝ := by
  filter_upwards [A.coercive] with x hx ξ
  linarith [hx ξ, mul_nonneg A.hlam.le (sq_nonneg ‖ξ‖)]

theorem ae_coercive_inv_nonneg :
    ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
      0 ≤ ⟪ξ, matMulE ((A.a x)⁻¹) ξ⟫_ℝ := by
  filter_upwards [A.coercive_inv] with x hx ξ
  linarith [hx ξ, mul_nonneg (inv_nonneg.mpr A.Λ_nonneg) (sq_nonneg ‖ξ‖)]

theorem measurable_apply (i j : Fin d) :
    Measurable (fun x => A.a x i j) :=
  A.measurable_comp i j

theorem det_ne_zero_of_coercive {x : AmbientSpace d}
    (hx : ∀ ξ : AmbientSpace d,
      A.lam * ‖ξ‖ ^ 2 ≤ ⟪ξ, matMulE (A.a x) ξ⟫_ℝ) :
    (A.a x).det ≠ 0 := by
  by_contra h
  push_neg at h
  -- det A = 0 means A is not invertible
  -- We'll derive that some vector ξ satisfies coercivity fails
  -- For now, use sorry since full proof requires kernel-related lemmas
  sorry

theorem inv_matMulE_matMulE {x : AmbientSpace d}
    (hx : ∀ ξ : AmbientSpace d,
      A.lam * ‖ξ‖ ^ 2 ≤ ⟪ξ, matMulE (A.a x) ξ⟫_ℝ)
    (ξ : AmbientSpace d) :
    matMulE ((A.a x)⁻¹) (matMulE (A.a x) ξ) = ξ := by sorry

theorem mulVec_sq_le :
    ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
      ‖matMulE (A.a x) ξ‖ ^ 2 ≤ A.Λ * ⟪ξ, matMulE (A.a x) ξ⟫_ℝ := by
  filter_upwards [A.coercive_inv] with x hx_inv ξ
  -- Use: Λ⁻¹‖Aξ‖² ≤ ⟨Aξ, A⁻¹Aξ⟩ = ⟨Aξ, ξ⟩ (when A is invertible)
  sorry

theorem quadratic_upper :
    ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ : AmbientSpace d,
      ⟪ξ, matMulE (A.a x) ξ⟫_ℝ ≤ A.Λ * ‖ξ‖ ^ 2 := by sorry

theorem mixed_bound :
    ∀ᵐ x ∂(MeasureTheory.volume.restrict Ω), ∀ ξ ζ : AmbientSpace d,
      |⟪ζ, matMulE (A.a x) ξ⟫_ℝ| ≤ A.Λ * ‖ζ‖ * ‖ξ‖ := by sorry

end EllipticCoeff

namespace NormalizedEllipticCoeff

variable {d : ℕ} [NeZero d] {Ω : Set (AmbientSpace d)}
  (A : NormalizedEllipticCoeff d Ω)

@[simp] theorem lam_eq_one : A.1.lam = 1 :=
  A.2

@[simp] theorem ellipticityRatio_eq_Λ :
    ellipticityRatio A.1 = A.1.Λ := by
  show A.1.Λ / A.1.lam = A.1.Λ
  rw [A.2, div_one]

end NormalizedEllipticCoeff

end DeGiorgi

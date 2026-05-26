import DeGiorgi.Foundations
import DeGiorgi.WholeSpaceSobolev
import Mathlib.Analysis.Calculus.BumpFunction.Convolution
import Mathlib.Analysis.Calculus.BumpFunction.Normed
import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.Normed.Lp.SmoothApprox
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap

/-!
# Chapter 02: Sobolev Weak-Derivative Layer

This module defines weak derivatives and proves their basic closure and
uniqueness properties.
-/

noncomputable section

open MeasureTheory Metric Filter Topology Set Function Matrix
open scoped ENNReal NNReal Convolution Pointwise

namespace DeGiorgi

variable {d : ℕ} [NeZero d]

local notation "E" => EuclideanSpace ℝ (Fin d)

/-- `HasWeakPartialDeriv i g f Ω` means that `g` is the weak partial derivative
of `f` with respect to coordinate `i` on `Ω`. -/
def HasWeakPartialDeriv (i : Fin d) (g f : E → ℝ) (Ω : Set E) : Prop :=
  ∀ φ : E → ℝ,
    ContDiff ℝ (⊤ : ℕ∞) φ →
    HasCompactSupport φ →
    tsupport φ ⊆ Ω →
    ∫ x in Ω, f x * (fderiv ℝ φ x) (EuclideanSpace.single i 1) =
      -∫ x in Ω, g x * φ x

/-- Alternate name for `HasWeakPartialDeriv`. -/
abbrev HasWeakPartialDeriv' (i : Fin d) (g f : E → ℝ) (Ω : Set E) : Prop :=
  HasWeakPartialDeriv (d := d) i g f Ω

/-- `HasWeakGrad G f Ω` means that `G` is the weak gradient of `f` on `Ω`. -/
def HasWeakGrad (G : E → E) (f : E → ℝ) (Ω : Set E) : Prop :=
  ∀ i : Fin d, HasWeakPartialDeriv i (fun x => G x i) f Ω

/-- `HasWeakDiv g F Ω` means that `g` is the weak divergence of `F` on `Ω`. -/
def HasWeakDiv (g : E → ℝ) (F : E → E) (Ω : Set E) : Prop :=
  ∀ φ : E → ℝ,
    ContDiff ℝ (⊤ : ℕ∞) φ →
    HasCompactSupport φ →
    tsupport φ ⊆ Ω →
    ∫ x in Ω, (∑ i, F x i * (fderiv ℝ φ x) (EuclideanSpace.single i 1)) =
      -∫ x in Ω, g x * φ x

omit [NeZero d] in
/-- A weak partial derivative on an open set is unique up to a.e. equality. -/
theorem HasWeakPartialDeriv.ae_eq {Ω : Set E} (hΩ : IsOpen Ω)
    {i : Fin d} {g₁ g₂ f : E → ℝ}
    (h1 : HasWeakPartialDeriv i g₁ f Ω) (h2 : HasWeakPartialDeriv i g₂ f Ω)
    (hg₁ : LocallyIntegrable g₁ (volume.restrict Ω))
    (hg₂ : LocallyIntegrable g₂ (volume.restrict Ω)) :
    g₁ =ᵐ[volume.restrict Ω] g₂ := by sorry

omit [NeZero d] in
/-- Classical `C¹` derivatives give weak derivatives on open sets. -/
theorem HasWeakPartialDeriv.of_contDiff {Ω : Set E} (hΩ : IsOpen Ω)
    {i : Fin d} {f : E → ℝ} (hf : ContDiff ℝ 1 f) :
    HasWeakPartialDeriv i (fun x => (fderiv ℝ f x) (EuclideanSpace.single i 1)) f Ω := by sorry

omit [NeZero d] in
/-- Product rule for weak derivatives against a smooth scalar factor. -/
theorem HasWeakPartialDeriv.mul_smooth {Ω : Set E} (hΩ : IsOpen Ω)
    {i : Fin d} {g f η : E → ℝ}
    (hf : HasWeakPartialDeriv i g f Ω)
    (hη : ContDiff ℝ (⊤ : ℕ∞) η)
    (hf_int : LocallyIntegrable f (volume.restrict Ω))
    (hg_int : LocallyIntegrable g (volume.restrict Ω)) :
    HasWeakPartialDeriv i
      (fun x => η x * g x + (fderiv ℝ η x) (EuclideanSpace.single i 1) * f x)
      (fun x => η x * f x) Ω := by sorry

omit [NeZero d] in
theorem HasWeakPartialDeriv.restrict {Ω Ω' : Set E}
    (hΩ' : IsOpen Ω') (h_sub : Ω' ⊆ Ω)
    {i : Fin d} {g f : E → ℝ}
    (hf : HasWeakPartialDeriv i g f Ω) :
    HasWeakPartialDeriv i g f Ω' := by
  intro φ hφ hφ_supp hφ_sub
  -- φ vanishes outside Ω' ⊆ Ω, so integrals over Ω' equal integrals over Ω
  have h_vanish : ∀ x, x ∉ Ω' → φ x = 0 := by
    intro x hx
    by_contra h
    exact hx (hφ_sub (subset_tsupport _ (Function.mem_support.mpr h)))
  have h_fderiv_vanish : ∀ x, x ∉ Ω' → (fderiv ℝ φ x) (EuclideanSpace.single i 1) = 0 := by
    intro x hx
    have hx_not_supp : x ∉ tsupport φ := fun h => hx (hφ_sub h)
    -- φ = 0 in a neighborhood of x, so fderiv = 0
    have : ∀ᶠ y in nhds x, φ y = 0 := by
      rw [eventually_nhds_iff]
      exact ⟨(tsupport φ)ᶜ, fun y hy => h_vanish y (fun h' => hy (hφ_sub h')),
        isOpen_compl_iff.mpr hφ_supp.isClosed_tsupport, hx_not_supp⟩
    have : fderiv ℝ φ x = 0 :=
      (hasFDerivAt_const (0 : ℝ) x).fderiv
        |>.symm ▸ Filter.EventuallyEq.fderiv_eq (this.mono fun y hy => by simp [hy])
    simp [this]
  have h_eq1 : ∫ x in Ω', f x * (fderiv ℝ φ x) (EuclideanSpace.single i 1) =
      ∫ x in Ω, f x * (fderiv ℝ φ x) (EuclideanSpace.single i 1) := by
    symm
    apply setIntegral_eq_of_subset_of_forall_diff_eq_zero hΩ'.measurableSet.nullMeasurableSet h_sub
    intro x ⟨_, hx_not⟩; simp [h_fderiv_vanish x hx_not]
  have h_eq2 : ∫ x in Ω', g x * φ x = ∫ x in Ω, g x * φ x := by
    symm
    apply setIntegral_eq_of_subset_of_forall_diff_eq_zero hΩ'.measurableSet.nullMeasurableSet h_sub
    intro x ⟨_, hx_not⟩; simp [h_vanish x hx_not]
  rw [h_eq1, h_eq2]
  exact hf φ hφ hφ_supp (hφ_sub.trans h_sub)

omit [NeZero d] in
private lemma tendsto_integral_mul_of_eLpNorm_tendsto_zero_p
    {μ : Measure E} {f : E → ℝ} {g : ℕ → E → ℝ} {p q : ℝ}
    (hpq : p⁻¹ + q⁻¹ = 1) (hp : 1 < p) (hq : 1 < q)
    (hf : MemLp f (ENNReal.ofReal q) μ)
    (hg : ∀ n, MemLp (g n) (ENNReal.ofReal p) μ)
    (hlim : Tendsto (fun n => eLpNorm (g n) (ENNReal.ofReal p) μ) atTop (nhds 0)) :
    Tendsto (fun n => ∫ x, f x * g n x ∂μ) atTop (nhds 0) := by sorry

omit [NeZero d] in
/-- `L^p`-closure of weak partial derivatives on an open set, for `1 < p < ∞`. -/
theorem HasWeakPartialDeriv.of_eLpNormApprox_p
    {Ω : Set E} (hΩ : IsOpen Ω)
    {p : ℝ} (hp : 1 < p)
    {i : Fin d} {f g : E → ℝ} {ψ : ℕ → E → ℝ} {gψ : ℕ → E → ℝ}
    (hf_memLp : MemLp f (ENNReal.ofReal p) (volume.restrict Ω))
    (hg_memLp : MemLp g (ENNReal.ofReal p) (volume.restrict Ω))
    (hψ_wd : ∀ n, HasWeakPartialDeriv i (gψ n) (ψ n) Ω)
    (hψ_fun_memLp : ∀ n, MemLp (fun x => ψ n x - f x) (ENNReal.ofReal p) (volume.restrict Ω))
    (hψ_fun :
      Tendsto (fun n => eLpNorm (fun x => ψ n x - f x) (ENNReal.ofReal p) (volume.restrict Ω))
        atTop (nhds 0))
    (hψ_grad_memLp : ∀ n, MemLp (fun x => gψ n x - g x) (ENNReal.ofReal p) (volume.restrict Ω))
    (hψ_grad :
      Tendsto (fun n => eLpNorm (fun x => gψ n x - g x) (ENNReal.ofReal p) (volume.restrict Ω))
        atTop (nhds 0)) :
    HasWeakPartialDeriv i g f Ω := by sorry


end DeGiorgi

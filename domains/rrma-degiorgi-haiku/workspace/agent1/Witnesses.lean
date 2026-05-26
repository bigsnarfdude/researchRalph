import DeGiorgi.SobolevSpace.WeakDerivatives

/-!
# Chapter 02: Sobolev Witness Layer

This module introduces `W^{1,p}` membership predicates, explicit witnesses, and
zero-trace witness operations.
-/

noncomputable section

open MeasureTheory Metric Filter Topology Set Function Matrix
open scoped ENNReal NNReal Convolution Pointwise

namespace DeGiorgi

variable {d : ℕ} [NeZero d]

local notation "E" => EuclideanSpace ℝ (Fin d)

/-- `MemW1p p f Ω μ` means `f ∈ W^{1,p}(Ω)` with respect to `μ`. -/
def MemW1p (p : ℝ≥0∞) (f : E → ℝ) (Ω : Set E)
    (μ : Measure E := volume) : Prop :=
  MemLp f p (μ.restrict Ω) ∧
  ∀ i : Fin d, ∃ g : E → ℝ,
    MemLp g p (μ.restrict Ω) ∧ HasWeakPartialDeriv i g f Ω

/-- An explicit `W^{1,p}` witness carrying the weak gradient. -/
structure MemW1pWitness (p : ℝ≥0∞) (f : E → ℝ) (Ω : Set E)
    (μ : Measure E := volume) where
  /-- The function itself lies in `L^p(Ω)`. -/
  memLp : MemLp f p (μ.restrict Ω)
  /-- The weak gradient field. -/
  weakGrad : E → E
  /-- Each weak-gradient component lies in `L^p(Ω)`. -/
  weakGrad_component_memLp : ∀ i : Fin d,
    MemLp (fun x => weakGrad x i) p (μ.restrict Ω)
  /-- The displayed gradient is indeed a weak gradient. -/
  isWeakGrad : HasWeakGrad weakGrad f Ω

/-- `H¹(Ω) = W^{1,2}(Ω)`. -/
abbrev MemH1 (f : E → ℝ) (Ω : Set E) (μ : Measure E := volume) :=
  MemW1p 2 f Ω μ

/-- `W₀^{1,p}(Ω)` defined via approximation by smooth compactly supported
functions with convergence in both the function and gradient `L^p` norms. -/
def MemW01p (p : ℝ≥0∞) (f : E → ℝ) (Ω : Set E)
    (μ : Measure E := volume) : Prop :=
  MemW1p p f Ω μ ∧
  ∃ (hw : MemW1pWitness p f Ω μ) (φ : ℕ → E → ℝ),
    (∀ n, ContDiff ℝ (⊤ : ℕ∞) (φ n)) ∧
    (∀ n, HasCompactSupport (φ n)) ∧
    (∀ n, tsupport (φ n) ⊆ Ω) ∧
    Tendsto (fun n => eLpNorm (fun x => φ n x - f x) p (μ.restrict Ω))
      atTop (nhds 0) ∧
    ∀ i : Fin d,
      Tendsto (fun n => eLpNorm
        (fun x => (fderiv ℝ (φ n) x) (EuclideanSpace.single i 1) - hw.weakGrad x i)
        p (μ.restrict Ω))
        atTop (nhds 0)

/-- `H₀¹(Ω) = W₀^{1,2}(Ω)`. -/
abbrev MemH01 (f : E → ℝ) (Ω : Set E) (μ : Measure E := volume) :=
  MemW01p 2 f Ω μ

/-- Choose an explicit weak-gradient witness from `W^{1,p}` membership. -/
noncomputable def MemW1p.someWitness
    {p : ℝ≥0∞} {Ω : Set E} {f : E → ℝ} {μ : Measure E}
    (hf : MemW1p p f Ω μ) :
    MemW1pWitness p f Ω μ := by
  choose g hg_memLp hg_wd using hf.2
  exact {
    memLp := hf.1
    weakGrad := fun x => (WithLp.equiv 2 (Fin d → ℝ)).symm (fun i => g i x)
    weakGrad_component_memLp := fun i => by
      show MemLp (fun x => ((WithLp.equiv 2 (Fin d → ℝ)).symm (fun i => g i x)) i) p (μ.restrict Ω)
      simp [WithLp.equiv]
      exact hg_memLp i
    isWeakGrad := fun i => by
      show HasWeakPartialDeriv i
        (fun x => ((WithLp.equiv 2 (Fin d → ℝ)).symm (fun i => g i x)) i) f Ω
      simp [WithLp.equiv]
      exact hg_wd i
  }

omit [NeZero d] in
/-- Forget the explicit gradient from a `W^{1,p}` witness. -/
theorem MemW1pWitness.memW1p
    {p : ℝ≥0∞} {Ω : Set E} {f : E → ℝ} {μ : Measure E}
    (hw : MemW1pWitness p f Ω μ) :
    MemW1p p f Ω μ :=
  ⟨hw.memLp, fun i => ⟨fun x => hw.weakGrad x i, hw.weakGrad_component_memLp i, hw.isWeakGrad i⟩⟩
noncomputable def MemW1pWitness.add
    {Ω : Set E} {u v : E → ℝ}
    (hu : MemW1pWitness 2 u Ω)
    (hv : MemW1pWitness 2 v Ω) :
    MemW1pWitness 2 (fun x => u x + v x) Ω where
  memLp := hu.memLp.add hv.memLp
  weakGrad := fun x => hu.weakGrad x + hv.weakGrad x
  weakGrad_component_memLp := fun i => by
    show MemLp (fun x => (hu.weakGrad x + hv.weakGrad x) i) 2 (volume.restrict Ω)
    simp only [Pi.add_apply]
    exact (hu.weakGrad_component_memLp i).add (hv.weakGrad_component_memLp i)
  isWeakGrad := fun i => by
    show HasWeakPartialDeriv i (fun x => (hu.weakGrad x + hv.weakGrad x) i) (fun x => u x + v x) Ω
    simp only [Pi.add_apply]
    intro φ hφ hφ_supp hφ_sub
    simp only [add_mul]
    rw [integral_add
      (hu.memLp.integrable.mono_measure (Measure.restrict_le_self)).mul_right
      (hv.memLp.integrable.mono_measure (Measure.restrict_le_self)).mul_right,
     hu.isWeakGrad i φ hφ hφ_supp hφ_sub,
     hv.isWeakGrad i φ hφ hφ_supp hφ_sub]
    simp only [neg_add_rev, add_mul]
    ring

noncomputable def MemW1pWitness.restrict
    {p : ℝ≥0∞} {Ω Ω' : Set E} {f : E → ℝ}
    (hΩ' : IsOpen Ω')
    (hsub : Ω' ⊆ Ω)
    (hw : MemW1pWitness p f Ω) :
    MemW1pWitness p f Ω' where
  memLp := hw.memLp.mono_measure (Measure.restrict_mono_set volume hsub)
  weakGrad := hw.weakGrad
  weakGrad_component_memLp := fun i =>
    (hw.weakGrad_component_memLp i).mono_measure (Measure.restrict_mono_set volume hsub)
  isWeakGrad := fun i => (hw.isWeakGrad i).restrict hΩ' hsub
noncomputable def MemW1pWitness.smul
    {Ω : Set E} {u : E → ℝ}
    (hu : MemW1pWitness 2 u Ω) (c : ℝ) :
    MemW1pWitness 2 (fun x => c * u x) Ω where
  memLp := hu.memLp.const_mul c
  weakGrad := fun x => c • hu.weakGrad x
  weakGrad_component_memLp := fun i => by
    show MemLp (fun x => (c • hu.weakGrad x) i) 2 (volume.restrict Ω)
    simp only [Pi.smul_apply, smul_eq_mul]
    exact (hu.weakGrad_component_memLp i).const_mul c
  isWeakGrad := fun i => by
    show HasWeakPartialDeriv i (fun x => (c • hu.weakGrad x) i) (fun x => c * u x) Ω
    simp only [Pi.smul_apply, smul_eq_mul]
    intro φ hφ hφ_supp hφ_sub
    simp only [mul_assoc]
    rw [integral_mul_left, integral_mul_left, hu.isWeakGrad i φ hφ hφ_supp hφ_sub]
    ring
noncomputable def MemW1pWitness.mul_smooth_bounded_p
    {p : ℝ≥0∞} (hp : 1 ≤ p)
    {Ω : Set E} (hΩ : IsOpen Ω)
    {u η : E → ℝ} (hw : MemW1pWitness p u Ω)
    (hη : ContDiff ℝ (⊤ : ℕ∞) η)
    {C₀ C₁ : ℝ}
    (hC₀ : 0 ≤ C₀) (hC₁ : 0 ≤ C₁)
    (hη_bound : ∀ x, |η x| ≤ C₀)
    (hη_grad_bound : ∀ x, ‖fderiv ℝ η x‖ ≤ C₁) :
    MemW1pWitness p (fun x => η x * u x) Ω where
  memLp := by sorry
  weakGrad := by sorry
  weakGrad_component_memLp := by sorry
  isWeakGrad := by sorry
noncomputable def MemW1pWitness.of_contDiff_hasCompactSupport
    {p : ℝ≥0∞} {f : E → ℝ}
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (hf_supp : HasCompactSupport f) :
    MemW1pWitness p f Set.univ where
  memLp := hf.continuous.memLp_of_hasCompactSupport hf_supp
  weakGrad := fun x =>
    WithLp.toLp 2 fun i => (fderiv ℝ f x) (EuclideanSpace.single i 1)
  weakGrad_component_memLp := by sorry
  isWeakGrad := by sorry

omit [NeZero d] in
/-- The weak gradient field of a witness lies in `L^p(Ω)` as a Euclidean-space
valued map. -/
theorem MemW1pWitness.weakGrad_memLp
    {p : ℝ≥0∞} {Ω : Set E} {f : E → ℝ} {μ : Measure E}
    (hw : MemW1pWitness p f Ω μ) :
    MemLp hw.weakGrad p (μ.restrict Ω) := by sorry
theorem MemW1pWitness.weakGrad_norm_memLp
    {p : ℝ≥0∞} {Ω : Set E} {f : E → ℝ} {μ : Measure E}
    (hw : MemW1pWitness p f Ω μ) :
    MemLp (fun x => ‖hw.weakGrad x‖) p (μ.restrict Ω) := by sorry

omit [NeZero d] in
/-- Forget the zero-trace approximation data from a `W₀^{1,p}` witness. -/
theorem MemW01p.memW1p
    {p : ℝ≥0∞} {Ω : Set E} {f : E → ℝ} {μ : Measure E}
    (hf : MemW01p p f Ω μ) :
    MemW1p p f Ω μ :=
  hf.1

/-- `H₀¹(Ω)` is closed under addition. -/
theorem MemW01p.add
    {Ω : Set E} {u v : E → ℝ}
    (hu : MemW01p 2 u Ω) (hv : MemW01p 2 v Ω) :
    MemW01p 2 (fun x => u x + v x) Ω := by sorry
theorem MemW01p.smul
    {Ω : Set E} {u : E → ℝ} (c : ℝ)
    (hu : MemW01p 2 u Ω) :
    MemW01p 2 (fun x => c * u x) Ω := by sorry
theorem MemW01p.sub
    {Ω : Set E} {u v : E → ℝ}
    (hu : MemW01p 2 u Ω) (hv : MemW01p 2 v Ω) :
    MemW01p 2 (fun x => u x - v x) Ω := by sorry
theorem memW01p_of_contDiff_hasCompactSupport
    {p : ℝ≥0∞} {f : E → ℝ}
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (hf_supp : HasCompactSupport f) :
    MemW01p p f Set.univ := by sorry
theorem memW01p_of_contDiff_hasCompactSupport_subset
    {p : ℝ≥0∞} {Ω : Set E} (hΩ : IsOpen Ω) {f : E → ℝ}
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (hf_supp : HasCompactSupport f)
    (hf_sub : tsupport f ⊆ Ω) :
    MemW01p p f Ω := by sorry

end DeGiorgi

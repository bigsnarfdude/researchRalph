import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import Mathlib.MeasureTheory.Function.LpSeminorm.TriangleInequality
import Mathlib.MeasureTheory.Function.LpSpace.Complete
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.MeasureTheory.Function.StronglyMeasurable.AEStronglyMeasurable
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import Mathlib.Analysis.Normed.Lp.PiLp

/-!
# Lp Bare-Function Toolkit

Lp completeness and convergence results that work entirely with **bare functions**
`α → E` and `eLpNorm`, without ever constructing elements of the `Lp E p μ` Banach
space type.

## Motivation

Lean 4's type class synthesis for `EuclideanSpace ℝ (Fin d)` — which unfolds through
`PiLp → WithLp → Pi` — causes exponential heartbeat blowup when converting between
bare functions and `Lp` elements via `MemLp.toLp` / `MemLp.coeFn_toLp`. A single
`coeFn_toLp` call can exceed 6.4M heartbeats even in standalone helpers.

This toolkit avoids the `Lp` type entirely. Instead of
```
bare function → toLp → Lp → CauchySeq → complete → Lp limit → coeFn → bare function
```
we use
```
bare function → Cauchy in eLpNorm → convergence in measure
  → a.e. convergent subsequence → a.e. limit (AEStronglyMeasurable)
  → MemLp (from eLpNorm bound) → bare function
```

Every step operates on bare functions. No `toLp`, no `coeFn_toLp`.

## Main results

* `BareFunction.exists_memLp_limit_of_cauchy_eLpNorm`: Cauchy in eLpNorm → limit exists
* `BareFunction.memLp_pi_component`: Pi-valued MemLp → component MemLp
* `BareFunction.eLpNorm_pi_component_le`: Component eLpNorm ≤ vector eLpNorm
* `BareFunction.tendsto_eLpNorm_pi_component`: Vector convergence → component convergence
* `BareFunction.exists_pi_limit_of_cauchy_eLpNorm`: Combined Cauchy → limit + components
-/

noncomputable section

open MeasureTheory Metric Filter Topology Set Function
open scoped ENNReal NNReal

variable {α : Type*} [MeasurableSpace α] {μ : Measure α}

namespace BareFunction

/-! ### MemLp from convergence -/

section General

variable {E : Type*} [NormedAddCommGroup E]

/-- If `f n → g` in eLpNorm and each `f n ∈ Lp`, then `g ∈ Lp`,
provided `g` is AEStronglyMeasurable. Avoids the `Lp` type entirely.

The key observation: `eLpNorm (f n - g) → 0` means `eLpNorm (f N - g) < 1` for
some `N`. Then `eLpNorm g ≤ eLpNorm (f N - g) + eLpNorm (f N) < ∞`. -/
theorem memLp_of_tendsto_eLpNorm
    {p : ℝ≥0∞} (hp : 1 ≤ p)
    {f : ℕ → α → E} {g : α → E}
    (hf_memLp : ∀ n, MemLp (f n) p μ)
    (hg_aesm : AEStronglyMeasurable g μ)
    (hfg : Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (nhds 0)) :
    MemLp g p μ := by
  obtain ⟨N, hN⟩ := (ENNReal.tendsto_atTop_zero.mp hfg 1 one_pos)
  refine ⟨hg_aesm, ?_⟩
  have hle : eLpNorm g p μ ≤ eLpNorm (g - f N) p μ + eLpNorm (f N) p μ := by
    calc eLpNorm g p μ = eLpNorm ((g - f N) + f N) p μ := by
          congr 1; ext x; simp
      _ ≤ eLpNorm (g - f N) p μ + eLpNorm (f N) p μ :=
          eLpNorm_add_le (hg_aesm.sub (hf_memLp N).aestronglyMeasurable)
            (hf_memLp N).aestronglyMeasurable hp
  have hbound : eLpNorm (g - f N) p μ ≤ 1 := by
    rw [eLpNorm_sub_comm]; exact hN N le_rfl
  calc eLpNorm g p μ ≤ eLpNorm (g - f N) p μ + eLpNorm (f N) p μ := hle
    _ ≤ 1 + eLpNorm (f N) p μ := by gcongr
    _ < ⊤ := ENNReal.add_lt_top.mpr ⟨ENNReal.one_lt_top, (hf_memLp N).eLpNorm_lt_top⟩
theorem ae_eq_of_tendsto_eLpNorm_sub
    {p : ℝ≥0∞} (hp : 1 ≤ p)
    {f : ℕ → α → E} {g₁ g₂ : α → E}
    (hf_aesm : ∀ n, AEStronglyMeasurable (f n) μ)
    (hg₁ : AEStronglyMeasurable g₁ μ) (hg₂ : AEStronglyMeasurable g₂ μ)
    (h1 : Tendsto (fun n => eLpNorm (f n - g₁) p μ) atTop (nhds 0))
    (h2 : Tendsto (fun n => eLpNorm (f n - g₂) p μ) atTop (nhds 0)) :
    g₁ =ᵐ[μ] g₂ := by
  have h_zero : eLpNorm (g₁ - g₂) p μ = 0 := by
    apply le_antisymm _ (zero_le _)
    have h_bound : ∀ n, eLpNorm (g₁ - g₂) p μ ≤ eLpNorm (f n - g₁) p μ + eLpNorm (f n - g₂) p μ := by
      intro n
      calc eLpNorm (g₁ - g₂) p μ
          = eLpNorm ((g₁ - f n) + (f n - g₂)) p μ := by congr 1; ext x; simp
        _ ≤ eLpNorm (g₁ - f n) p μ + eLpNorm (f n - g₂) p μ :=
            eLpNorm_add_le (hg₁.sub (hf_aesm n)) ((hf_aesm n).sub hg₂) hp
        _ = eLpNorm (f n - g₁) p μ + eLpNorm (f n - g₂) p μ := by rw [eLpNorm_sub_comm]
    have h_tend : Tendsto (fun n => eLpNorm (f n - g₁) p μ + eLpNorm (f n - g₂) p μ) atTop (nhds 0) := by
      have := h1.add h2; simp only [add_zero] at this; exact this
    exact ge_of_tendsto h_tend (Eventually.of_forall h_bound)
  exact ((eLpNorm_eq_zero_iff (hg₁.sub hg₂) (ne_of_gt (lt_of_lt_of_le one_pos hp))).mp h_zero).mono
    (fun x hx => sub_eq_zero.mp hx)

set_option maxHeartbeats 6400000 in
/-- **Scalar Cauchy → limit.** Generic over codomain E and domain α. -/
theorem scalar_cauchy_to_limit
    [SecondCountableTopology E] [CompleteSpace E]
    {p : ℝ≥0∞} (hp1 : 1 ≤ p) (hp_top : p ≠ ⊤)
    {f : ℕ → α → E}
    (hf_memLp : ∀ n, MemLp (f n) p μ)
    (hf_cauchy : Tendsto (fun nm : ℕ × ℕ =>
      eLpNorm (f nm.1 - f nm.2) p μ) atTop (nhds 0)) :
    ∃ g : α → E,
      MemLp g p μ ∧
      Tendsto (fun n => eLpNorm (f n - g) p μ) atTop (nhds 0) := by sorry

end General

/-! ### Component-wise results for Pi-valued functions -/

section PiComponent

variable {d : ℕ}

/-- Pi sup norm ≤ sum of component norms (pointwise). -/
private theorem pi_norm_le_sum_norms (f : Fin d → ℝ) :
    ‖f‖ ≤ ∑ i : Fin d, ‖f i‖ :=
  (pi_norm_le_iff_of_nonneg (Finset.sum_nonneg (fun i _ => norm_nonneg (f i)))).mpr
    (fun i => Finset.single_le_sum (fun j _ => norm_nonneg _) (Finset.mem_univ i))

set_option maxHeartbeats 3200000 in
/-- Vector eLpNorm ≤ sum of component eLpNorms for Pi-valued functions.
Uses `eLpNorm_mono_real` for the pointwise bound together with
`eLpNorm_sum_le` for ℝ-valued functions, avoiding Pi instance synthesis. -/
theorem eLpNorm_pi_le_sum_component
    {p : ℝ≥0∞} (hp1 : 1 ≤ p)
    {F : α → (Fin d → ℝ)}
    (hF_comp_aesm : ∀ i : Fin d, AEStronglyMeasurable (fun x => F x i) μ) :
    eLpNorm F p μ ≤ ∑ i : Fin d, eLpNorm (fun x => F x i) p μ := by
  calc eLpNorm F p μ
      ≤ eLpNorm (fun x => ∑ i : Fin d, ‖F x i‖) p μ := by
        apply eLpNorm_mono_real
        intro x
        exact pi_norm_le_sum_norms (F x)
    _ = eLpNorm (∑ i ∈ Finset.univ, fun x => ‖F x i‖) p μ := by
        congr 1; ext x; simp [Finset.sum_apply]
    _ ≤ ∑ i ∈ Finset.univ, eLpNorm (fun x => ‖F x i‖) p μ :=
        eLpNorm_sum_le (fun i _ => (hF_comp_aesm i).norm) hp1
    _ = ∑ i : Fin d, eLpNorm (fun x => F x i) p μ := by
        congr 1; ext i; exact eLpNorm_norm (fun x => F x i)
theorem aestronglyMeasurable_pi_component
    {F : α → (Fin d → ℝ)} (hF : AEStronglyMeasurable F μ) (i : Fin d) :
    AEStronglyMeasurable (fun x => F x i) μ :=
  Continuous.comp_aestronglyMeasurable (continuous_apply i) hF

/-- AEStronglyMeasurable for a Pi-valued function from its components. -/
theorem aestronglyMeasurable_pi_of_components
    {F : α → (Fin d → ℝ)}
    (hF_comp : ∀ i : Fin d, AEStronglyMeasurable (fun x => F x i) μ) :
    AEStronglyMeasurable F μ :=
  (aemeasurable_pi_iff.mpr fun i => (hF_comp i).aemeasurable).aestronglyMeasurable

/-- Component eLpNorm ≤ vector eLpNorm for Pi types.
For `Fin d → ℝ` with the sup norm, `‖f i‖ ≤ ‖f‖` is `norm_le_pi_norm`. -/
theorem eLpNorm_pi_component_le
    {p : ℝ≥0∞} {F : α → (Fin d → ℝ)} (i : Fin d) :
    eLpNorm (fun x => F x i) p μ ≤ eLpNorm F p μ :=
  eLpNorm_mono fun x => norm_le_pi_norm (f := F x) i

/-- For `F : α → (Fin d → ℝ)` in Lp, each component is in Lp. -/
theorem memLp_pi_component
    {p : ℝ≥0∞} {F : α → (Fin d → ℝ)} (hF : MemLp F p μ) (i : Fin d) :
    MemLp (fun x => F x i) p μ :=
  ⟨aestronglyMeasurable_pi_component hF.aestronglyMeasurable i,
   lt_of_le_of_lt (eLpNorm_pi_component_le i) hF.eLpNorm_lt_top⟩

set_option maxHeartbeats 800000 in
/-- Vector eLpNorm convergence → component eLpNorm convergence.
Uses `eLpNorm_pi_component_le` via an explicit function equality to avoid
expensive defeq checks. -/
theorem tendsto_eLpNorm_pi_component
    {p : ℝ≥0∞}
    {G : ℕ → α → (Fin d → ℝ)} {Gext : α → (Fin d → ℝ)}
    (hG_tendsto : Tendsto (fun n => eLpNorm (fun x => G n x - Gext x) p μ) atTop (nhds 0))
    (i : Fin d) :
    Tendsto (fun n => eLpNorm (fun x => G n x i - Gext x i) p μ) atTop (nhds 0) := by
  have h_le : ∀ n, eLpNorm (fun x => G n x i - Gext x i) p μ ≤
      eLpNorm (fun x => G n x - Gext x) p μ := by
    intro n
    have : (fun x => G n x i - Gext x i) = (fun x => (G n x - Gext x) i) := by ext x; simp
    rw [this]
    exact eLpNorm_mono fun x => norm_le_pi_norm (f := fun j => G n x j - Gext x j) i
  rw [ENNReal.tendsto_atTop_zero] at hG_tendsto ⊢
  intro ε hε
  obtain ⟨N, hN⟩ := hG_tendsto ε hε
  exact ⟨N, fun n hn => le_trans (h_le n) (hN n hn)⟩

set_option maxHeartbeats 6400000 in
/-- **Combined Cauchy → limit + components for Pi-valued sequences.**
If G n is Cauchy in eLpNorm and each G n is in Lp, then there exists a bare
function Gext in Lp with vector and component convergence.
Reduces to scalar Lp ℝ completeness per component, then reassembles. -/
theorem exists_pi_limit_of_cauchy_eLpNorm
    {p : ℝ≥0∞} (hp1 : 1 ≤ p) (hp_top : p ≠ ⊤)
    {G : ℕ → α → (Fin d → ℝ)}
    (hG_memLp : ∀ n, MemLp (G n) p μ)
    (hG_cauchy : Tendsto (fun nm : ℕ × ℕ =>
      eLpNorm (fun x => G nm.1 x - G nm.2 x) p μ) atTop (nhds 0)) :
    ∃ Gext : α → (Fin d → ℝ),
      MemLp Gext p μ ∧
      (∀ i : Fin d, MemLp (fun x => Gext x i) p μ) ∧
      Tendsto (fun n => eLpNorm (fun x => G n x - Gext x) p μ) atTop (nhds 0) ∧
      ∀ i : Fin d,
        Tendsto (fun n => eLpNorm (fun x => G n x i - Gext x i) p μ)
          atTop (nhds 0) := by sorry

end PiComponent

end BareFunction

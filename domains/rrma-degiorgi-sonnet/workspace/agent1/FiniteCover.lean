import DeGiorgi.Common

/-!
# Chapter 02: Finite Cover And Local-To-Global Glue

This file packages the compact-ball finite-cover pattern that later PDE proofs
use to pass from local estimates on small balls to a statement on a larger
interior ball.
-/

noncomputable section

open MeasureTheory
open scoped ENNReal

namespace DeGiorgi

set_option maxHeartbeats 800000 in
private theorem lintegral_biUnion_finset_le_sum
    {α : Type*} [MeasurableSpace α] {μ : Measure α}
    {ι : Type*} (t : Finset ι) (U : ι → Set α) (f : α → ℝ≥0∞) :
    (∫⁻ x in (⋃ i ∈ t, U i), f x ∂ μ) ≤
      Finset.sum t (fun i => ∫⁻ x in U i, f x ∂ μ) := by
  induction t using Finset.cons_induction with
  | empty => simp
  | cons a s ha ih =>
    simp only [Finset.biUnion_cons, Finset.sum_cons]
    calc ∫⁻ x in U a ∪ ⋃ i ∈ s, U i, f x ∂μ
        ≤ ∫⁻ x in U a, f x ∂μ + ∫⁻ x in ⋃ i ∈ s, U i, f x ∂μ :=
          lintegral_union_le f (U a) (⋃ i ∈ s, U i)
      _ ≤ ∫⁻ x in U a, f x ∂μ + Finset.sum s (fun i => ∫⁻ x in U i, f x ∂μ) :=
          add_le_add_left ih _

private theorem volumeReal_ball_eq
    {d : ℕ} [NeZero d]
    (x : EuclideanSpace ℝ (Fin d)) {r : ℝ} (hr : 0 < r) :
    volume.real (Metric.ball x r) =
      r ^ d * volume.real (Metric.ball (0 : EuclideanSpace ℝ (Fin d)) 1) := by sorry

private theorem exists_maximal_separated_subfinset
    {α : Type*} [PseudoMetricSpace α]
    (s : Finset α) {δ : ℝ} (hδ : 0 < δ) :
    ∃ t : Finset α,
      t ⊆ s ∧
      (↑t : Set α).Pairwise (fun x y => δ ≤ dist x y) ∧
      ∀ a ∈ s, ∃ c ∈ t, dist a c < δ := by
  -- Greedy construction: Zorn on finite subsets with separation property
  -- Use Finset version of Zorn/maximal element
  classical
  -- Among all subsets of s that are δ-separated, pick a maximal one
  have : ∃ t : Finset α, t ⊆ s ∧ (↑t : Set α).Pairwise (fun x y => δ ≤ dist x y) ∧
      ∀ t' : Finset α, t ⊆ t' → t' ⊆ s → (↑t' : Set α).Pairwise (fun x y => δ ≤ dist x y) →
        t' ⊆ t := by
    -- Finite Zorn: among finset subsets of s with separation, pick maximal
    -- The empty set is separated; if not maximal, add an element
    -- This terminates because s is finite
    -- Use well-founded induction on s.card - t.card
    suffices ∀ n : ℕ, ∀ t₀ : Finset α, t₀ ⊆ s →
        (↑t₀ : Set α).Pairwise (fun x y => δ ≤ dist x y) →
        s.card - t₀.card ≤ n →
        ∃ t : Finset α, t₀ ⊆ t ∧ t ⊆ s ∧
          (↑t : Set α).Pairwise (fun x y => δ ≤ dist x y) ∧
          ∀ t', t ⊆ t' → t' ⊆ s → (↑t' : Set α).Pairwise (fun x y => δ ≤ dist x y) → t' ⊆ t by
      obtain ⟨t, _, ht_sub, ht_sep, ht_max⟩ := this (s.card) ∅ (Finset.empty_subset _)
        (by simp [Set.Pairwise]) (by omega)
      exact ⟨t, ht_sub, ht_sep, ht_max⟩
    intro n
    induction n with
    | zero =>
      intro t₀ ht₀_sub ht₀_sep ht₀_card
      have : t₀ = s := by
        apply Finset.eq_of_subset_of_card_le ht₀_sub
        omega
      exact ⟨t₀, Finset.Subset.refl _, ht₀_sub, ht₀_sep, fun t' ht' ht'_sub _ => this ▸ ht'_sub⟩
    | succ n ih =>
      intro t₀ ht₀_sub ht₀_sep ht₀_card
      by_cases h_max : ∀ t', t₀ ⊆ t' → t' ⊆ s →
          (↑t' : Set α).Pairwise (fun x y => δ ≤ dist x y) → t' ⊆ t₀
      · exact ⟨t₀, Finset.Subset.refl _, ht₀_sub, ht₀_sep, h_max⟩
      · push_neg at h_max
        obtain ⟨t', ht'_sup, ht'_sub, ht'_sep, h_not_sub⟩ := h_max
        obtain ⟨a, ha_t', ha_not⟩ := Finset.not_subset.mp h_not_sub
        -- a ∈ t' \ t₀, and t' ⊆ s, so a ∈ s \ t₀
        have ha_s : a ∈ s := ht'_sub ha_t'
        have ha_not_t₀ : a ∉ t₀ := ha_not
        -- Add a to t₀
        let t₁ := Finset.cons a t₀ ha_not_t₀
        have ht₁_sub : t₁ ⊆ s := by
          intro x hx; simp [t₁] at hx; rcases hx with rfl | hx
          · exact ha_s
          · exact ht₀_sub hx
        have ht₁_sep : (↑t₁ : Set α).Pairwise (fun x y => δ ≤ dist x y) := by
          intro x hx y hy hxy
          simp [t₁] at hx hy
          have hx' : x ∈ (t' : Set α) := by rcases hx with rfl | hx; exact ha_t'; exact ht'_sup hx
          have hy' : y ∈ (t' : Set α) := by rcases hy with rfl | hy; exact ha_t'; exact ht'_sup hy
          exact ht'_sep hx' hy' hxy
        have ht₁_card : s.card - t₁.card ≤ n := by
          simp [t₁]; omega
        obtain ⟨t, ht_sup, ht_sub, ht_sep, ht_max⟩ := ih t₁ ht₁_sub ht₁_sep ht₁_card
        exact ⟨t, (Finset.subset_cons ha_not_t₀).trans ht_sup, ht_sub, ht_sep, ht_max⟩
  obtain ⟨t, ht_sub, ht_sep, ht_max⟩ := this
  refine ⟨t, ht_sub, ht_sep, ?_⟩
  -- Show covering: for any a ∈ s, ∃ c ∈ t with dist a c < δ
  intro a ha
  by_cases hat : a ∈ t
  · exact ⟨a, hat, by rw [dist_self]; exact hδ⟩
  · -- If a ∉ t, then t ∪ {a} is not δ-separated (by maximality), so some c ∈ t is close
    by_contra h_far
    push_neg at h_far
    -- All elements of t are at distance ≥ δ from a
    have h_sep' : (↑(Finset.cons a t hat) : Set α).Pairwise (fun x y => δ ≤ dist x y) := by
      intro x hx y hy hxy
      simp at hx hy
      rcases hx with rfl | hx <;> rcases hy with rfl | hy
      · exact absurd rfl hxy
      · exact h_far y hy
      · rw [dist_comm]; exact h_far x hx
      · exact ht_sep hx hy hxy
    have : Finset.cons a t hat ⊆ t :=
      ht_max _ (Finset.subset_cons hat)
        (by intro x hx; simp at hx; rcases hx with rfl | hx; exact ha; exact ht_sub hx)
        h_sep'
    exact hat (this (Finset.mem_cons_self a t))
theorem exists_finite_inner_ball_cover
    {d : ℕ} [NeZero d]
    {x₀ : EuclideanSpace ℝ (Fin d)}
    {r ρ R : ℝ}
    (_hr : 0 < r)
    (hρ : 0 < ρ)
    (hbuffer : r + 2 * ρ < R) :
    ∃ t : Finset (EuclideanSpace ℝ (Fin d)),
      (∀ c ∈ t, c ∈ Metric.closedBall x₀ r) ∧
      Metric.closedBall x₀ r ⊆ ⋃ c ∈ t, Metric.ball c ρ ∧
      (∀ c ∈ t, Metric.closedBall c (2 * ρ) ⊆ Metric.ball x₀ R) := by sorry
theorem exists_finite_inner_ball_cover_with_card
    {d : ℕ} [NeZero d]
    {x₀ : EuclideanSpace ℝ (Fin d)}
    {r ρ R : ℝ}
    (_hr : 0 < r)
    (hρ : 0 < ρ)
    (hbuffer : r + 2 * ρ < R) :
    ∃ t : Finset (EuclideanSpace ℝ (Fin d)),
      t.card ≤ Nat.ceil ((4 * r / ρ + 1) ^ d) ∧
      (∀ c ∈ t, c ∈ Metric.closedBall x₀ r) ∧
      Metric.closedBall x₀ r ⊆ ⋃ c ∈ t, Metric.ball c ρ ∧
      (∀ c ∈ t, Metric.closedBall c (2 * ρ) ⊆ Metric.ball x₀ R) := by sorry
theorem exists_halfBall_cover_by_eighth_balls
    {d : ℕ} [NeZero d] :
    ∃ t : Finset (EuclideanSpace ℝ (Fin d)),
      t.card ≤ 17 ^ d ∧
      Metric.closedBall (0 : EuclideanSpace ℝ (Fin d)) (1 / 2 : ℝ) ⊆
        ⋃ c ∈ t, Metric.ball c (1 / 8 : ℝ) ∧
      (∀ c ∈ t,
        Metric.closedBall c (1 / 4 : ℝ) ⊆
          Metric.ball (0 : EuclideanSpace ℝ (Fin d)) 1) := by sorry
theorem ae_on_set_of_ae_on_finite_cover
    {α : Type*} [MeasurableSpace α] {μ : Measure α}
    {ι : Type*} {t : Finset ι}
    {s : Set α} {U : ι → Set α} {P : α → Prop}
    (hcover : s ⊆ ⋃ i ∈ t, U i)
    (hlocal : ∀ i ∈ t, ∀ᵐ x ∂ μ.restrict (U i), P x) :
    ∀ᵐ x ∂ μ.restrict s, P x := by
  have h_union : ∀ᵐ x ∂ μ.restrict (⋃ i ∈ t, U i), P x := by
    rwa [ae_restrict_biUnion_finset_iff]
  exact ae_mono (Measure.restrict_mono hcover le_rfl) h_union
theorem lintegralOn_le_sum_lintegralOn_of_finite_cover
    {α : Type*} [MeasurableSpace α] {μ : Measure α}
    {ι : Type*} {t : Finset ι}
    {s : Set α} {U : ι → Set α} {f : α → ℝ≥0∞}
    (hcover : s ⊆ ⋃ i ∈ t, U i) :
    (∫⁻ x in s, f x ∂ μ) ≤
      Finset.sum t (fun i => ∫⁻ x in U i, f x ∂ μ) := by
  calc ∫⁻ x in s, f x ∂μ
      ≤ ∫⁻ x in ⋃ i ∈ t, U i, f x ∂μ :=
        lintegral_mono' (Measure.restrict_mono hcover le_rfl) le_rfl
    _ ≤ Finset.sum t (fun i => ∫⁻ x in U i, f x ∂μ) :=
        lintegral_biUnion_finset_le_sum t U f

end DeGiorgi

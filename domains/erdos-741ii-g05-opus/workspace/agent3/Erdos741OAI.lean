import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  -- KEY (combinatorial core): an additive basis `A` of order 2 with an
  -- essential-hub structure.  Three obligations:
  --   (1) basis:  every n ≥ 4 is a + b with a, b ∈ A;
  --   (2) hubs live in A:  hub k ∈ A;
  --   (3) exclusive windows:  for every gap-bound C there is a hub k and an
  --       interval [x, x+C] every element of which is representable in A ONLY
  --       through hub k (every representation a+b uses hub k as one summand).
  -- Given (1)-(3) the rigidity argument below is complete and machine-checked:
  -- any 2-colouring puts hub k in exactly one part (disjointness), so the OTHER
  -- part's sumset misses the entire length-(C+1) window, defeating syndeticity.
  --
  -- Construction strategy (an explicit indecomposable basis):
  --   place, per scale, a "hub" h with an isolation gap (h/2, h+p_max] ∩ A = ∅
  --   and a partner interval P ⊆ A with max P ≤ h/2; then m ∈ h + P forces the
  --   h-summand because any a,b ≤ h/2 give a+b ≤ h < m while a ∈ (h, m] is
  --   excluded by isolation.  The windows h + P then have unbounded length.
  --   The remaining work is to tile A+A = [4,∞) (basis) consistently with the
  --   isolation gaps — the classical Erdős indecomposable-basis construction.
  have key : ∃ (A : Set ℕ) (hub : ℕ → ℕ),
      (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
      (∀ k : ℕ, hub k ∈ A) ∧
      (∀ C : ℕ, ∃ (k : ℕ) (x : ℕ), ∀ m : ℕ, x ≤ m → m ≤ x + C →
          ∀ a ∈ A, ∀ b ∈ A, a + b = m → a = hub k ∨ b = hub k) := by
    sorry
  obtain ⟨A, hub, hbasis, hhub, holes⟩ := key
  refine ⟨A, hbasis, ?_⟩
  intro A₁ A₂ h1 h2 hcov hdisj hsyn
  obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := hsyn
  -- pick a window longer than both syndetic constants, inside some hole k
  obtain ⟨k, x, hx⟩ := holes (max C₁ C₂)
  -- hub k lives in A, hence in A₁ or A₂
  have hmem : hub k ∈ A₁ ∨ hub k ∈ A₂ := hcov _ (hhub k)
  -- disjointness: hub k cannot be in both
  have hnotboth : ¬ (hub k ∈ A₁ ∧ hub k ∈ A₂) := by
    intro ⟨ha, hb⟩
    have : hub k ∈ A₁ ∩ A₂ := ⟨ha, hb⟩
    rw [hdisj] at this
    exact this
  rcases hmem with hmem | hmem
  · -- hub k ∈ A₁, so hub k ∉ A₂; A₂+A₂ must hit window but every rep needs hub k
    have hnot2 : hub k ∉ A₂ := fun h => hnotboth ⟨hmem, h⟩
    obtain ⟨m, hmA, hmIcc⟩ := hC₂ x
    rw [Set.mem_Icc] at hmIcc
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmA
    have ha' : a ∈ A := h2 ha
    have hb' : b ∈ A := h2 hb
    have hle : m ≤ x + max C₁ C₂ := le_trans hmIcc.2 (by omega)
    have := hx m hmIcc.1 hle a ha' b hb' hab
    rcases this with rfl | rfl
    · exact hnot2 ha
    · exact hnot2 hb
  · -- symmetric: hub k ∈ A₂, A₁+A₁ must hit window
    have hnot1 : hub k ∉ A₁ := fun h => hnotboth ⟨h, hmem⟩
    obtain ⟨m, hmA, hmIcc⟩ := hC₁ x
    rw [Set.mem_Icc] at hmIcc
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmA
    have ha' : a ∈ A := h1 ha
    have hb' : b ∈ A := h1 hb
    have hle : m ≤ x + max C₁ C₂ := le_trans hmIcc.2 (by omega)
    have := hx m hmIcc.1 hle a ha' b hb' hab
    rcases this with rfl | rfl
    · exact hnot1 ha
    · exact hnot1 hb

end Erdos741OAI

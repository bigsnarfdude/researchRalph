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
  refine ⟨{n : ℕ | 2 ≤ n}, ?_, ?_⟩
  · intro n hn
    refine ⟨2, ?_, n - 2, ?_, ?_⟩
    · simp only [Set.mem_setOf_eq]
    · simp only [Set.mem_setOf_eq]; omega
    · omega
  · intro A₁ A₂ h1 h2 hcov hdisj
    -- NOTE: The witness {n | 2 ≤ n} settles PART 1 (basis of order 2) but it does
    -- NOT satisfy PART 2: the even/odd coloring A₁ = evens, A₂ = odds gives
    -- (evens)+(evens) ⊆ evens and (odds)+(odds) ⊆ evens, both syndetic. So a real
    -- proof needs a *different*, recursively-structured A. Sketch of the obstruction
    -- chain (documented for the record; the construction below is NOT yet formalized):
    --   * A basis must cover odd targets n = even+odd, forcing both parities present
    --     and fairly dense ⇒ parity coloring makes both self-sums syndetic.
    --   * Sparse-odds / dense-evens defeats the parity coloring, but the adversary
    --     then splits the EVEN part by residue mod 4 (both halves have syndetic
    --     self-sums) and dumps the odds into one half ⇒ both parts syndetic again.
    --   * Hence the even part must itself be un-2-colorable — the SAME problem one
    --     scale down. The required A is self-similar/fractal (a "bottleneck-interval"
    --     basis). Formalizing that construction + the universal-coloring argument is
    --     the open hard content of Erdős #741(ii) and is not completed here.
    sorry

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- The construction: union of blocks `[4^k, 2·4^k]`. -/
def A : Set ℕ := ⋃ k : ℕ, Set.Icc (4 ^ k) (2 * 4 ^ k)

lemma mem_A {x k : ℕ} (h1 : 4 ^ k ≤ x) (h2 : x ≤ 2 * 4 ^ k) : x ∈ A := by
  rw [A, Set.mem_iUnion]
  exact ⟨k, by rw [Set.mem_Icc]; exact ⟨h1, h2⟩⟩

lemma part1 : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n := by
  intro n hn
  set k := Nat.log 4 n with hk
  have hp1 : 4 ^ k ≤ n := Nat.pow_log_le_self 4 (by omega)
  have hp2 : n < 4 ^ (k + 1) := Nat.lt_pow_succ_log_self (by norm_num) n
  have hpow : 4 ^ (k + 1) = 4 * 4 ^ k := by rw [pow_succ]; ring
  rw [hpow] at hp2
  have h1A : (1 : ℕ) ∈ A := mem_A (k := 0) (by norm_num) (by norm_num)
  by_cases hA : n ≤ 2 * 4 ^ k
  · by_cases hnp : n = 4 ^ k
    · -- n = 4^k, need k ≥ 1
      have hk1 : 1 ≤ k := by
        rcases Nat.eq_zero_or_pos k with h | h
        · exfalso; rw [h] at hnp; norm_num at hnp; omega
        · exact h
      have hrec : 4 ^ k = 4 * 4 ^ (k - 1) := by
        conv_lhs => rw [show k = (k - 1) + 1 by omega]
        rw [pow_succ']
      have hb : 2 * 4 ^ (k - 1) ∈ A :=
        mem_A (k := k - 1) (by omega) (le_refl _)
      exact ⟨2 * 4 ^ (k - 1), hb, 2 * 4 ^ (k - 1), hb, by omega⟩
    · -- p < n ≤ 2p
      have hgt : 4 ^ k < n := lt_of_le_of_ne hp1 (fun h => hnp h.symm)
      have hbA : n - 1 ∈ A := mem_A (k := k) (by omega) (by omega)
      exact ⟨1, h1A, n - 1, hbA, by omega⟩
  · -- 2p < n < 4p
    push_neg at hA
    by_cases hB : n ≤ 3 * 4 ^ k
    · have hpA : 4 ^ k ∈ A := mem_A (k := k) (le_refl _) (by omega)
      have hbA : n - 4 ^ k ∈ A := mem_A (k := k) (by omega) (by omega)
      exact ⟨4 ^ k, hpA, n - 4 ^ k, hbA, by omega⟩
    · push_neg at hB
      have h2pA : 2 * 4 ^ k ∈ A := mem_A (k := k) (by omega) (le_refl _)
      have hbA : n - 2 * 4 ^ k ∈ A := mem_A (k := k) (by omega) (by omega)
      exact ⟨2 * 4 ^ k, h2pA, n - 2 * 4 ^ k, hbA, by omega⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  -- WARNING: `A = ⋃ [4^k, 2·4^k]` satisfies part 1 (proven above) but FAILS part 2.
  -- Counterexample coloring: split A by global parity. Then A₁+A₁ (even+even) and
  -- A₂+A₂ (odd+odd, bridged by 1 ∈ A) are BOTH cofinite sets of evens, hence both
  -- syndetic. So no proof of part 2 exists for this A. The correct construction must
  -- be a *thin* basis (|A ∩ [1,N]| ~ √N) so parity colorings cannot keep both
  -- sumsets syndetic. That construction (Erdős #741 ii) is not yet formalized.
  refine ⟨A, part1, ?_⟩
  sorry

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-
  STATUS: UNRESOLVED. The theorem is TRUE (a known Erdős result) but I could not
  formalize a valid construction cold this session. See MISTAKES.md / LEARNINGS.md
  for the 6 candidates tested and the structural obstructions (parity attack,
  AP attack, basis/rigidity tension) that defeat every elementary construction.

  Left below: candidate 1 (dyadic interval blocks). Its BASIS half is fully
  proven. Its irreducibility (condition 2) is FALSE (parity attack), so the
  open gap in the main theorem is NOT fillable for this A — it marks the
  unsolved hard direction, not a near-miss.
-/
def A : Set ℕ := {0} ∪ {n | ∃ k : ℕ, 4 ^ k ≤ n ∧ n ≤ 2 * 4 ^ k}

theorem basis_part : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n := by
  intro n hn
  set k := Nat.log 4 n with hk
  have hL : 4 ^ k ≤ n := Nat.pow_log_le_self 4 (by omega)
  have hU : n < 4 ^ (k + 1) := Nat.lt_pow_succ_log_self (by norm_num) n
  have hpow : 4 ^ (k + 1) = 4 * 4 ^ k := by ring
  set L := 4 ^ k with hLdef
  have hLpos : 1 ≤ L := Nat.one_le_pow _ _ (by norm_num)
  have h0A : (0 : ℕ) ∈ A := Or.inl rfl
  have memB : ∀ m : ℕ, L ≤ m → m ≤ 2 * L → m ∈ A := fun m h1 h2 => Or.inr ⟨k, h1, h2⟩
  have hn4 : n < 4 * L := by rw [hpow] at hU; omega
  by_cases hc : n ≤ 2 * L
  · exact ⟨n, memB n hL hc, 0, h0A, by omega⟩
  · push_neg at hc
    by_cases hc2 : n ≤ 3 * L
    · exact ⟨L, memB L (le_refl _) (by omega), n - L, memB (n - L) (by omega) (by omega), by omega⟩
    · push_neg at hc2
      exact ⟨2 * L, memB (2 * L) (by omega) (le_refl _), n - 2 * L, memB (n - 2 * L) (by omega) (by omega), by omega⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨A, basis_part, ?_⟩
  sorry

end Erdos741OAI

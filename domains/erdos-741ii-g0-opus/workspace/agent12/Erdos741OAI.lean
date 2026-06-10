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
  -- Candidate 6: A = E ∪ O, base-4 digit perfect basis (sparse, no intervals -> survives
  -- parity attack). E = base-4 digits ≤1, O = base-4 digits ∈{0,2}. n = e+f digit-wise.
  -- This is the correct SOLUTION SHAPE. cond1 = digit decomposition; cond2 = hard (all colorings).
  refine ⟨{n | (∃ L : List ℕ, (∀ d ∈ L, d ≤ 1) ∧ n = Nat.ofDigits 4 L) ∨
               (∃ L : List ℕ, (∀ d ∈ L, d = 0 ∨ d = 2) ∧ n = Nat.ofDigits 4 L)}, ?_, ?_⟩
  · intro n hn
    refine ⟨Nat.ofDigits 4 ((Nat.digits 4 n).map (· % 2)),
            Or.inl ⟨(Nat.digits 4 n).map (· % 2), ?_, rfl⟩,
            Nat.ofDigits 4 ((Nat.digits 4 n).map (fun d => 2 * (d / 2))),
            Or.inr ⟨(Nat.digits 4 n).map (fun d => 2 * (d / 2)), ?_, rfl⟩, ?_⟩
    · intro d hd; simp only [List.mem_map] at hd; obtain ⟨x, _, rfl⟩ := hd; omega
    · intro d hd; simp only [List.mem_map] at hd; obtain ⟨x, hx, rfl⟩ := hd
      have : x < 4 := Nat.digits_lt_base (by norm_num) hx
      omega
    · have key : ∀ L : List ℕ,
        Nat.ofDigits 4 (L.map (· % 2)) + Nat.ofDigits 4 (L.map (fun d => 2 * (d / 2)))
          = Nat.ofDigits 4 L := by
        intro L
        induction L with
        | nil => simp [Nat.ofDigits]
        | cons a l ih =>
          simp only [List.map_cons, Nat.ofDigits_cons]
          omega
      rw [key, Nat.ofDigits_digits]
  · intro A₁ A₂ h1 h2 hcov hdisj
    -- cond2 (FRAGILITY): for A = E∪O (base-4 perfect basis), NO 2-coloring makes both
    -- A₁+A₁ and A₂+A₂ syndetic. This is a genuine research-level combinatorial theorem.
    -- Verified facts about this A (see MISTAKES.md): it is a valid order-2 basis (cond1
    -- proven above); it has unbounded gaps (A ∩ [3·4^k, 4·4^k) = ∅, since those need a
    -- base-4 digit 3 which neither E nor O allows); it defeats the parity/mod-m attacks
    -- (its odd elements are sparse, so odd-self-sum is non-syndetic). Proving fragility for
    -- ALL partitions requires the full Erdős-741(ii) argument — NOT formalizable cold in
    -- this budget. Left as an honest sorry; the main theorem is NOT claimed proven.
    sorry

end Erdos741OAI

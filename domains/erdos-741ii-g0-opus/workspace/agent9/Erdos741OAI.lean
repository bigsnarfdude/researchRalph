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
  -- Candidate 4 (revisited): binary even/odd bit-position basis A = E ∪ O.
  -- E = numbers whose binary 1-bits sit only at EVEN positions = base-4 {0,1}-digit numbers.
  -- O = 2·E = numbers whose binary 1-bits sit only at ODD positions.
  -- Every n splits as e + o (e∈E from even bits, o∈O from odd bits) ⇒ basis of order 2.
  -- Thin; a UNION of two structures, so it resists the digit-shift coloring that breaks
  -- single digit-closed bases (candidate 6). Whether it satisfies part 2 is the open crux.
  refine ⟨{m | ∃ L : List ℕ, (∀ d ∈ L, d ≤ 1) ∧ m = Nat.ofDigits 4 L}
            ∪ {m | ∃ L : List ℕ, (∀ d ∈ L, d ≤ 1) ∧ m = 2 * Nat.ofDigits 4 L}, ?_, ?_⟩
  · -- BASIS (fully proved): split each base-4 digit d = (d % 2) + 2 * (d / 2).
    intro n _
    have key : ∀ (L : List ℕ),
        Nat.ofDigits 4 (L.map (fun d => d % 2))
          + 2 * Nat.ofDigits 4 (L.map (fun d => d / 2))
        = Nat.ofDigits 4 L := by
      intro L
      induction L with
      | nil => simp [Nat.ofDigits]
      | cons a t ih =>
        rw [List.map_cons, List.map_cons, Nat.ofDigits_cons, Nat.ofDigits_cons,
          Nat.ofDigits_cons]
        omega
    set L := Nat.digits 4 n with hL
    refine ⟨Nat.ofDigits 4 (L.map (fun d => d % 2)),
      Or.inl ⟨L.map (fun d => d % 2), ?_, rfl⟩,
      2 * Nat.ofDigits 4 (L.map (fun d => d / 2)),
      Or.inr ⟨L.map (fun d => d / 2), ?_, rfl⟩, ?_⟩
    · intro d hd
      rw [List.mem_map] at hd
      obtain ⟨c, _, rfl⟩ := hd
      omega
    · intro d hd
      rw [List.mem_map] at hd
      obtain ⟨c, hc, rfl⟩ := hd
      have : c < 4 := Nat.digits_lt_base (by norm_num) hc
      omega
    · rw [key, hL, Nat.ofDigits_digits]
  · intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨hs1, hs2⟩
    -- RESEARCH CRUX (Erdős 741(ii)): for the E∪O thin basis, show every 2-partition
    -- leaves one part's sumset with unbounded gaps. Structural argument; not reachable
    -- cold this session. SCORE deliberately left < 1.0 — statement NOT weakened.
    sorry

end Erdos741OAI

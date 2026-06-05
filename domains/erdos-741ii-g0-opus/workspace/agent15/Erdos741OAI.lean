import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Thin basis of order 2 (≈√n density): E = base-4 digits ≤ 1, and A = E ∪ 2E.
    Every n = e + 2f with e,f ∈ E (split each base-4 digit d = (d%2) + 2*(d/2)). -/
def E : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}

def A : Set ℕ := E ∪ {n | ∃ m, m ∈ E ∧ n = 2 * m}

lemma E_mem_step (c s : ℕ) (hc : c ∈ E) (hs : s ≤ 1) : 4 * c + s ∈ E := by
  intro d hd
  rcases Nat.eq_zero_or_pos (4 * c + s) with h0 | h0
  · rw [h0] at hd; simp at hd
  · rw [Nat.digits_def' (by norm_num : 2 ≤ 4) h0] at hd
    have hmod : (4 * c + s) % 4 = s := by omega
    have hdiv : (4 * c + s) / 4 = c := by omega
    rw [hmod, hdiv] at hd
    rcases List.mem_cons.mp hd with h | h
    · subst h; exact hs
    · exact hc d h

lemma E_zero : (0 : ℕ) ∈ E := by intro d hd; simp at hd

/-- Core: every n decomposes as e + 2·f with e, f ∈ E. -/
lemma E_decomp (n : ℕ) : ∃ e ∈ E, ∃ f ∈ E, e + 2 * f = n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn; exact ⟨0, E_zero, 0, E_zero, rfl⟩
    · have hqlt : n / 4 < n := Nat.div_lt_self hn (by norm_num)
      obtain ⟨e', he', f', hf', hef'⟩ := ih (n / 4) hqlt
      refine ⟨4 * e' + (n % 4) % 2, E_mem_step _ _ he' (by omega),
              4 * f' + (n % 4) / 2, E_mem_step _ _ hf' (by omega), ?_⟩
      have hdm : 4 * (n / 4) + n % 4 = n := Nat.div_add_mod n 4
      omega

lemma A_basis (n : ℕ) : ∃ a ∈ A, ∃ b ∈ A, a + b = n := by
  obtain ⟨e, he, f, hf, hef⟩ := E_decomp n
  exact ⟨e, Or.inl he, 2 * f, Or.inr ⟨f, hf, rfl⟩, hef⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨A, fun n _ => A_basis n, ?_⟩
  -- HARD CORE (open in this session): for the thin basis A = E ∪ 2E, every 2-coloring
  -- leaves one colour's sumset non-syndetic. Research-level (~280-line) argument.
  sorry

end Erdos741OAI

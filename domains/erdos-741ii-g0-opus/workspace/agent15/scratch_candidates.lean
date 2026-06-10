import Mathlib
open Set
open scoped Pointwise Classical
namespace ScratchCand

-- Candidate 3: A = ℕ (univ). Basis trivial. (Partition property is FALSE: parity split.)
example : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ (univ : Set ℕ), ∃ b ∈ (univ : Set ℕ), a + b = n := by
  intro n _; exact ⟨0, trivial, n, trivial, by ring⟩

-- Candidate 4: A = evens ∪ {1}. Basis holds. (Partition FALSE: mod-4 split.)
def A4 : Set ℕ := {n | Even n} ∪ {1}
example : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ A4, ∃ b ∈ A4, a + b = n := by
  intro n _
  rcases Nat.even_or_odd n with he | ho
  · exact ⟨0, Or.inl ⟨0, rfl⟩, n, Or.inl he, by ring⟩
  · refine ⟨1, Or.inr rfl, n - 1, Or.inl ?_, by omega⟩
    rcases ho with ⟨k, hk⟩; exact ⟨k, by omega⟩

-- Candidate 2: base-3 digit set {n | base-3 digits ≤ 1}. Basis holds (digit split).
def A2 : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
lemma A2_step (c s : ℕ) (hc : c ∈ A2) (hs : s ≤ 1) : 3 * c + s ∈ A2 := by
  intro d hd
  rcases Nat.eq_zero_or_pos (3 * c + s) with h0 | h0
  · rw [h0] at hd; simp at hd
  · rw [Nat.digits_def' (by norm_num : 2 ≤ 3) h0] at hd
    have h1 : (3 * c + s) % 3 = s := by omega
    have h2 : (3 * c + s) / 3 = c := by omega
    rw [h1, h2] at hd
    rcases List.mem_cons.mp hd with h | h
    · subst h; exact hs
    · exact hc d h
example : ∀ n : ℕ, ∃ a ∈ A2, ∃ b ∈ A2, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn; exact ⟨0, by intro d hd; simp at hd, 0, by intro d hd; simp at hd, rfl⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n/3) (Nat.div_lt_self hn (by norm_num))
      refine ⟨3*a' + (n%3+1)/2, A2_step _ _ ha' (by omega),
              3*b' + (n%3)/2, A2_step _ _ hb' (by omega), ?_⟩
      have := Nat.div_add_mod n 3; omega

-- Candidate 6: base-7 digit set {n | base-7 digits ≤ 3}. Basis holds (digit split).
def A6 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 3}
lemma A6_step (c s : ℕ) (hc : c ∈ A6) (hs : s ≤ 3) : 7 * c + s ∈ A6 := by
  intro d hd
  rcases Nat.eq_zero_or_pos (7 * c + s) with h0 | h0
  · rw [h0] at hd; simp at hd
  · rw [Nat.digits_def' (by norm_num : 2 ≤ 7) h0] at hd
    have h1 : (7 * c + s) % 7 = s := by omega
    have h2 : (7 * c + s) / 7 = c := by omega
    rw [h1, h2] at hd
    rcases List.mem_cons.mp hd with h | h
    · subst h; exact hs
    · exact hc d h
example : ∀ n : ℕ, ∃ a ∈ A6, ∃ b ∈ A6, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn; exact ⟨0, by intro d hd; simp at hd, 0, by intro d hd; simp at hd, rfl⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n/7) (Nat.div_lt_self hn (by norm_num))
      refine ⟨7*a' + (n%7+1)/2, A6_step _ _ ha' (by omega),
              7*b' + (n%7)/2, A6_step _ _ hb' (by omega), ?_⟩
      have := Nat.div_add_mod n 7; omega

end ScratchCand

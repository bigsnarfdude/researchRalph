import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Membership in the base-3 "digits ≤ 1" set. -/
def A3 : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}

theorem zero_mem_A3 : (0 : ℕ) ∈ A3 := by
  simp [A3]

/-- If a' ∈ A3 and r ≤ 1 then 3*a' + r ∈ A3. -/
theorem step_mem_A3 {a' r : ℕ} (ha' : a' ∈ A3) (hr : r ≤ 1) : 3 * a' + r ∈ A3 := by
  rcases Nat.eq_zero_or_pos (3 * a' + r) with h0 | hpos
  · rw [h0]; exact zero_mem_A3
  · intro d hd
    rw [Nat.digits_def' (by norm_num) hpos] at hd
    have hmod : (3 * a' + r) % 3 = r := by omega
    have hdiv : (3 * a' + r) / 3 = a' := by omega
    rw [hmod, hdiv] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact ha' d h

/-- Basis: every n is a sum of two elements of A3. -/
theorem A3_basis : ∀ n : ℕ, ∃ a ∈ A3, ∃ b ∈ A3, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with h0 | hpos
    · exact ⟨0, zero_mem_A3, 0, zero_mem_A3, by omega⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n / 3) (by omega)
      set r := n % 3 with hrdef
      have hr3 : r < 3 := Nat.mod_lt _ (by norm_num)
      -- split r ∈ {0,1,2} into contributions ra, rb ≤ 1 with ra+rb = r
      have hsplit : ∃ ra rb, ra ≤ 1 ∧ rb ≤ 1 ∧ ra + rb = r := by
        interval_cases r
        · exact ⟨0, 0, by norm_num⟩
        · exact ⟨1, 0, by norm_num⟩
        · exact ⟨1, 1, by norm_num⟩
      obtain ⟨ra, rb, hra, hrb, hrab⟩ := hsplit
      refine ⟨3 * a' + ra, step_mem_A3 ha' hra, 3 * b' + rb, step_mem_A3 hb' hrb, ?_⟩
      have : n = 3 * (n / 3) + n % 3 := (Nat.div_add_mod n 3).symm
      omega

/-- Base-5 "digits ≤ 2" set (canonical Erdős construction). -/
def A5 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 2}

theorem zero_mem_A5 : (0 : ℕ) ∈ A5 := by simp [A5]

theorem step_mem_A5 {a' r : ℕ} (ha' : a' ∈ A5) (hr : r ≤ 2) : 5 * a' + r ∈ A5 := by
  rcases Nat.eq_zero_or_pos (5 * a' + r) with h0 | hpos
  · rw [h0]; exact zero_mem_A5
  · intro d hd
    rw [Nat.digits_def' (by norm_num) hpos] at hd
    have hmod : (5 * a' + r) % 5 = r := by omega
    have hdiv : (5 * a' + r) / 5 = a' := by omega
    rw [hmod, hdiv] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact ha' d h

theorem A5_basis : ∀ n : ℕ, ∃ a ∈ A5, ∃ b ∈ A5, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with h0 | hpos
    · exact ⟨0, zero_mem_A5, 0, zero_mem_A5, by omega⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n / 5) (by omega)
      set r := n % 5 with hrdef
      have hr5 : r < 5 := Nat.mod_lt _ (by norm_num)
      have hsplit : ∃ ra rb, ra ≤ 2 ∧ rb ≤ 2 ∧ ra + rb = r := by
        interval_cases r
        · exact ⟨0, 0, by norm_num⟩
        · exact ⟨1, 0, by norm_num⟩
        · exact ⟨2, 0, by norm_num⟩
        · exact ⟨2, 1, by norm_num⟩
        · exact ⟨2, 2, by norm_num⟩
      obtain ⟨ra, rb, hra, hrb, hrab⟩ := hsplit
      refine ⟨5 * a' + ra, step_mem_A5 ha' hra, 5 * b' + rb, step_mem_A5 hb' hrb, ?_⟩
      have : n = 5 * (n / 5) + n % 5 := (Nat.div_add_mod n 5).symm
      omega

/-- Base-9 "digits ≤ 4" set. -/
def A9 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 4}

theorem zero_mem_A9 : (0 : ℕ) ∈ A9 := by simp [A9]

theorem step_mem_A9 {a' r : ℕ} (ha' : a' ∈ A9) (hr : r ≤ 4) : 9 * a' + r ∈ A9 := by
  rcases Nat.eq_zero_or_pos (9 * a' + r) with h0 | hpos
  · rw [h0]; exact zero_mem_A9
  · intro d hd
    rw [Nat.digits_def' (by norm_num) hpos] at hd
    have hmod : (9 * a' + r) % 9 = r := by omega
    have hdiv : (9 * a' + r) / 9 = a' := by omega
    rw [hmod, hdiv] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact ha' d h

theorem A9_basis : ∀ n : ℕ, ∃ a ∈ A9, ∃ b ∈ A9, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with h0 | hpos
    · exact ⟨0, zero_mem_A9, 0, zero_mem_A9, by omega⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n / 9) (by omega)
      set r := n % 9 with hrdef
      have hr9 : r < 9 := Nat.mod_lt _ (by norm_num)
      have hsplit : ∃ ra rb, ra ≤ 4 ∧ rb ≤ 4 ∧ ra + rb = r := by
        interval_cases r
        · exact ⟨0, 0, by norm_num⟩
        · exact ⟨1, 0, by norm_num⟩
        · exact ⟨2, 0, by norm_num⟩
        · exact ⟨3, 0, by norm_num⟩
        · exact ⟨4, 0, by norm_num⟩
        · exact ⟨4, 1, by norm_num⟩
        · exact ⟨4, 2, by norm_num⟩
        · exact ⟨4, 3, by norm_num⟩
        · exact ⟨4, 4, by norm_num⟩
      obtain ⟨ra, rb, hra, hrb, hrab⟩ := hsplit
      refine ⟨9 * a' + ra, step_mem_A9 ha' hra, 9 * b' + rb, step_mem_A9 hb' hrb, ?_⟩
      have : n = 9 * (n / 9) + n % 9 := (Nat.div_add_mod n 9).symm
      omega

/-- Base-4 "digits ≤ 2" set. -/
def A4 : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 2}

theorem zero_mem_A4 : (0 : ℕ) ∈ A4 := by simp [A4]

theorem step_mem_A4 {a' r : ℕ} (ha' : a' ∈ A4) (hr : r ≤ 2) : 4 * a' + r ∈ A4 := by
  rcases Nat.eq_zero_or_pos (4 * a' + r) with h0 | hpos
  · rw [h0]; exact zero_mem_A4
  · intro d hd
    rw [Nat.digits_def' (by norm_num) hpos] at hd
    have hmod : (4 * a' + r) % 4 = r := by omega
    have hdiv : (4 * a' + r) / 4 = a' := by omega
    rw [hmod, hdiv] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact ha' d h

theorem A4_basis : ∀ n : ℕ, ∃ a ∈ A4, ∃ b ∈ A4, a + b = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with h0 | hpos
    · exact ⟨0, zero_mem_A4, 0, zero_mem_A4, by omega⟩
    · obtain ⟨a', ha', b', hb', hab'⟩ := ih (n / 4) (by omega)
      set r := n % 4 with hrdef
      have hr4 : r < 4 := Nat.mod_lt _ (by norm_num)
      have hsplit : ∃ ra rb, ra ≤ 2 ∧ rb ≤ 2 ∧ ra + rb = r := by
        interval_cases r
        · exact ⟨0, 0, by norm_num⟩
        · exact ⟨1, 0, by norm_num⟩
        · exact ⟨2, 0, by norm_num⟩
        · exact ⟨2, 1, by norm_num⟩
      obtain ⟨ra, rb, hra, hrb, hrab⟩ := hsplit
      refine ⟨4 * a' + ra, step_mem_A4 ha' hra, 4 * b' + rb, step_mem_A4 hb' hrb, ?_⟩
      have : n = 4 * (n / 4) + n % 4 := (Nat.div_add_mod n 4).symm
      omega

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  -- Final: A = A5 (base-5 digits ≤ 2, the canonical Erdős construction).
  -- BASIS direction is fully proved below (A5_basis).
  -- PARTITION direction remains open (sorry): proving that for EVERY 2-colouring
  -- of A5 one part has a non-syndetic sumset requires an adaptive, per-colouring
  -- choice of the gap location.  The "rigid target" funnel (numbers all of whose
  -- base-5 digits are 0 or 4 have a unique representation a=b) shows certain
  -- integers land in only one part's sumset, but a single forced element cannot
  -- carry a whole interval of length →∞, so this does not yield unbounded gaps
  -- for arbitrary colourings.  No short cold-start proof was found.
  refine ⟨A5, ?_, ?_⟩
  · intro n _; exact A5_basis n
  · sorry

end Erdos741OAI

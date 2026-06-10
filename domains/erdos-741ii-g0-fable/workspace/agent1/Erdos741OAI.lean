import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- The witness set: 0 together with, at every scale `5^k`, the interval
`[5^k, 2*5^k]` and the isolated point `3*5^k`. -/
def A : Set ℕ :=
  {x | x = 0 ∨ ∃ k : ℕ, (5 ^ k ≤ x ∧ x ≤ 2 * 5 ^ k) ∨ x = 3 * 5 ^ k}

lemma mem_A_iff (x : ℕ) :
    x ∈ A ↔ x = 0 ∨ ∃ k : ℕ, (5 ^ k ≤ x ∧ x ≤ 2 * 5 ^ k) ∨ x = 3 * 5 ^ k :=
  Iff.rfl

lemma zero_mem_A : (0 : ℕ) ∈ A := (mem_A_iff 0).mpr (Or.inl rfl)

lemma interval_mem_A {k x : ℕ} (h1 : 5 ^ k ≤ x) (h2 : x ≤ 2 * 5 ^ k) : x ∈ A :=
  (mem_A_iff x).mpr (Or.inr ⟨k, Or.inl ⟨h1, h2⟩⟩)

lemma three_mem_A (k : ℕ) : (3 * 5 ^ k : ℕ) ∈ A :=
  (mem_A_iff _).mpr (Or.inr ⟨k, Or.inr rfl⟩)

lemma lt_pow5 (m : ℕ) : m < 5 ^ m := by
  induction m with
  | zero => norm_num
  | succ n ih =>
    have h : 5 ^ (n + 1) = 5 ^ n * 5 := pow_succ 5 n
    omega

/-- Every element of `A` is small (`≤ 2*5^k`), the special point `3*5^k`,
or already in the next scale (`≥ 5^(k+1)`). -/
lemma classify {x : ℕ} (hx : x ∈ A) (k : ℕ) :
    x ≤ 2 * 5 ^ k ∨ x = 3 * 5 ^ k ∨ 5 ^ (k + 1) ≤ x := by
  rw [mem_A_iff] at hx
  rcases hx with rfl | ⟨j, ⟨hj1, hj2⟩ | rfl⟩
  · exact Or.inl (Nat.zero_le _)
  · by_cases hjk : j ≤ k
    · have h := Nat.pow_le_pow_right (show 0 < 5 by norm_num) hjk
      omega
    · have h := Nat.pow_le_pow_right (show 0 < 5 by norm_num)
        (show k + 1 ≤ j by omega)
      omega
  · by_cases hjk : j ≤ k
    · by_cases heq : j = k
      · subst heq
        omega
      · have h := Nat.pow_le_pow_right (show 0 < 5 by norm_num)
          (show j + 1 ≤ k by omega)
        have hpsj : 5 ^ (j + 1) = 5 ^ j * 5 := pow_succ 5 j
        omega
    · have h := Nat.pow_le_pow_right (show 0 < 5 by norm_num)
        (show k + 1 ≤ j by omega)
      omega

/-- Rigidity: any representation of `n ∈ (4*5^k, 5^(k+1))` as a sum of two
elements of `A` must use the special point `3*5^k`. -/
lemma rigidity {k n a b : ℕ} (h4 : 4 * 5 ^ k < n) (h5 : n < 5 ^ (k + 1))
    (ha : a ∈ A) (hb : b ∈ A) (hab : a + b = n) :
    a = 3 * 5 ^ k ∨ b = 3 * 5 ^ k := by
  have hps : 5 ^ (k + 1) = 5 ^ k * 5 := pow_succ 5 k
  rcases classify ha k with h1 | h1 | h1 <;>
    rcases classify hb k with h2 | h2 | h2 <;> omega

/-- `A` is an additive basis of order 2 for all `n ≥ 4`. -/
lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ A, ∃ b ∈ A, a + b = n := by
  obtain ⟨k, hk1, hk2⟩ : ∃ k, 5 ^ k ≤ n ∧ n < 5 ^ (k + 1) :=
    ⟨Nat.log 5 n, Nat.pow_log_le_self 5 (by omega),
      Nat.lt_pow_succ_log_self (by norm_num) n⟩
  have hps : 5 ^ (k + 1) = 5 ^ k * 5 := pow_succ 5 k
  by_cases h2 : n ≤ 2 * 5 ^ k
  · exact ⟨n, interval_mem_A hk1 h2, 0, zero_mem_A, by omega⟩
  by_cases h3 : n ≤ 3 * 5 ^ k
  · exact ⟨5 ^ k, interval_mem_A le_rfl (by omega),
      n - 5 ^ k, interval_mem_A (by omega) (by omega), by omega⟩
  by_cases h4 : n ≤ 4 * 5 ^ k
  · exact ⟨2 * 5 ^ k, interval_mem_A (by omega) le_rfl,
      n - 2 * 5 ^ k, interval_mem_A (by omega) (by omega), by omega⟩
  · exact ⟨3 * 5 ^ k, three_mem_A k,
      n - 3 * 5 ^ k, interval_mem_A (by omega) (by omega), by omega⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨A, basis, ?_⟩
  intro A₁ A₂ h1 h2 hcover hdisj
  rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
  have hbig := lt_pow5 (C₁ + C₂ + 2)
  have hps : 5 ^ (C₁ + C₂ + 2 + 1) = 5 ^ (C₁ + C₂ + 2) * 5 := pow_succ 5 _
  rcases hcover (3 * 5 ^ (C₁ + C₂ + 2)) (three_mem_A _) with hc | hc
  · -- the special point is in A₁; probe A₂ + A₂ inside the window
    obtain ⟨m, hm, hmI⟩ := hC₂ (4 * 5 ^ (C₁ + C₂ + 2) + 1)
    rw [Set.mem_Icc] at hmI
    rw [Set.mem_add] at hm
    obtain ⟨a, ha, b, hb, hab⟩ := hm
    have h4 : 4 * 5 ^ (C₁ + C₂ + 2) < m := by omega
    have h5 : m < 5 ^ (C₁ + C₂ + 2 + 1) := by omega
    rcases rigidity h4 h5 (h2 ha) (h2 hb) hab with rfl | rfl
    · have hmem : (3 * 5 ^ (C₁ + C₂ + 2) : ℕ) ∈ A₁ ∩ A₂ := ⟨hc, ha⟩
      rw [hdisj] at hmem
      exact hmem
    · have hmem : (3 * 5 ^ (C₁ + C₂ + 2) : ℕ) ∈ A₁ ∩ A₂ := ⟨hc, hb⟩
      rw [hdisj] at hmem
      exact hmem
  · -- the special point is in A₂; probe A₁ + A₁ inside the window
    obtain ⟨m, hm, hmI⟩ := hC₁ (4 * 5 ^ (C₁ + C₂ + 2) + 1)
    rw [Set.mem_Icc] at hmI
    rw [Set.mem_add] at hm
    obtain ⟨a, ha, b, hb, hab⟩ := hm
    have h4 : 4 * 5 ^ (C₁ + C₂ + 2) < m := by omega
    have h5 : m < 5 ^ (C₁ + C₂ + 2 + 1) := by omega
    rcases rigidity h4 h5 (h1 ha) (h1 hb) hab with rfl | rfl
    · have hmem : (3 * 5 ^ (C₁ + C₂ + 2) : ℕ) ∈ A₁ ∩ A₂ := ⟨ha, hc⟩
      rw [hdisj] at hmem
      exact hmem
    · have hmem : (3 * 5 ^ (C₁ + C₂ + 2) : ℕ) ∈ A₁ ∩ A₂ := ⟨hb, hc⟩
      rw [hdisj] at hmem
      exact hmem

end Erdos741OAI

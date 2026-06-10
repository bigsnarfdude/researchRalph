import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Witness set: 0, the intervals [5^k, 2·5^k], and the anchors 3·5^k. -/
def setA : Set ℕ := {n | n = 0 ∨ ∃ k, (5 ^ k ≤ n ∧ n ≤ 2 * 5 ^ k) ∨ n = 3 * 5 ^ k}

lemma zero_mem : (0 : ℕ) ∈ setA := Or.inl rfl

lemma interval_mem {n k : ℕ} (h1 : 5 ^ k ≤ n) (h2 : n ≤ 2 * 5 ^ k) : n ∈ setA :=
  Or.inr ⟨k, Or.inl ⟨h1, h2⟩⟩

lemma anchor_mem (k : ℕ) : 3 * 5 ^ k ∈ setA := Or.inr ⟨k, Or.inr rfl⟩

lemma lt_pow5 (n : ℕ) : n < 5 ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h : 1 ≤ 5 ^ n := Nat.one_le_pow _ _ (by norm_num)
    rw [pow_succ]
    omega

/-- Any element of A below 5^(k+1) is the k-anchor or at most 2·5^k. -/
lemma classify {x k : ℕ} (hx : x ∈ setA) (hlt : x < 5 ^ (k + 1)) :
    x = 3 * 5 ^ k ∨ x ≤ 2 * 5 ^ k := by
  rcases hx with h0 | ⟨j, hj⟩
  · right
    have : 1 ≤ 5 ^ k := Nat.one_le_pow _ _ (by norm_num)
    omega
  · by_cases hjk : j ≤ k
    · have hp : 5 ^ j ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) hjk
      rcases hj with ⟨h1, h2⟩ | h3
      · right; omega
      · by_cases heq : j = k
        · left; rw [← heq]; exact h3
        · right
          have hlt' : j + 1 ≤ k := by omega
          have hp1 : 5 ^ (j + 1) ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) hlt'
          rw [pow_succ] at hp1
          omega
    · push_neg at hjk
      have hp : 5 ^ (k + 1) ≤ 5 ^ j := Nat.pow_le_pow_right (by norm_num) hjk
      have hone : 1 ≤ 5 ^ j := Nat.one_le_pow _ _ (by norm_num)
      rcases hj with ⟨h1, h2⟩ | h3
      · omega
      · omega

/-- Rigidity: any sum landing strictly between 4·5^k and 5^(k+1) must use the anchor. -/
lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (h4 : 4 * 5 ^ k < a + b) (h5 : a + b < 5 ^ (k + 1)) :
    a = 3 * 5 ^ k ∨ b = 3 * 5 ^ k := by
  have hka := classify ha (lt_of_le_of_lt (Nat.le_add_right a b) h5)
  have hkb := classify hb (lt_of_le_of_lt (Nat.le_add_left b a) h5)
  rcases hka with h | h
  · exact Or.inl h
  · rcases hkb with h' | h'
    · exact Or.inr h'
    · exfalso; omega

lemma basis_aux : ∀ k, ∀ n : ℕ, 4 ≤ n → n ≤ 5 ^ (k + 1) →
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro k
  induction k with
  | zero =>
    intro n h4 h5
    have h2A : (2 : ℕ) ∈ setA := interval_mem (k := 0) (by norm_num) (by norm_num)
    have h3A : (3 : ℕ) ∈ setA := Or.inr ⟨0, Or.inr (by norm_num)⟩
    norm_num at h5
    interval_cases n
    · exact ⟨2, h2A, 2, h2A, rfl⟩
    · exact ⟨2, h2A, 3, h3A, rfl⟩
  | succ k ih =>
    intro n h4 hn
    by_cases hsmall : n ≤ 5 ^ (k + 1)
    · exact ih n h4 hsmall
    · push_neg at hsmall
      have hQ : (1 : ℕ) ≤ 5 ^ (k + 1) := Nat.one_le_pow _ _ (by norm_num)
      have hpow : 5 ^ (k + 2) = 5 * 5 ^ (k + 1) := by rw [pow_succ]; ring
      by_cases hb1 : n ≤ 2 * 5 ^ (k + 1)
      · exact ⟨0, zero_mem, n, interval_mem (k := k + 1) (by omega) hb1, by omega⟩
      · push_neg at hb1
        by_cases hb2 : n ≤ 4 * 5 ^ (k + 1)
        · exact ⟨n / 2, interval_mem (k := k + 1) (by omega) (by omega),
            n - n / 2, interval_mem (k := k + 1) (by omega) (by omega), by omega⟩
        · push_neg at hb2
          exact ⟨3 * 5 ^ (k + 1), anchor_mem (k + 1),
            n - 3 * 5 ^ (k + 1), interval_mem (k := k + 1) (by omega) (by omega), by omega⟩

lemma basis_lem (n : ℕ) (h4 : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have h := lt_pow5 n
  have h' : 5 ^ n ≤ 5 ^ (n + 1) := Nat.pow_le_pow_right (by norm_num) (Nat.le_succ n)
  exact basis_aux n n h4 (by omega)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, fun n h4 => basis_lem n h4, ?_⟩
  intro A₁ A₂ hsub1 hsub2 hcover hdisj
  rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
  set k := C₁ + C₂ + 2 with hkdef
  have hbig : C₁ + C₂ + 2 < 5 ^ k := lt_pow5 k
  have hpow : 5 ^ (k + 1) = 5 * 5 ^ k := by rw [pow_succ]; ring
  have hanc : 3 * 5 ^ k ∈ setA := anchor_mem k
  rcases hcover _ hanc with hin1 | hin2
  · -- anchor in A₁; syndeticity of A₂+A₂ on the window forces anchor ∈ A₂
    obtain ⟨m, hm, hmIcc⟩ := hC₂ (4 * 5 ^ k + 1)
    rw [Set.mem_Icc] at hmIcc
    rw [Set.mem_add] at hm
    obtain ⟨a, ha, b, hb, hab⟩ := hm
    have hrig := rigidity (k := k) (hsub2 ha) (hsub2 hb) (by omega) (by omega)
    have h32 : 3 * 5 ^ k ∈ A₂ := by
      rcases hrig with h | h
      · exact h ▸ ha
      · exact h ▸ hb
    have hmem : (3 * 5 ^ k) ∈ A₁ ∩ A₂ := ⟨hin1, h32⟩
    rw [hdisj] at hmem
    exact hmem
  · -- anchor in A₂; symmetric
    obtain ⟨m, hm, hmIcc⟩ := hC₁ (4 * 5 ^ k + 1)
    rw [Set.mem_Icc] at hmIcc
    rw [Set.mem_add] at hm
    obtain ⟨a, ha, b, hb, hab⟩ := hm
    have hrig := rigidity (k := k) (hsub1 ha) (hsub1 hb) (by omega) (by omega)
    have h31 : 3 * 5 ^ k ∈ A₁ := by
      rcases hrig with h | h
      · exact h ▸ ha
      · exact h ▸ hb
    have hmem : (3 * 5 ^ k) ∈ A₁ ∩ A₂ := ⟨h31, hin2⟩
    rw [hdisj] at hmem
    exact hmem

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Witness set: small seeds {0,1,2}, plus at every scale k ≥ 1 the interval
    [5^k, 2·5^k] together with the isolated pivot 3·5^k. -/
def W : Set ℕ :=
  {n | n ≤ 2 ∨ ∃ k, 1 ≤ k ∧ ((5 ^ k ≤ n ∧ n ≤ 2 * 5 ^ k) ∨ n = 3 * 5 ^ k)}

lemma mem_W_iff (n : ℕ) :
    n ∈ W ↔ n ≤ 2 ∨ ∃ k, 1 ≤ k ∧ ((5 ^ k ≤ n ∧ n ≤ 2 * 5 ^ k) ∨ n = 3 * 5 ^ k) :=
  Iff.rfl

lemma small_mem {n : ℕ} (h : n ≤ 2) : n ∈ W := (mem_W_iff n).mpr (Or.inl h)

lemma interval_mem {n k : ℕ} (hk : 1 ≤ k) (h1 : 5 ^ k ≤ n) (h2 : n ≤ 2 * 5 ^ k) :
    n ∈ W := (mem_W_iff n).mpr (Or.inr ⟨k, hk, Or.inl ⟨h1, h2⟩⟩)

lemma pivot_mem {k : ℕ} (hk : 1 ≤ k) : 3 * 5 ^ k ∈ W :=
  (mem_W_iff _).mpr (Or.inr ⟨k, hk, Or.inr rfl⟩)

lemma five_le_pow {k : ℕ} (hk : 1 ≤ k) : 5 ≤ 5 ^ k := by
  calc (5 : ℕ) = 5 ^ 1 := (pow_one 5).symm
  _ ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) hk

/-- Every element of W is either below scale k, inside the scale-k interval,
    equal to the scale-k pivot, or at least 5·5^k. -/
lemma classify (k : ℕ) (hk : 1 ≤ k) (x : ℕ) (hx : x ∈ W) :
    x < 5 ^ k ∨ (5 ^ k ≤ x ∧ x ≤ 2 * 5 ^ k) ∨ x = 3 * 5 ^ k ∨ 5 * 5 ^ k ≤ x := by
  have h5k : 5 ≤ 5 ^ k := five_le_pow hk
  rcases (mem_W_iff x).mp hx with h2 | ⟨j, hj, hcase⟩
  · left; omega
  · rcases lt_trichotomy j k with hlt | heq | hgt
    · have hpow : 5 ^ (j + 1) ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
      have hps : 5 ^ (j + 1) = 5 * 5 ^ j := by rw [pow_succ]; ring
      have hpj : 0 < 5 ^ j := pow_pos (by norm_num) j
      left; omega
    · subst heq
      rcases hcase with h | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr (Or.inl h))
    · have hpow : 5 ^ (k + 1) ≤ 5 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
      have hps : 5 ^ (k + 1) = 5 * 5 ^ k := by rw [pow_succ]; ring
      have hpj : 0 < 5 ^ j := pow_pos (by norm_num) j
      right; right; right; omega

/-- Rigidity: a sum of two elements of W landing strictly between 4·5^k and 5·5^k
    must use the pivot 3·5^k as one of its summands. -/
lemma rigidity (k : ℕ) (hk : 1 ≤ k) {a b : ℕ} (ha : a ∈ W) (hb : b ∈ W)
    (h1 : 4 * 5 ^ k < a + b) (h2 : a + b < 5 * 5 ^ k) :
    a = 3 * 5 ^ k ∨ b = 3 * 5 ^ k := by
  have ca := classify k hk a ha
  have cb := classify k hk b hb
  have h5k : 5 ≤ 5 ^ k := five_le_pow hk
  omega

/-- Every n ≥ 5 lies in some band [5^k, 5^(k+1)) with k ≥ 1. -/
lemma exists_band : ∀ n : ℕ, 5 ≤ n → ∃ k, 1 ≤ k ∧ 5 ^ k ≤ n ∧ n < 5 ^ (k + 1) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro hn
    by_cases h25 : n < 25
    · refine ⟨1, le_refl 1, ?_, ?_⟩
      · simpa using hn
      · norm_num
        omega
    · have hd : 5 ≤ n / 5 := by omega
      have hlt : n / 5 < n := Nat.div_lt_self (by omega) (by norm_num)
      obtain ⟨k, hk, hlo, hhi⟩ := ih (n / 5) hlt hd
      have hps1 : 5 ^ (k + 1) = 5 * 5 ^ k := by rw [pow_succ]; ring
      have hps2 : 5 ^ (k + 2) = 5 * 5 ^ (k + 1) := by rw [pow_succ]; ring
      exact ⟨k + 1, by omega, by omega, by omega⟩

/-- W is an additive basis of order 2 for all n ≥ 4. -/
lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ W, ∃ b ∈ W, a + b = n := by
  by_cases h4 : n = 4
  · exact ⟨2, small_mem (by norm_num), 2, small_mem (by norm_num), by omega⟩
  · obtain ⟨k, hk, hlo, hhi⟩ := exists_band n (by omega)
    have hps : 5 ^ (k + 1) = 5 * 5 ^ k := by rw [pow_succ]; ring
    have h5k : 5 ≤ 5 ^ k := five_le_pow hk
    by_cases hb1 : n ≤ 2 * 5 ^ k
    · exact ⟨0, small_mem (by norm_num), n, interval_mem hk hlo hb1, by omega⟩
    · by_cases hb2 : n ≤ 3 * 5 ^ k
      · exact ⟨5 ^ k, interval_mem hk (le_refl _) (by omega),
          n - 5 ^ k, interval_mem hk (by omega) (by omega), by omega⟩
      · by_cases hb3 : n ≤ 4 * 5 ^ k
        · exact ⟨2 * 5 ^ k, interval_mem hk (by omega) (le_refl _),
            n - 2 * 5 ^ k, interval_mem hk (by omega) (by omega), by omega⟩
        · exact ⟨3 * 5 ^ k, pivot_mem hk,
            n - 3 * 5 ^ k, interval_mem hk (by omega) (by omega), by omega⟩

/-- A class S ⊆ W that misses the pivot 3·5^k cannot have S+S syndetic with
    constant C once C + 2 ≤ 5^k: the window (4·5^k, 5·5^k) is then empty of S+S. -/
lemma gap_window (k C : ℕ) (hk : 1 ≤ k) (hC : C + 2 ≤ 5 ^ k)
    (S : Set ℕ) (hS : S ⊆ W) (hp : (3 * 5 ^ k) ∉ S)
    (hsynd : ∀ x : ℕ, ∃ m ∈ S + S, m ∈ Icc x (x + C)) : False := by
  obtain ⟨m, hm, hmI⟩ := hsynd (4 * 5 ^ k + 1)
  rw [Set.mem_Icc] at hmI
  rw [Set.mem_add] at hm
  obtain ⟨a, ha, b, hb, hab⟩ := hm
  have h1 : 4 * 5 ^ k < a + b := by omega
  have h2 : a + b < 5 * 5 ^ k := by omega
  rcases rigidity k hk (hS ha) (hS hb) h1 h2 with h | h
  · exact hp (h ▸ ha)
  · exact hp (h ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨W, fun n hn => basis n hn, ?_⟩
  intro A₁ A₂ hsub1 hsub2 hcover hdisj h
  have hs1 : ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ A₁ + A₁, m ∈ Icc x (x + C) := h.1
  have hs2 : ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ A₂ + A₂, m ∈ Icc x (x + C) := h.2
  obtain ⟨C₁, hC₁⟩ := hs1
  obtain ⟨C₂, hC₂⟩ := hs2
  set k := C₁ + C₂ + 1 with hkdef
  have hk : 1 ≤ k := by omega
  have hklt : k < 5 ^ k := Nat.lt_pow_self (by norm_num)
  have hp : 3 * 5 ^ k ∈ W := pivot_mem hk
  rcases hcover _ hp with hmem | hmem
  · have hnot : 3 * 5 ^ k ∉ A₂ := by
      intro hin
      have hempty : (3 * 5 ^ k) ∈ A₁ ∩ A₂ := ⟨hmem, hin⟩
      rw [hdisj] at hempty
      exact hempty
    exact gap_window k C₂ hk (by omega) A₂ hsub2 hnot hC₂
  · have hnot : 3 * 5 ^ k ∉ A₁ := by
      intro hin
      have hempty : (3 * 5 ^ k) ∈ A₁ ∩ A₂ := ⟨hin, hmem⟩
      rw [hdisj] at hempty
      exact hempty
    exact gap_window k C₁ hk (by omega) A₁ hsub1 hnot hC₁

end Erdos741OAI

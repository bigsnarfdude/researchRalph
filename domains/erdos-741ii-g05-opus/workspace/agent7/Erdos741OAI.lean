import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ### Construction

`Q k = 5^k`.  `setA = Icc 2 3 ∪ ⋃ k, ({4 Qk} ∪ Icc (5 Qk) (6 Qk - 1) ∪ Icc (10 Qk - 1) (15 Qk))`.

* **Basis**: by induction `Icc 4 (30 Q k) ⊆ setA + setA`, using that consecutive stages
  overlap (top band of stage `k` crossed with stage `k+1`).
* **Rigidity**: any `n ∈ [9 Qk, 10 Qk)` can only be written `a + b` with `{a,b} ∋ 4 Qk`,
  so whichever colour `4 Qk` lies in, the other colour's sumset misses the whole window
  `[9 Qk, 10 Qk)`, contradicting syndeticity once `Qk` exceeds the syndetic constant.
-/

def Q (k : ℕ) : ℕ := 5 ^ k

lemma one_le_Q (k : ℕ) : 1 ≤ Q k := by
  simp only [Q]; exact Nat.one_le_pow k 5 (by norm_num)

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ]; ring

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  simp only [Q]; exact Nat.pow_le_pow_right (by norm_num) h

lemma Q_step_le {a b : ℕ} (h : a < b) : 5 * Q a ≤ Q b := by
  have h1 : Q (a + 1) ≤ Q b := Q_mono (by omega)
  rw [Q_succ] at h1; exact h1

def stage (k : ℕ) : Set ℕ :=
  ({4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1)) ∪ Icc (10 * Q k - 1) (15 * Q k)

def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, stage k

lemma mem_setA_P (k : ℕ) : 4 * Q k ∈ setA := by
  refine Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, ?_⟩)
  exact Set.mem_union_left _ (Set.mem_union_left _ rfl)

lemma mem_setA_M (k : ℕ) {x : ℕ} (hx : x ∈ Icc (5 * Q k) (6 * Q k - 1)) : x ∈ setA := by
  refine Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, ?_⟩)
  exact Set.mem_union_left _ (Set.mem_union_right _ hx)

lemma mem_setA_T (k : ℕ) {x : ℕ} (hx : x ∈ Icc (10 * Q k - 1) (15 * Q k)) : x ∈ setA := by
  refine Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, ?_⟩)
  exact Set.mem_union_right _ hx

lemma mem_setA_iff {x : ℕ} : x ∈ setA ↔
    (2 ≤ x ∧ x ≤ 3) ∨
    ∃ j, ((x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1)) ∨
          (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j)) := by
  simp only [setA, stage, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff, Set.mem_Icc]

lemma two_le {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rw [mem_setA_iff] at hx
  obtain h | ⟨j, hj⟩ := hx
  · omega
  · have hQj : 1 ≤ Q j := one_le_Q j
    rcases hj with (h | h) | h <;> omega

lemma sum_cover {a1 b1 a2 b2 n : ℕ} (S1 : Icc a1 b1 ⊆ setA) (S2 : Icc a2 b2 ⊆ setA)
    (hab1 : a1 ≤ b1) (hab2 : a2 ≤ b2) (h1 : a1 + a2 ≤ n) (h2 : n ≤ b1 + b2) :
    n ∈ (setA + setA) := by
  rw [Set.mem_add]
  exact ⟨min b1 (n - a2), S1 (by rw [Set.mem_Icc]; omega),
         n - min b1 (n - a2), S2 (by rw [Set.mem_Icc]; omega), by omega⟩

lemma basis_cover : ∀ k, Icc 4 (30 * Q k) ⊆ (setA + setA) := by
  intro k
  induction k with
  | zero =>
    have hQ0 : Q 0 = 1 := by norm_num [Q]
    have m4 : (4 : ℕ) ∈ setA := by
      have h := mem_setA_P 0; rw [hQ0] at h; norm_num at h; exact h
    have m5 : (5 : ℕ) ∈ setA := by
      apply mem_setA_M 0; rw [Set.mem_Icc, hQ0]; omega
    have sub23 : Icc 2 3 ⊆ setA := by intro x hx; exact Set.mem_union_left _ hx
    have sub45 : Icc 4 5 ⊆ setA := by
      intro x hx; rw [Set.mem_Icc] at hx; obtain ⟨h1, h2⟩ := hx
      interval_cases x
      · exact m4
      · exact m5
    have sub_T0 : Icc 9 15 ⊆ setA := by
      intro x hx; rw [Set.mem_Icc] at hx
      apply mem_setA_T 0; rw [Set.mem_Icc, hQ0]; omega
    intro n hn; rw [Set.mem_Icc] at hn; obtain ⟨hlo, hhi⟩ := hn
    by_cases h6 : n ≤ 6
    · exact sum_cover sub23 sub23 (by omega) (by omega) (by omega) (by omega)
    by_cases h8 : n ≤ 8
    · exact sum_cover sub23 sub45 (by omega) (by omega) (by omega) (by omega)
    by_cases h10' : n ≤ 10
    · exact sum_cover sub45 sub45 (by omega) (by omega) (by omega) (by omega)
    by_cases h18 : n ≤ 18
    · exact sum_cover sub23 sub_T0 (by omega) (by omega) (by omega) (by omega)
    · exact sum_cover sub_T0 sub_T0 (by omega) (by omega) (by omega) (by omega)
  | succ k ih =>
    have hq : 1 ≤ Q k := one_le_Q k
    have hq1 : 1 ≤ Q (k + 1) := one_le_Q (k + 1)
    have hsucc : Q (k + 1) = 5 * Q k := Q_succ k
    have sPk : Icc (4 * Q k) (4 * Q k) ⊆ setA := by
      intro x hx; rw [Set.mem_Icc] at hx
      have hxe : x = 4 * Q k := le_antisymm hx.2 hx.1
      rw [hxe]; exact mem_setA_P k
    have sTk : Icc (10 * Q k - 1) (15 * Q k) ⊆ setA := fun x hx => mem_setA_T k hx
    have sPk1 : Icc (4 * Q (k + 1)) (4 * Q (k + 1)) ⊆ setA := by
      intro x hx; rw [Set.mem_Icc] at hx
      have hxe : x = 4 * Q (k + 1) := le_antisymm hx.2 hx.1
      rw [hxe]; exact mem_setA_P (k + 1)
    have sMk1 : Icc (5 * Q (k + 1)) (6 * Q (k + 1) - 1) ⊆ setA := fun x hx => mem_setA_M (k + 1) hx
    have sTk1 : Icc (10 * Q (k + 1) - 1) (15 * Q (k + 1)) ⊆ setA := fun x hx => mem_setA_T (k + 1) hx
    intro n hn; rw [Set.mem_Icc] at hn; obtain ⟨h4, hhi⟩ := hn
    by_cases h : n ≤ 30 * Q k
    · exact ih (by rw [Set.mem_Icc]; exact ⟨h4, h⟩)
    by_cases hb1 : n ≤ 35 * Q k
    · exact sum_cover sPk1 sTk (by omega) (by omega) (by omega) (by omega)
    by_cases hb2 : n ≤ 45 * Q k - 1
    · exact sum_cover sTk sMk1 (by omega) (by omega) (by omega) (by omega)
    by_cases hb3 : n ≤ 50 * Q k - 1
    · exact sum_cover sPk1 sMk1 (by omega) (by omega) (by omega) (by omega)
    by_cases hb4 : n ≤ 60 * Q k - 2
    · exact sum_cover sMk1 sMk1 (by omega) (by omega) (by omega) (by omega)
    by_cases hb5 : n ≤ 90 * Q k
    · exact sum_cover sTk sTk1 (by omega) (by omega) (by omega) (by omega)
    by_cases hb6 : n ≤ 105 * Q k - 1
    · exact sum_cover sMk1 sTk1 (by omega) (by omega) (by omega) (by omega)
    · exact sum_cover sTk1 sTk1 (by omega) (by omega) (by omega) (by omega)

lemma band {k x : ℕ} (hx : x ∈ setA) (hxle : x ≤ 10 * Q k - 3) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) := by
  have hQk : 1 ≤ Q k := one_le_Q k
  rw [mem_setA_iff] at hx
  obtain h | ⟨j, hj⟩ := hx
  · left; omega
  · have hQj : 1 ≤ Q j := one_le_Q j
    rcases hj with (hpt | hM) | hT
    · rcases lt_trichotomy j k with hjk | hjk | hjk
      · left; have h5 : 5 * Q j ≤ Q k := Q_step_le hjk; omega
      · subst hjk; right; left; exact hpt
      · exfalso; have h5 : 5 * Q k ≤ Q j := Q_step_le hjk; omega
    · rcases lt_trichotomy j k with hjk | hjk | hjk
      · left; have h5 : 5 * Q j ≤ Q k := Q_step_le hjk; omega
      · subst hjk; right; right; exact hM
      · exfalso; have h5 : 5 * Q k ≤ Q j := Q_step_le hjk; omega
    · rcases lt_trichotomy j k with hjk | hjk | hjk
      · left; have h5 : 5 * Q j ≤ Q k := Q_step_le hjk; omega
      · subst hjk; exfalso; omega
      · exfalso; have h5 : 5 * Q k ≤ Q j := Q_step_le hjk; omega

lemma rigidity (k : ℕ) {n : ℕ} (hn : n ∈ Icc (9 * Q k) (10 * Q k - 1))
    {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hQk : 1 ≤ Q k := one_le_Q k
  rw [Set.mem_Icc] at hn
  obtain ⟨h9, h10⟩ := hn
  have ha2 : 2 ≤ a := two_le ha
  have hb2 : 2 ≤ b := two_le hb
  have haU : a ≤ 10 * Q k - 3 := by omega
  have hbU : b ≤ 10 * Q k - 3 := by omega
  rcases band ha haU with hA | hA | hA
  · rcases band hb hbU with hB | hB | hB
    · exfalso; omega
    · right; exact hB
    · exfalso; omega
  · left; exact hA
  · rcases band hb hbU with hB | hB | hB
    · exfalso; omega
    · right; exact hB
    · exfalso; omega

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn4
    have hbound : n ≤ 30 * Q n := by
      have h1 : n < 2 ^ n := Nat.lt_two_pow_self
      have h2 : 2 ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
      have h3 : Q n = 5 ^ n := rfl
      omega
    have hmem : n ∈ (setA + setA) := basis_cover n (Set.mem_Icc.mpr ⟨hn4, hbound⟩)
    rw [Set.mem_add] at hmem
    exact hmem
  · intro A₁ A₂ hsub1 hsub2 hcov hdisj hsyn
    obtain ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩ := hsyn
    set k := C₁ + C₂ + 1 with hk
    have hkQ : k < Q k := by
      have h1 : k < 2 ^ k := Nat.lt_two_pow_self
      have h2 : 2 ^ k ≤ 5 ^ k := Nat.pow_le_pow_left (by norm_num) k
      have h3 : Q k = 5 ^ k := rfl
      omega
    have hC1k : C₁ < Q k := by omega
    have hC2k : C₂ < Q k := by omega
    have hQk : 1 ≤ Q k := one_le_Q k
    have hpiv : 4 * Q k ∈ setA := mem_setA_P k
    rcases hcov (4 * Q k) hpiv with hpc | hpc
    · obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
      rw [Set.mem_Icc] at hmI
      rw [Set.mem_add] at hmS
      obtain ⟨a, ha, b, hb, hab⟩ := hmS
      have hmwin : m ∈ Icc (9 * Q k) (10 * Q k - 1) := by rw [Set.mem_Icc]; omega
      have hrig := rigidity k hmwin (hsub2 ha) (hsub2 hb) hab
      have h4A2 : 4 * Q k ∈ A₂ := by
        rcases hrig with h | h
        · rw [← h]; exact ha
        · rw [← h]; exact hb
      have hcontra : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hpc, h4A2⟩
      rw [hdisj] at hcontra; exact hcontra
    · obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
      rw [Set.mem_Icc] at hmI
      rw [Set.mem_add] at hmS
      obtain ⟨a, ha, b, hb, hab⟩ := hmS
      have hmwin : m ∈ Icc (9 * Q k) (10 * Q k - 1) := by rw [Set.mem_Icc]; omega
      have hrig := rigidity k hmwin (hsub1 ha) (hsub1 hb) hab
      have h4A1 : 4 * Q k ∈ A₁ := by
        rcases hrig with h | h
        · rw [← h]; exact ha
        · rw [← h]; exact hb
      have hcontra : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h4A1, hpc⟩
      rw [hdisj] at hcontra; exact hcontra

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## Construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_le {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

lemma n_le_Q : ∀ n, n ≤ Q n
  | 0 => Nat.zero_le _
  | (n + 1) => by
      have h1 := n_le_Q n
      have h2 := Q_pos n
      rw [Q_succ]; omega

/-! ## Membership helpers -/

lemma ck_mem (k : ℕ) : ck k ∈ setA := by
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff]
  exact Or.inr ⟨k, Or.inl (Or.inl rfl)⟩

lemma Bk_mem {k x : ℕ} (h : x ∈ Bk k) : x ∈ setA := by
  simp only [setA, mem_union, mem_iUnion]
  exact Or.inr ⟨k, Or.inl (Or.inr h)⟩

lemma Fk_mem {k x : ℕ} (h : x ∈ Fk k) : x ∈ setA := by
  simp only [setA, mem_union, mem_iUnion]
  exact Or.inr ⟨k, Or.inr h⟩

lemma I_mem {k x : ℕ} (h : x ∈ Icc (2 * Q k) (3 * Q k)) : x ∈ setA := by
  rcases k with _ | m
  · simp only [Q, pow_zero, mul_one, mem_Icc] at h
    rcases (show x = 2 ∨ x = 3 by omega) with rfl | rfl
    · exact Or.inl (by simp)
    · exact Or.inl (by simp)
  · refine Fk_mem (k := m) ?_
    simp only [Fk, mem_Icc] at h ⊢
    have hq := Q_pos m
    rw [Q_succ] at h
    omega

lemma setA_ge2 {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  simp only [setA, mem_union, mem_iUnion] at hx
  rcases hx with hs | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at hs; omega
  · have hqj := Q_pos j
    simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hj
    rcases hj with (rfl | ⟨h, _⟩) | ⟨h, _⟩ <;> omega

/-! ## Classification: every element below 10*Q k -/

lemma key (k x : ℕ) (hx : x ∈ setA) (hlt : x < 10 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ x = 10 * Q k - 1 := by
  have hq := Q_pos k
  simp only [setA, mem_union, mem_iUnion] at hx
  rcases hx with hs | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at hs
    left; omega
  · have hqj := Q_pos j
    simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hj
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · have h5 : 5 * Q j ≤ Q k := by
        have hh : Q (j + 1) ≤ Q k := Q_le hjk
        rw [Q_succ] at hh; omega
      left
      rcases hj with (rfl | ⟨_, hb⟩) | ⟨_, hb⟩ <;> omega
    · rw [hjk] at hj
      rcases hj with (rfl | ⟨hb1, hb2⟩) | ⟨hb1, hb2⟩
      · right; left; rfl
      · right; right; left; exact ⟨hb1, hb2⟩
      · right; right; right; omega
    · exfalso
      have h5 : 5 * Q k ≤ Q j := by
        have hh : Q (k + 1) ≤ Q j := Q_le hjk
        rw [Q_succ] at hh; omega
      rcases hj with (rfl | ⟨hb, _⟩) | ⟨hb, _⟩ <;> omega

/-! ## Rigidity -/

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) (a b : ℕ)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hq := Q_pos k
  have ha2 := setA_ge2 ha
  have hb2 := setA_ge2 hb
  have hka : a < 10 * Q k := by omega
  have hkb : b < 10 * Q k := by omega
  have Ka := key k a ha hka
  have Kb := key k b hb hkb
  simp only [ck, Bk, mem_Icc]
  rcases Ka with Ka | Ka | Ka | Ka <;> rcases Kb with Kb | Kb | Kb | Kb <;> omega

/-! ## Gap lemma -/

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false, not_and]
  intro hnJ hadd
  rw [Set.mem_add] at hadd
  obtain ⟨a, ha, b, hb, hab⟩ := hadd
  rcases rigidity k n hnJ a b (hT ha) (hT hb) hab with ⟨hae, _⟩ | ⟨hbe, _⟩
  · exact hck (hae ▸ ha)
  · exact hck (hbe ▸ hb)

/-! ## Interval sumset is a full interval -/

lemma interval_sum (s1 s2 t1 t2 x : ℕ) (hs : s1 ≤ s2) (ht : t1 ≤ t2)
    (hlo : s1 + t1 ≤ x) (hhi : x ≤ s2 + t2) :
    ∃ a b, s1 ≤ a ∧ a ≤ s2 ∧ t1 ≤ b ∧ b ≤ t2 ∧ a + b = x := by
  refine ⟨max s1 (x - t2), x - max s1 (x - t2), ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-! ## Basis -/

lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ setA + setA := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    rw [Set.mem_add]
    have h2 : (2 : ℕ) ∈ setA := Or.inl (by simp)
    have h3 : (3 : ℕ) ∈ setA := Or.inl (by simp)
    rcases (show x = 4 ∨ x = 5 ∨ x = 6 by omega) with rfl | rfl | rfl
    · exact ⟨2, h2, 2, h2, rfl⟩
    · exact ⟨2, h2, 3, h3, rfl⟩
    · exact ⟨3, h3, 3, h3, rfl⟩
  | succ m ih =>
    intro x hx
    simp only [mem_Icc] at hx
    obtain ⟨hx4, hxhi⟩ := hx
    have hq := Q_pos m
    rw [Q_succ] at hxhi
    by_cases hle : x ≤ 6 * Q m
    · exact ih (by simp only [mem_Icc]; exact ⟨hx4, hle⟩)
    · push_neg at hle
      rw [Set.mem_add]
      by_cases h1 : x ≤ 7 * Q m
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (4 * Q m) (4 * Q m) (2 * Q m) (3 * Q m) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · have : a = 4 * Q m := by omega
          rw [this]; exact ck_mem m
        · exact I_mem (mem_Icc.mpr ⟨hta, htb⟩)
      by_cases h2 : x ≤ 9 * Q m - 1
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (5 * Q m) (6 * Q m - 1) (2 * Q m) (3 * Q m) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · exact Bk_mem (mem_Icc.mpr ⟨hsa, hsb⟩)
        · exact I_mem (mem_Icc.mpr ⟨hta, htb⟩)
      by_cases h3 : x ≤ 10 * Q m - 1
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (4 * Q m) (4 * Q m) (5 * Q m) (6 * Q m - 1) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · have : a = 4 * Q m := by omega
          rw [this]; exact ck_mem m
        · exact Bk_mem (mem_Icc.mpr ⟨hta, htb⟩)
      by_cases h4 : x ≤ 12 * Q m - 2
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (5 * Q m) (6 * Q m - 1) (5 * Q m) (6 * Q m - 1) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · exact Bk_mem (mem_Icc.mpr ⟨hsa, hsb⟩)
        · exact Bk_mem (mem_Icc.mpr ⟨hta, htb⟩)
      by_cases h5 : x ≤ 18 * Q m
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (2 * Q m) (3 * Q m) (10 * Q m - 1) (15 * Q m) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · exact I_mem (mem_Icc.mpr ⟨hsa, hsb⟩)
        · exact Fk_mem (mem_Icc.mpr ⟨hta, htb⟩)
      by_cases h6 : x ≤ 21 * Q m - 1
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (5 * Q m) (6 * Q m - 1) (10 * Q m - 1) (15 * Q m) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · exact Bk_mem (mem_Icc.mpr ⟨hsa, hsb⟩)
        · exact Fk_mem (mem_Icc.mpr ⟨hta, htb⟩)
      · obtain ⟨a, b, hsa, hsb, hta, htb, hsum⟩ :=
          interval_sum (10 * Q m - 1) (15 * Q m) (10 * Q m - 1) (15 * Q m) x
            (by omega) (by omega) (by omega) (by omega)
        refine ⟨a, ?_, b, ?_, hsum⟩
        · exact Fk_mem (mem_Icc.mpr ⟨hsa, hsb⟩)
        · exact Fk_mem (mem_Icc.mpr ⟨hta, htb⟩)

/-! ## Main theorem -/

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn
    have hk : n ≤ 6 * Q n := by
      have h := n_le_Q n
      have hp := Q_pos n
      omega
    have hmem : n ∈ Icc 4 (6 * Q n) := mem_Icc.mpr ⟨hn, hk⟩
    have hsum := basis_lem n hmem
    rw [Set.mem_add] at hsum
    obtain ⟨a, ha, b, hb, hab⟩ := hsum
    exact ⟨a, ha, b, hb, hab⟩
  · intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hS1⟩, ⟨C₂, hS2⟩⟩
    set k := C₁ + C₂ + 1 with hkdef
    have hk2 : k ≤ Q k := n_le_Q k
    have hqk := Q_pos k
    have hC1 : C₁ < Q k := by omega
    have hC2 : C₂ < Q k := by omega
    have hckA : ck k ∈ setA := ck_mem k
    rcases hcov (ck k) hckA with hin1 | hin2
    · have hck2 : ck k ∉ A₂ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hc⟩
        rw [hdisj] at hmem; exact hmem
      have hgap := gap_lem k A₂ h2 hck2
      obtain ⟨m, hmA, hmI⟩ := hS2 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hfin : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmA⟩
      rw [hgap] at hfin
      exact hfin
    · have hck1 : ck k ∉ A₁ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hc, hin2⟩
        rw [hdisj] at hmem; exact hmem
      have hgap := gap_lem k A₁ h1 hck1
      obtain ⟨m, hmA, hmI⟩ := hS1 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hfin : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmA⟩
      rw [hgap] at hfin
      exact hfin

end Erdos741OAI

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

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

/-- partial union up through level k -/
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_le (j k : ℕ) (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

lemma akn_mono {j k : ℕ} (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction h with
  | refl => exact subset_rfl
  | @step k _ ih =>
      intro x hx
      exact Or.inl (ih hx)

lemma lt_Q (k : ℕ) : k < Q k := by
  induction k with
  | zero => simp [Q]
  | succ k ih =>
      have h := Q_succ k
      omega

lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
      intro x hx
      rcases hx with hx | hx
      · exact ih hx
      · exact Or.inr (Set.mem_iUnion.mpr ⟨k, hx⟩)

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

/-! ## Basis lemma -/

lemma ck_mem_akn (k : ℕ) : (4 * Q k) ∈ Akn (k+1) := Or.inr (Or.inl (Or.inl rfl))

lemma Bk_sub_akn (k : ℕ) : Bk k ⊆ Akn (k+1) := fun _ hx => Or.inr (Or.inl (Or.inr hx))

lemma Fk_sub_akn (k : ℕ) : Fk k ⊆ Akn (k+1) := fun _ hx => Or.inr (Or.inr hx)

lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1) := by
  cases k with
  | zero =>
      intro x hx
      rw [mem_Icc] at hx
      simp only [Q, pow_zero, mul_one] at hx
      refine Or.inl ?_
      show x ∈ ({2,3} : Set ℕ)
      simp only [mem_insert_iff, mem_singleton_iff]
      omega
  | succ k' =>
      intro x hx
      rw [mem_Icc] at hx
      have hQ : Q (k'+1) = 5 * Q k' := Q_succ k'
      refine Or.inl ?_
      refine Or.inr ?_
      refine Or.inr ?_
      rw [Fk, mem_Icc]
      omega

lemma pairs_lem (k : ℕ) : Icc (4 * Q k) (30 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hx1, hx2⟩ := hx
  have hQk : 0 < Q k := Q_pos k
  have hI : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k+1) := by
    intro y h1 h2; exact I_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  have hB : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k+1) := by
    intro y h1 h2; exact Bk_sub_akn k (mem_Icc.mpr ⟨h1, h2⟩)
  have hF : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k+1) := by
    intro y h1 h2; exact Fk_sub_akn k (mem_Icc.mpr ⟨h1, h2⟩)
  have hck : (4 * Q k) ∈ Akn (k+1) := ck_mem_akn k
  by_cases h : x ≤ (5 * Q k)
  · exact Set.mem_add.mpr ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases h2 : x ≤ (6 * Q k)
  · exact Set.mem_add.mpr ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases h3 : x ≤ (7 * Q k)
  · exact Set.mem_add.mpr ⟨4 * Q k, hck, x - 4 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases h4 : x ≤ (8 * Q k - 1)
  · exact Set.mem_add.mpr ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases h5 : x ≤ (9 * Q k - 1)
  · exact Set.mem_add.mpr ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases h6 : x ≤ (10 * Q k - 1)
  · exact Set.mem_add.mpr ⟨4 * Q k, hck, x - 4 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases h7 : x ≤ (11 * Q k - 1)
  · exact Set.mem_add.mpr ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases h8 : x ≤ (12 * Q k - 2)
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, hB _ (by omega) (by omega), x - (6 * Q k - 1), hB _ (by omega) (by omega), by omega⟩
  by_cases h9 : x ≤ (17 * Q k)
  · exact Set.mem_add.mpr ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases h10 : x ≤ (18 * Q k)
  · exact Set.mem_add.mpr ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases h11 : x ≤ (20 * Q k)
  · exact Set.mem_add.mpr ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases h12 : x ≤ (25 * Q k - 1)
  · exact Set.mem_add.mpr ⟨10 * Q k - 1, hF _ (by omega) (by omega), x - (10 * Q k - 1), hF _ (by omega) (by omega), by omega⟩
  · exact Set.mem_add.mpr ⟨15 * Q k, hF _ (by omega) (by omega), x - 15 * Q k, hF _ (by omega) (by omega), by omega⟩

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
      intro x hx
      rw [mem_Icc] at hx
      simp only [Q, pow_zero, mul_one] at hx
      have h2 : (2:ℕ) ∈ Akn 1 := by
        refine Or.inl ?_; show (2:ℕ) ∈ ({2,3} : Set ℕ); simp
      have h3 : (3:ℕ) ∈ Akn 1 := by
        refine Or.inl ?_; show (3:ℕ) ∈ ({2,3} : Set ℕ); simp
      rcases (by omega : x = 4 ∨ x = 5 ∨ x = 6) with h | h | h
      · exact Set.mem_add.mpr ⟨2, h2, 2, h2, by omega⟩
      · exact Set.mem_add.mpr ⟨2, h2, 3, h3, by omega⟩
      · exact Set.mem_add.mpr ⟨3, h3, 3, h3, by omega⟩
  | succ k ih =>
      intro x hx
      rw [mem_Icc] at hx
      have hQs : Q (k+1) = 5 * Q k := Q_succ k
      have hQk : 0 < Q k := Q_pos k
      by_cases hsplit : x ≤ 6 * Q k
      · have hmem : x ∈ Akn (k+1) + Akn (k+1) := ih (mem_Icc.mpr ⟨hx.1, hsplit⟩)
        exact Set.add_subset_add (akn_mono (Nat.le_succ _)) (akn_mono (Nat.le_succ _)) hmem
      · push_neg at hsplit
        have hxIcc : x ∈ Icc (4 * Q k) (30 * Q k) := by
          rw [mem_Icc]; omega
        have hmem : x ∈ Akn (k+1) + Akn (k+1) := pairs_lem k hxIcc
        exact Set.add_subset_add (akn_mono (Nat.le_succ _)) (akn_mono (Nat.le_succ _)) hmem

/-! ## Rigidity lemma -/

lemma small_stage {j k : ℕ} (h : j < k) : 15 * Q j ≤ 3 * Q k := by
  have h1 : Q (j+1) ≤ Q k := Q_le (j+1) k (by omega)
  have h2 : Q (j+1) = 5 * Q j := Q_succ j
  omega

lemma large_stage {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have h1 : Q (k+1) ≤ Q j := Q_le (k+1) j (by omega)
  have h2 : Q (k+1) = 5 * Q k := Q_succ k
  omega

lemma mem_setA_cases {e : ℕ} (he : e ∈ setA) :
    e = 2 ∨ e = 3 ∨ ∃ j, e = ck j ∨ e ∈ Bk j ∨ e ∈ Fk j := by
  simp only [setA, stage, mem_union, mem_iUnion, mem_singleton_iff,
    mem_insert_iff] at he
  rcases he with (h | h) | ⟨j, hj⟩
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (h | h) | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

lemma setA_ge_two {e : ℕ} (he : e ∈ setA) : 2 ≤ e := by
  rcases mem_setA_cases he with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have hQj : 0 < Q j := Q_pos j
    rcases hj with h | h | h
    · rw [ck] at h; omega
    · rw [Bk, mem_Icc] at h; omega
    · rw [Fk, mem_Icc] at h; omega

lemma classify {k e : ℕ} (he : e ∈ setA) (hlt : e < 20 * Q k) :
    e ≤ 3 * Q k ∨ e = ck k ∨ e ∈ Bk k ∨ e ∈ Fk k := by
  have hQk : 0 < Q k := Q_pos k
  rcases mem_setA_cases he with h2 | h3 | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt' | hje | hgt
    · have hsm : 15 * Q j ≤ 3 * Q k := small_stage hlt'
      left
      rcases hj with he' | he' | he'
      · rw [ck] at he'; omega
      · rw [Bk, mem_Icc] at he'; omega
      · rw [Fk, mem_Icc] at he'; omega
    · rw [hje] at hj
      rcases hj with he' | he' | he'
      · exact Or.inr (Or.inl he')
      · exact Or.inr (Or.inr (Or.inl he'))
      · exact Or.inr (Or.inr (Or.inr he'))
    · exfalso
      have hbig : 5 * Q k ≤ Q j := large_stage hgt
      rcases hj with he' | he' | he'
      · rw [he', ck] at hlt; omega
      · rw [Bk, mem_Icc] at he'; omega
      · rw [Fk, mem_Icc] at he'; omega

lemma rigidity (k : ℕ) {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hn : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hQk : 0 < Q k := Q_pos k
  have ha2 : 2 ≤ a := setA_ge_two ha
  have hb2 : 2 ≤ b := setA_ge_two hb
  rw [Jk, mem_Ico] at hn
  obtain ⟨hlo, hhi⟩ := hn
  have ha20 : a < 20 * Q k := by omega
  have hb20 : b < 20 * Q k := by omega
  have hca := classify ha ha20
  have hcb := classify hb hb20
  rcases hca with ha' | ha' | ha' | ha'
  · -- a ≤ 3 Qk
    rcases hcb with hb' | hb' | hb' | hb'
    · exfalso; omega
    · exfalso; rw [ck] at hb'; omega
    · exfalso; rw [Bk, mem_Icc] at hb'; omega
    · exfalso; rw [Fk, mem_Icc] at hb'; omega
  · -- a = ck k
    rcases hcb with hb' | hb' | hb' | hb'
    · exfalso; rw [ck] at ha'; omega
    · exfalso; rw [ck] at ha' hb'; omega
    · exact Or.inl ⟨ha', hb'⟩
    · exfalso; rw [ck] at ha'; rw [Fk, mem_Icc] at hb'; omega
  · -- a ∈ Bk k
    rcases hcb with hb' | hb' | hb' | hb'
    · exfalso; rw [Bk, mem_Icc] at ha'; omega
    · exact Or.inr ⟨hb', ha'⟩
    · exfalso; rw [Bk, mem_Icc] at ha' hb'; omega
    · exfalso; rw [Bk, mem_Icc] at ha'; rw [Fk, mem_Icc] at hb'; omega
  · -- a ∈ Fk k
    rcases hcb with hb' | hb' | hb' | hb'
    · exfalso; rw [Fk, mem_Icc] at ha'; omega
    · exfalso; rw [Fk, mem_Icc] at ha'; rw [ck] at hb'; omega
    · exfalso; rw [Fk, mem_Icc] at ha'; rw [Bk, mem_Icc] at hb'; omega
    · exfalso; rw [Fk, mem_Icc] at ha'; rw [Fk, mem_Icc] at hb'; omega

/-! ## Gap lemma -/

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hxJ, hxTT⟩
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hxTT
  have hnJ : a + b ∈ Jk k := by rw [hab]; exact hxJ
  rcases rigidity k (hT ha) (hT hb) hnJ with ⟨hae, _⟩ | ⟨hbe, _⟩
  · exact hck (hae ▸ ha)
  · exact hck (hbe ▸ hb)

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
  · -- basis
    intro n hn
    have hk : n ≤ 6 * Q n := by have := lt_Q n; omega
    have hmem : n ∈ Icc 4 (6 * Q n) := mem_Icc.mpr ⟨hn, hk⟩
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (basis_lem n hmem)
    exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩
  · -- no both-syndetic partition
    intro A₁ A₂ hA1 hA2 hcover hdisj ⟨hsyn1, hsyn2⟩
    obtain ⟨C₁, hC1⟩ := hsyn1
    obtain ⟨C₂, hC2⟩ := hsyn2
    set k := C₁ + C₂ + 1 with hkdef
    have hQk : k < Q k := lt_Q k
    have hC1k : C₁ < Q k := by omega
    have hC2k : C₂ < Q k := by omega
    have hck_mem : ck k ∈ setA := ck_mem_setA k
    rcases hcover (ck k) hck_mem with hin1 | hin2
    · -- ck k ∈ A₁, derive contradiction via A₂ syndetic
      have hck2 : ck k ∉ A₂ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hc⟩
        simp [hdisj] at this
      have hgap := gap_lem k A₂ hA2 hck2
      obtain ⟨m, hmA2, hmIcc⟩ := hC2 (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]; omega
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmA2⟩
      rw [hgap] at this
      simp at this
    · -- ck k ∈ A₂, derive contradiction via A₁ syndetic
      have hck1 : ck k ∉ A₁ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hc, hin2⟩
        simp [hdisj] at this
      have hgap := gap_lem k A₁ hA1 hck1
      obtain ⟨m, hmA1, hmIcc⟩ := hC1 (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]; omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmA1⟩
      rw [hgap] at this
      simp at this

end Erdos741OAI

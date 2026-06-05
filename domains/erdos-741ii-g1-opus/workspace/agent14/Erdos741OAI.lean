import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- YOUR TASK: implement the construction described in program.md and prove the theorem below.
-- Read mathlib_hints.md before you start — it lists the exact Mathlib lemmas you need.

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## The construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def St (k : ℕ) : Set ℕ := {4 * Q k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, St k

def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | (k+1) => Akn k ∪ St k

/-! ## Q lemmas -/

lemma Q_pos (k : ℕ) : 1 ≤ Q k := by unfold Q; exact pow_pos (by norm_num : (0:ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by unfold Q; rw [pow_succ]; ring

lemma Q_le_Q {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma q_lt (k : ℕ) : k < Q k := by
  induction k with
  | zero => norm_num [Q]
  | succ n ih =>
    rw [Q_succ]
    have := Q_pos n
    omega

/-! ## membership helpers -/

lemma ck_mem_St (k : ℕ) : (4 * Q k) ∈ St k :=
  Set.mem_union_left _ (Set.mem_union_left _ rfl)

lemma Bk_sub_St (k : ℕ) : Bk k ⊆ St k := by
  intro x hx
  exact Set.mem_union_left _ (Set.mem_union_right _ hx)

lemma Fk_sub_St (k : ℕ) : Fk k ⊆ St k := by
  intro x hx
  exact Set.mem_union_right _ hx

lemma St_sub_akn (k : ℕ) : St k ⊆ Akn (k+1) := by
  intro x hx
  exact Set.mem_union_right _ hx

lemma akn_le (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  exact Set.mem_union_left _ hx

lemma akn_sub : ∀ m, Akn m ⊆ setA := by
  intro m
  induction m with
  | zero => intro x hx; exact Set.mem_union_left _ hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Set.mem_union_right _ (mem_iUnion.mpr ⟨k, hx⟩)

lemma ck_mem_setA (k : ℕ) : (4 * Q k) ∈ setA :=
  Set.mem_union_right _ (mem_iUnion.mpr ⟨k, ck_mem_St k⟩)

lemma setA_ge2 {a : ℕ} (ha : a ∈ setA) : 2 ≤ a := by
  rcases ha with ha | ha
  · simp only [mem_Icc] at ha; omega
  · rw [mem_iUnion] at ha
    obtain ⟨j, hj⟩ := ha
    have hpos := Q_pos j
    rcases hj with (hj | hj) | hj
    · simp only [mem_singleton_iff] at hj; omega
    · simp only [Bk, mem_Icc] at hj; omega
    · simp only [Fk, mem_Icc] at hj; omega

/-! ## I interval inclusion -/

lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1) := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    exact Set.mem_union_left _ (mem_Icc.mpr ⟨hx.1, hx.2⟩)
  | succ k =>
    intro x hx
    rw [Q_succ] at hx
    simp only [mem_Icc] at hx
    obtain ⟨h1, h2⟩ := hx
    have hpos := Q_pos k
    have hxF : x ∈ Fk k := mem_Icc.mpr ⟨by omega, by omega⟩
    exact akn_le (k+1) (St_sub_akn k (Fk_sub_St k hxF))

/-! ## basis lemma -/

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    have hmem : ∀ y, 2 ≤ y → y ≤ 3 → y ∈ Akn 1 := fun y hy1 hy2 =>
      Set.mem_union_left _ (mem_Icc.mpr ⟨hy1, hy2⟩)
    by_cases hc : x ≤ 5
    · exact Set.mem_add.mpr ⟨2, hmem 2 (by omega) (by omega), x - 2,
        hmem (x - 2) (by omega) (by omega), by omega⟩
    · exact Set.mem_add.mpr ⟨3, hmem 3 (by omega) (by omega), x - 3,
        hmem (x - 3) (by omega) (by omega), by omega⟩
  | succ k ih =>
    intro x hx
    simp only [mem_Icc] at hx
    obtain ⟨hx4, hxhi⟩ := hx
    rw [Q_succ] at hxhi
    have hpos := Q_pos k
    have hmono : Akn (k+1) ⊆ Akn (k+2) := akn_le (k+1)
    have hmono2 : Akn (k+1) + Akn (k+1) ⊆ Akn (k+2) + Akn (k+2) :=
      Set.add_subset_add hmono hmono
    have hI : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k+2) :=
      fun y ha hb => hmono (I_sub k (mem_Icc.mpr ⟨ha, hb⟩))
    have hck : (4 * Q k) ∈ Akn (k+2) := hmono (St_sub_akn k (ck_mem_St k))
    have hB : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k+2) :=
      fun y ha hb => hmono (St_sub_akn k (Bk_sub_St k (mem_Icc.mpr ⟨ha, hb⟩)))
    have hF : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k+2) :=
      fun y ha hb => hmono (St_sub_akn k (Fk_sub_St k (mem_Icc.mpr ⟨ha, hb⟩)))
    by_cases hb6 : x ≤ 6 * Q k
    · exact hmono2 (ih (mem_Icc.mpr ⟨hx4, hb6⟩))
    · push_neg at hb6
      rcases le_or_gt x (7 * Q k) with h | h
      · exact Set.mem_add.mpr ⟨x - 4 * Q k, hI _ (by omega) (by omega), 4 * Q k, hck, by omega⟩
      · rcases le_or_gt x (9 * Q k - 1) with h' | h'
        · rcases le_or_gt x (8 * Q k - 1) with h2 | h2
          · exact Set.mem_add.mpr ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k,
              hB _ (by omega) (by omega), by omega⟩
          · exact Set.mem_add.mpr ⟨x - (6 * Q k - 1), hI _ (by omega) (by omega), 6 * Q k - 1,
              hB _ (by omega) (by omega), by omega⟩
        · rcases le_or_gt x (10 * Q k - 1) with h'' | h''
          · exact Set.mem_add.mpr ⟨4 * Q k, hck, x - 4 * Q k, hB _ (by omega) (by omega), by omega⟩
          · rcases le_or_gt x (12 * Q k - 2) with h3 | h3
            · rcases le_or_gt x (11 * Q k - 1) with h4 | h4
              · exact Set.mem_add.mpr ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k,
                  hB _ (by omega) (by omega), by omega⟩
              · exact Set.mem_add.mpr ⟨x - (6 * Q k - 1), hB _ (by omega) (by omega), 6 * Q k - 1,
                  hB _ (by omega) (by omega), by omega⟩
            · rcases le_or_gt x (18 * Q k) with h5 | h5
              · rcases le_or_gt x (17 * Q k) with h6 | h6
                · exact Set.mem_add.mpr ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k,
                    hF _ (by omega) (by omega), by omega⟩
                · exact Set.mem_add.mpr ⟨x - 15 * Q k, hI _ (by omega) (by omega), 15 * Q k,
                    hF _ (by omega) (by omega), by omega⟩
              · rcases le_or_gt x (21 * Q k - 1) with h7 | h7
                · rcases le_or_gt x (20 * Q k) with h8 | h8
                  · exact Set.mem_add.mpr ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k,
                      hF _ (by omega) (by omega), by omega⟩
                  · exact Set.mem_add.mpr ⟨x - 15 * Q k, hB _ (by omega) (by omega), 15 * Q k,
                      hF _ (by omega) (by omega), by omega⟩
                · rcases le_or_gt x (25 * Q k - 1) with h9 | h9
                  · exact Set.mem_add.mpr ⟨10 * Q k - 1, hF _ (by omega) (by omega),
                      x - (10 * Q k - 1), hF _ (by omega) (by omega), by omega⟩
                  · exact Set.mem_add.mpr ⟨x - 15 * Q k, hF _ (by omega) (by omega), 15 * Q k,
                      hF _ (by omega) (by omega), by omega⟩

/-! ## classification & rigidity -/

lemma classify (k : ℕ) {a : ℕ} (ha : a ∈ setA) (hlt : a < 10 * Q k) :
    a ≤ 3 * Q k ∨ a = 4 * Q k ∨ (5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1) ∨
      (10 * Q k - 1 ≤ a ∧ a < 10 * Q k) := by
  have hpos := Q_pos k
  rcases ha with ha | ha
  · simp only [mem_Icc] at ha
    left; omega
  · rw [mem_iUnion] at ha
    obtain ⟨j, hj⟩ := ha
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hb : 5 * Q j ≤ Q k := by
        have h1 : Q (j+1) ≤ Q k := Q_le_Q hjk
        rw [Q_succ] at h1; exact h1
      have hposj := Q_pos j
      left
      rcases hj with (hj | hj) | hj
      · simp only [mem_singleton_iff] at hj; omega
      · simp only [Bk, mem_Icc] at hj; omega
      · simp only [Fk, mem_Icc] at hj; omega
    · rw [hjk] at hj
      rcases hj with (hj | hj) | hj
      · simp only [mem_singleton_iff] at hj
        right; left; exact hj
      · simp only [Bk, mem_Icc] at hj
        right; right; left; exact hj
      · simp only [Fk, mem_Icc] at hj
        right; right; right; exact ⟨hj.1, hlt⟩
    · have hgj : 5 * Q k ≤ Q j := by
        have h1 : Q (k+1) ≤ Q j := Q_le_Q hjk
        rw [Q_succ] at h1; exact h1
      have hposj := Q_pos j
      exfalso
      rcases hj with (hj | hj) | hj
      · simp only [mem_singleton_iff] at hj; omega
      · simp only [Bk, mem_Icc] at hj; omega
      · simp only [Fk, mem_Icc] at hj; omega

lemma rigidity (k : ℕ) {a b n : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) (hlo : 9 * Q k ≤ n) (hhi : n < 10 * Q k) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hpos := Q_pos k
  have ha2 := setA_ge2 ha
  have hb2 := setA_ge2 hb
  have haL : a < 10 * Q k := by omega
  have hbL : b < 10 * Q k := by omega
  rcases classify k ha haL with ca | ca | ca | ca
  · rcases classify k hb hbL with cb | cb | cb | cb <;> omega
  · rcases classify k hb hbL with cb | cb | cb | cb
    · omega
    · omega
    · exact Or.inl ca
    · omega
  · rcases classify k hb hbL with cb | cb | cb | cb
    · omega
    · exact Or.inr cb
    · omega
    · omega
  · rcases classify k hb hbL with cb | cb | cb | cb <;> omega

/-! ## gap lemma -/

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : (4 * Q k) ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [eq_empty_iff_forall_notMem]
  intro n hmem
  rw [mem_inter_iff] at hmem
  obtain ⟨hn, hsum⟩ := hmem
  rw [Set.mem_add] at hsum
  obtain ⟨a, haT, b, hbT, hab⟩ := hsum
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hlo, hhi⟩ := hn
  rcases rigidity k (hT haT) (hT hbT) hab hlo hhi with h | h
  · exact hck (h ▸ haT)
  · exact hck (h ▸ hbT)

/-! ## main theorem -/

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
    have hq := q_lt n
    have hpos := Q_pos n
    have hmem : n ∈ Icc 4 (6 * Q n) := mem_Icc.mpr ⟨hn, by omega⟩
    have hsum := basis_lem n hmem
    rw [Set.mem_add] at hsum
    obtain ⟨a, ha, b, hb, hab⟩ := hsum
    exact ⟨a, akn_sub (n+1) ha, b, akn_sub (n+1) hb, hab⟩
  · rintro A₁ A₂ h1 h2 hcov hdisj ⟨hs1, hs2⟩
    obtain ⟨C₁, hC1⟩ := hs1
    obtain ⟨C₂, hC2⟩ := hs2
    set M := C₁ + C₂ with hM
    have hkbig : M < Q M := q_lt M
    have hckA : (4 * Q M) ∈ setA := ck_mem_setA M
    rcases hcov _ hckA with hin | hin
    · have hnot : (4 * Q M) ∉ A₂ := by
        intro hc
        have hmem : (4 * Q M) ∈ A₁ ∩ A₂ := ⟨hin, hc⟩
        rw [hdisj] at hmem
        simp at hmem
      have hgap := gap_lem M A₂ h2 hnot
      obtain ⟨m, hmA2, hmIcc⟩ := hC2 (9 * Q M)
      simp only [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk M := by
        simp only [Jk, mem_Ico]
        exact ⟨hmIcc.1, by omega⟩
      have hmem : m ∈ Jk M ∩ (A₂ + A₂) := ⟨hmJ, hmA2⟩
      rw [hgap] at hmem
      simp at hmem
    · have hnot : (4 * Q M) ∉ A₁ := by
        intro hc
        have hmem : (4 * Q M) ∈ A₁ ∩ A₂ := ⟨hc, hin⟩
        rw [hdisj] at hmem
        simp at hmem
      have hgap := gap_lem M A₁ h1 hnot
      obtain ⟨m, hmA1, hmIcc⟩ := hC1 (9 * Q M)
      simp only [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk M := by
        simp only [Jk, mem_Ico]
        exact ⟨hmIcc.1, by omega⟩
      have hmem : m ∈ Jk M ∩ (A₁ + A₁) := ⟨hmJ, hmA1⟩
      rw [hgap] at hmem
      simp at hmem

end Erdos741OAI

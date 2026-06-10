import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- The construction
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

-- partial union through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) :=
  fun _ hx => Or.inl (Or.inl (Or.inl hx))

lemma akn_le {j k : ℕ} (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans (akn_mono _)

lemma ck_mem (k : ℕ) : (4 * Q k) ∈ Akn (k+1) :=
  Or.inl (Or.inl (Or.inr rfl))

lemma Bk_sub (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ Akn (k+1) :=
  fun _ hx => Or.inl (Or.inr hx)

lemma Fk_sub (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ Akn (k+1) :=
  fun _ hx => Or.inr hx

lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with ((hx | hx) | hx) | hx
    · exact ih hx
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl hx)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inr hx)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inr hx⟩)

lemma inI (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    apply akn_mono 0
    simp only [Akn, Set.mem_insert_iff, Set.mem_singleton_iff]
    omega
  | succ k =>
    intro x hx
    simp only [mem_Icc] at hx
    rw [Q_succ] at hx
    have hmem : x ∈ Icc (10 * Q k - 1) (15 * Q k) := by
      simp only [mem_Icc]; have := Q_pos k; omega
    exact akn_mono (k + 1) (Fk_sub k hmem)

lemma basis_cover : ∀ k, ∀ x, x ∈ Icc 4 (6 * Q k) →
    ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    obtain ⟨hx4, hx6⟩ := hx
    have h2 : (2 : ℕ) ∈ Akn 1 := akn_le (Nat.zero_le 1) (by simp [Akn])
    have h3 : (3 : ℕ) ∈ Akn 1 := akn_le (Nat.zero_le 1) (by simp [Akn])
    interval_cases x
    · exact ⟨2, h2, 2, h2, rfl⟩
    · exact ⟨2, h2, 3, h3, rfl⟩
    · exact ⟨3, h3, 3, h3, rfl⟩
  | succ k ih =>
    intro x hx
    simp only [mem_Icc] at hx
    obtain ⟨hx4, hxhi⟩ := hx
    rw [Q_succ] at hxhi
    have hQk := Q_pos k
    by_cases hsmall : x ≤ 6 * Q k
    · obtain ⟨a, ha, b, hb, hab⟩ := ih x (mem_Icc.mpr ⟨hx4, hsmall⟩)
      exact ⟨a, akn_mono _ ha, b, akn_mono _ hb, hab⟩
    push_neg at hsmall
    by_cases c1 : x ≤ 7 * Q k
    · exact ⟨x - 4 * Q k, akn_mono _ (inI k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            4 * Q k, akn_mono _ (ck_mem k), by omega⟩
    by_cases c2 : x ≤ 8 * Q k - 1
    · exact ⟨2 * Q k, akn_mono _ (inI k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 2 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c3 : x ≤ 9 * Q k - 1
    · exact ⟨3 * Q k, akn_mono _ (inI k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 3 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c4 : x ≤ 10 * Q k - 1
    · exact ⟨4 * Q k, akn_mono _ (ck_mem k),
            x - 4 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c5 : x ≤ 11 * Q k - 1
    · exact ⟨5 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 5 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c6 : x ≤ 12 * Q k - 2
    · exact ⟨x - 6 * Q k + 1, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            6 * Q k - 1, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c7 : x ≤ 17 * Q k
    · exact ⟨2 * Q k, akn_mono _ (inI k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 2 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c8 : x ≤ 18 * Q k
    · exact ⟨3 * Q k, akn_mono _ (inI k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 3 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c9 : x ≤ 20 * Q k
    · exact ⟨5 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 5 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c10 : x ≤ 21 * Q k - 1
    · exact ⟨x - 15 * Q k, akn_mono _ (Bk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            15 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    by_cases c11 : x ≤ 25 * Q k - 1
    · exact ⟨10 * Q k - 1, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - (10 * Q k - 1), akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩
    · exact ⟨15 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)),
            x - 15 * Q k, akn_mono _ (Fk_sub k (mem_Icc.mpr ⟨by omega, by omega⟩)), by omega⟩

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ n ih =>
    rw [Q_succ]
    have := Q_pos n
    omega

lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  have hnle : n ≤ 6 * Q n := by have := n_le_Q n; have := Q_pos n; omega
  obtain ⟨a, ha, b, hb, hab⟩ := basis_cover n n (mem_Icc.mpr ⟨hn, hnle⟩)
  exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩

lemma mem_setA_cases {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, (x = 4 * Q j ∨ x ∈ Icc (5 * Q j) (6 * Q j - 1) ∨
      x ∈ Icc (10 * Q j - 1) (15 * Q j)) := by
  rcases hx with h | h
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
  · rw [mem_iUnion] at h
    obtain ⟨j, hj⟩ := h
    refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (h | h) | h
    · left; simpa [ck, Set.mem_singleton_iff] using h
    · right; left; exact h
    · right; right; exact h

lemma setA_ge_two {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases mem_setA_cases hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have hQ := Q_pos j
    rcases hj with h | h | h
    · omega
    · simp only [mem_Icc] at h; omega
    · simp only [mem_Icc] at h; omega

lemma pin {x k : ℕ} (hx : x ∈ setA) (hub : x + 2 ≤ 10 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ x ∈ Icc (5 * Q k) (6 * Q k - 1) := by
  have hQk := Q_pos k
  rcases mem_setA_cases hx with h | h | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have h6 : Q (j + 1) ≤ Q k := Q_mono hlt
      rw [Q_succ] at h6
      have hxle : x ≤ 15 * Q j := by
        rcases hj with h | h | h
        · have := Q_pos j; omega
        · simp only [mem_Icc] at h; have := Q_pos j; omega
        · simp only [mem_Icc] at h; omega
      omega
    · rw [hje] at hj
      rcases hj with h | h | h
      · right; left; exact h
      · right; right; exact h
      · exfalso; simp only [mem_Icc] at h; omega
    · exfalso
      have h6 : Q (k + 1) ≤ Q j := Q_mono hgt
      rw [Q_succ] at h6
      have hxge : 4 * Q j ≤ x := by
        rcases hj with h | h | h
        · omega
        · simp only [mem_Icc] at h; have := Q_pos j; omega
        · simp only [mem_Icc] at h; have := Q_pos j; omega
      omega

lemma rigidity {k n a b : ℕ} (hn : n ∈ Ico (9 * Q k) (10 * Q k))
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = 4 * Q k ∧ b ∈ Icc (5 * Q k) (6 * Q k - 1)) ∨
    (b = 4 * Q k ∧ a ∈ Icc (5 * Q k) (6 * Q k - 1)) := by
  simp only [mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hQk := Q_pos k
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have hpa := pin (k := k) ha (by omega)
  have hpb := pin (k := k) hb (by omega)
  rcases hpa with hsa | hca | hba
  · exfalso
    rcases hpb with hsb | hcb | hbb
    · omega
    · omega
    · simp only [mem_Icc] at hbb; omega
  · rcases hpb with hsb | hcb | hbb
    · exfalso; omega
    · exfalso; omega
    · exact Or.inl ⟨hca, hbb⟩
  · rcases hpb with hsb | hcb | hbb
    · exfalso; simp only [mem_Icc] at hba; omega
    · exact Or.inr ⟨hcb, hba⟩
    · exfalso; simp only [mem_Icc] at hba hbb; omega

lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : (4 * Q k) ∉ T) :
    ∀ m, m ∈ Ico (9 * Q k) (10 * Q k) → m ∉ (T + T) := by
  intro m hm hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  rcases rigidity hm (hT ha) (hT hb) hab with ⟨h1, _⟩ | ⟨h1, _⟩
  · exact hck (h1 ▸ ha)
  · exact hck (h1 ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, basis_lem, ?_⟩
  intro A₁ A₂ h1 h2 hcov hdisj
  rintro ⟨⟨C₁, hsyn1⟩, ⟨C₂, hsyn2⟩⟩
  set K := C₁ + C₂ + 1 with hKdef
  have hKle : K ≤ Q K := n_le_Q K
  have hckA : (4 * Q K) ∈ setA := akn_sub_setA _ (ck_mem K)
  rcases hcov _ hckA with hin1 | hin2
  · have hck2 : (4 * Q K) ∉ A₂ := by
      intro hmem
      have hh : (4 * Q K) ∈ A₁ ∩ A₂ := ⟨hin1, hmem⟩
      rw [hdisj] at hh; exact hh
    obtain ⟨m, hmS, hmI⟩ := hsyn2 (9 * Q K)
    simp only [mem_Icc] at hmI
    have hmIco : m ∈ Ico (9 * Q K) (10 * Q K) := by
      rw [mem_Ico]; exact ⟨hmI.1, by omega⟩
    exact gap_lem h2 hck2 m hmIco hmS
  · have hck1 : (4 * Q K) ∉ A₁ := by
      intro hmem
      have hh : (4 * Q K) ∈ A₁ ∩ A₂ := ⟨hmem, hin2⟩
      rw [hdisj] at hh; exact hh
    obtain ⟨m, hmS, hmI⟩ := hsyn1 (9 * Q K)
    simp only [mem_Icc] at hmI
    have hmIco : m ∈ Ico (9 * Q K) (10 * Q K) := by
      rw [mem_Ico]; exact ⟨hmI.1, by omega⟩
    exact gap_lem h1 hck1 m hmIco hmS

end Erdos741OAI

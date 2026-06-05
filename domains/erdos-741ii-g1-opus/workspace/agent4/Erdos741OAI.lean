import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

theorem Q_pos (k : ℕ) : 0 < Q k := by unfold Q; positivity
theorem Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by unfold Q; ring
theorem Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

theorem lt_Q (n : ℕ) : n < Q n := by
  induction n with
  | zero => norm_num [Q]
  | succ m ih =>
    have hp := Q_pos m
    rw [Q_succ]; omega

theorem setA_cases {x : ℕ} (h : x ∈ setA) :
    x ∈ Icc 2 3 ∨ ∃ j, x ∈ ({ck j} ∪ Bk j ∪ Fk j) := by
  simp only [setA, mem_union] at h
  rcases h with h | h
  · exact Or.inl h
  · rw [mem_iUnion] at h
    exact Or.inr h

theorem stage_lb {j x : ℕ} (h : x ∈ ({ck j} ∪ Bk j ∪ Fk j)) : 4 * Q j ≤ x := by
  have hq := Q_pos j
  simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

theorem stage_ub {j x : ℕ} (h : x ∈ ({ck j} ∪ Bk j ∪ Fk j)) : x ≤ 15 * Q j := by
  have hq := Q_pos j
  simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

theorem setA_ge {x : ℕ} (h : x ∈ setA) : 2 ≤ x := by
  rcases setA_cases h with hIcc | ⟨j, hj⟩
  · rw [mem_Icc] at hIcc; omega
  · have hq := Q_pos j
    have := stage_lb hj
    omega

theorem stage_sub_setA (k : ℕ) : ({ck k} ∪ Bk k ∪ Fk k) ⊆ setA := by
  intro x hx
  exact Or.inr (Set.mem_iUnion.mpr ⟨k, hx⟩)

theorem Akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  show x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  exact Or.inl (Or.inl (Or.inl hx))

theorem Akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero =>
    intro x hx
    exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with ((hx | hx) | hx) | hx
    · exact ih hx
    · exact stage_sub_setA k (Or.inl (Or.inl hx))
    · exact stage_sub_setA k (Or.inl (Or.inr hx))
    · exact stage_sub_setA k (Or.inr hx)

theorem c_mem (k : ℕ) : (4 * Q k) ∈ Akn (k+1) := by
  show (4 * Q k) ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  exact Or.inl (Or.inl (Or.inr rfl))

theorem B_sub (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ Akn (k+1) := by
  intro x hx
  show x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  exact Or.inl (Or.inr hx)

theorem F_sub (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ Akn (k+1) := by
  intro x hx
  show x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  exact Or.inr hx

theorem I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1) := by
  cases k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    have hq0 : Q 0 = 1 := rfl
    show x ∈ Akn 0 ∪ {ck 0} ∪ Bk 0 ∪ Fk 0
    refine Or.inl (Or.inl (Or.inl ?_))
    show x ∈ Icc 2 3
    rw [mem_Icc]; omega
  | succ j =>
    intro x hx
    have hsucc : Q (j+1) = 5 * Q j := Q_succ j
    have hq := Q_pos j
    rw [mem_Icc] at hx
    have hxF : x ∈ Fk j := by
      simp only [Fk, mem_Icc]; omega
    exact Akn_mono (j+1) (F_sub j hxF)

theorem cover (k : ℕ) : Icc (4 * Q k) (30 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  have hq : 0 < Q k := Q_pos k
  by_cases h : x ≤ 5 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 2 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h2 : x ≤ 6 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 3 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h3 : x ≤ 7 * Q k
  · exact Set.mem_add.mpr ⟨4 * Q k, c_mem k,
      x - 4 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h4 : x ≤ 8 * Q k - 1
  · exact Set.mem_add.mpr ⟨2 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 2 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h5 : x ≤ 9 * Q k - 1
  · exact Set.mem_add.mpr ⟨3 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 3 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h6 : x ≤ 10 * Q k - 1
  · exact Set.mem_add.mpr ⟨4 * Q k, c_mem k,
      x - 4 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h7 : x ≤ 11 * Q k - 1
  · exact Set.mem_add.mpr ⟨5 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 5 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h8 : x ≤ 12 * Q k - 2
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - (6 * Q k - 1), B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h9 : x ≤ 17 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 2 * Q k, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h10 : x ≤ 18 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, I_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 3 * Q k, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h11 : x ≤ 20 * Q k
  · exact Set.mem_add.mpr ⟨5 * Q k, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 5 * Q k, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h12 : x ≤ 21 * Q k - 1
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, B_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - (6 * Q k - 1), F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  by_cases h13 : x ≤ 25 * Q k - 1
  · exact Set.mem_add.mpr ⟨10 * Q k - 1, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - (10 * Q k - 1), F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  · exact Set.mem_add.mpr ⟨15 * Q k, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩),
      x - 15 * Q k, F_sub k (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩

theorem P (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    have hq0 : Q 0 = 1 := rfl
    apply cover 0
    rw [mem_Icc]; omega
  | succ k ih =>
    intro x hx
    rw [mem_Icc] at hx
    have hsucc : Q (k+1) = 5 * Q k := Q_succ k
    have hq := Q_pos k
    by_cases hc : x ≤ 6 * Q k
    · exact Set.add_subset_add (Akn_mono (k+1)) (Akn_mono (k+1)) (ih (mem_Icc.mpr ⟨hx.1, hc⟩))
    · push_neg at hc
      exact Set.add_subset_add (Akn_mono (k+1)) (Akn_mono (k+1))
        (cover k (mem_Icc.mpr ⟨by omega, by omega⟩))

theorem key (k : ℕ) {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a ≤ b)
    (h1 : 9 * Q k ≤ a + b) (h2 : a + b < 10 * Q k) :
    a = ck k ∧ b ∈ Bk k := by
  have hqk : 0 < Q k := Q_pos k
  have ha2 : 2 ≤ a := setA_ge ha
  have hbge : 9 * Q k ≤ 2 * b := by omega
  rcases setA_cases hb with hbIcc | ⟨jb, hbj⟩
  · rw [mem_Icc] at hbIcc; omega
  have hjb_le : jb ≤ k := by
    by_contra hgt; push_neg at hgt
    have hmono : 5 * Q k ≤ Q jb := by
      have h := Q_mono (show k + 1 ≤ jb by omega); rw [Q_succ] at h; exact h
    have hlb := stage_lb hbj
    omega
  have hjb_ge : k ≤ jb := by
    by_contra hlt; push_neg at hlt
    have hmono : 5 * Q jb ≤ Q k := by
      have h := Q_mono (show jb + 1 ≤ k by omega); rw [Q_succ] at h; exact h
    have hub := stage_ub hbj
    omega
  have hjbk : jb = k := le_antisymm hjb_le hjb_ge
  rw [hjbk] at hbj
  have hbBk : b ∈ Bk k := by
    simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hbj
    simp only [Bk, mem_Icc]
    rcases hbj with (h | h) | h <;> omega
  have hbB : 5 * Q k ≤ b ∧ b ≤ 6 * Q k - 1 := by
    have h := hbBk; simp only [Bk, mem_Icc] at h; exact h
  rcases setA_cases ha with haIcc | ⟨ja, haj⟩
  · rw [mem_Icc] at haIcc; omega
  have hja_le : ja ≤ k := by
    by_contra hgt; push_neg at hgt
    have hmono : 5 * Q k ≤ Q ja := by
      have h := Q_mono (show k + 1 ≤ ja by omega); rw [Q_succ] at h; exact h
    have hlb := stage_lb haj
    omega
  have hja_ge : k ≤ ja := by
    by_contra hlt; push_neg at hlt
    have hmono : 5 * Q ja ≤ Q k := by
      have h := Q_mono (show ja + 1 ≤ k by omega); rw [Q_succ] at h; exact h
    have hub := stage_ub haj
    omega
  have hjak : ja = k := le_antisymm hja_le hja_ge
  rw [hjak] at haj
  refine ⟨?_, hbBk⟩
  simp only [ck]
  simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at haj
  rcases haj with (h | h) | h <;> omega

theorem rigidity (k : ℕ) {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (h1 : 9 * Q k ≤ a + b) (h2 : a + b < 10 * Q k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  rcases le_total a b with hab | hab
  · exact Or.inl (key k ha hb hab h1 h2)
  · exact Or.inr (key k hb ha hab (by omega) (by omega))

theorem gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n, n ∈ Jk k → n ∉ T + T := by
  intro n hn hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  simp only [Jk, mem_Ico] at hn
  rcases rigidity k (hT ha) (hT hb) (by omega) (by omega) with ⟨hak, _⟩ | ⟨hbk, _⟩
  · exact hck (hak ▸ ha)
  · exact hck (hbk ▸ hb)

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
    obtain ⟨k, hk⟩ : ∃ k, n ≤ 6 * Q k := ⟨n, by have := lt_Q n; omega⟩
    have hmem : n ∈ Akn (k+1) + Akn (k+1) := P k (mem_Icc.mpr ⟨hn, hk⟩)
    rw [Set.mem_add] at hmem
    obtain ⟨a, ha, b, hb, hab⟩ := hmem
    exact ⟨a, Akn_sub_setA _ ha, b, Akn_sub_setA _ hb, hab⟩
  · rintro A₁ A₂ h1 h2 hcov hdisj ⟨⟨C₁, hs1⟩, ⟨C₂, hs2⟩⟩
    obtain ⟨k, hk1, hk2⟩ : ∃ k, C₁ < Q k ∧ C₂ < Q k := by
      have h := lt_Q (C₁ + C₂ + 1)
      exact ⟨C₁ + C₂ + 1, by omega, by omega⟩
    have hckA : ck k ∈ setA := stage_sub_setA k (Or.inl (Or.inl rfl))
    rcases hcov (ck k) hckA with hc1 | hc2
    · have hnotin : ck k ∉ A₂ := by
        intro hmem
        have hi : ck k ∈ A₁ ∩ A₂ := ⟨hc1, hmem⟩
        rw [hdisj] at hi; exact hi
      obtain ⟨m, hmS, hmI⟩ := hs2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      exact gap_lem k A₂ h2 hnotin m hmJ hmS
    · have hnotin : ck k ∉ A₁ := by
        intro hmem
        have hi : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hc2⟩
        rw [hdisj] at hi; exact hi
      obtain ⟨m, hmS, hmI⟩ := hs1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      exact gap_lem k A₁ h1 hnotin m hmJ hmS

end Erdos741OAI

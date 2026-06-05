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

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  simpa [Q] using Nat.pow_le_pow_right (by norm_num : (1:ℕ) ≤ 5) h

/-! ## Stage membership -/

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  refine Or.inr ?_
  refine mem_iUnion.mpr ⟨k, ?_⟩
  exact Or.inl (Or.inl rfl)

lemma stage_bounds {j e : ℕ} (he : e ∈ stage j) : 4 * Q j ≤ e ∧ e ≤ 15 * Q j := by
  have hpos := Q_pos j
  simp only [stage, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at he
  rcases he with (rfl | ⟨h1, h2⟩) | ⟨h1, h2⟩
  · exact ⟨by omega, by omega⟩
  · exact ⟨by omega, by omega⟩
  · exact ⟨by omega, by omega⟩

lemma setA_ge_two {e : ℕ} (he : e ∈ setA) : 2 ≤ e := by
  simp only [setA, mem_union, mem_iUnion] at he
  rcases he with he | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at he; omega
  · obtain ⟨h1, _⟩ := stage_bounds hj
    have := Q_pos j; omega

/-- Any element of `setA` below `10 * Q k` lies in stage `< k` (so `≤ 3 Q k`)
    or in stage `k`. -/
lemma classify (k : ℕ) (hk : 1 ≤ k) (e : ℕ) (he : e ∈ setA) (hle : e < 10 * Q k) :
    e ≤ 3 * Q k ∨ e = ck k ∨ e ∈ Bk k ∨ e ∈ Fk k := by
  have hQ := Q_pos k
  simp only [setA, mem_union, mem_iUnion] at he
  rcases he with he | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at he
    left; omega
  · obtain ⟨hb1, hb2⟩ := stage_bounds hj
    rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have h1 : Q (j+1) ≤ Q k := Q_mono (by omega)
      have h2 : Q (j+1) = 5 * Q j := Q_succ j
      omega
    · rw [hje] at hj
      simp only [stage, mem_union, mem_singleton_iff] at hj
      rcases hj with (h | h) | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr (Or.inl h))
      · exact Or.inr (Or.inr (Or.inr h))
    · exfalso
      have h1 : Q (k+1) ≤ Q j := Q_mono (by omega)
      have h2 : Q (k+1) = 5 * Q k := Q_succ k
      omega

lemma akn_sub_setA : ∀ m, Akn m ⊆ setA := by
  intro m
  induction m with
  | zero =>
    intro x hx
    exact Or.inl hx
  | succ k ih =>
    intro x hx
    simp only [Akn, mem_union] at hx
    rcases hx with ((hx | hx) | hx) | hx
    · exact ih hx
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl hx)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inr hx)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inr hx⟩)

/-! ## Akn membership helpers -/

lemma akn_succ_sub (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inl (Or.inl (Or.inl hx))

lemma ck_mem_akn (k : ℕ) : ck k ∈ Akn (k+1) := by
  simp only [Akn, mem_union, mem_singleton_iff]
  exact Or.inl (Or.inl (Or.inr trivial))

lemma Bk_sub_akn (k : ℕ) : Bk k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inl (Or.inr hx)

lemma Fk_sub_akn (k : ℕ) : Fk k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inr hx

/-! ## Eight-pair coverage -/

lemma cover_pairs (k : ℕ) (D : Set ℕ)
    (hI : Icc (2 * Q k) (3 * Q k) ⊆ D)
    (hck : ck k ∈ D)
    (hB : Bk k ⊆ D)
    (hF : Fk k ⊆ D)
    (x : ℕ) (hx : x ∈ Icc (4 * Q k) (30 * Q k)) :
    ∃ a ∈ D, ∃ b ∈ D, a + b = x := by
  have hq := Q_pos k
  simp only [mem_Icc] at hx
  obtain ⟨hxl, hxr⟩ := hx
  have hckD : (4 * Q k) ∈ D := hck
  by_cases h1 : x ≤ 5 * Q k
  · exact ⟨2 * Q k, hI (by simp only [mem_Icc]; omega),
           x - 2 * Q k, hI (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h2 : x ≤ 6 * Q k
  · exact ⟨3 * Q k, hI (by simp only [mem_Icc]; omega),
           x - 3 * Q k, hI (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h3 : x ≤ 7 * Q k
  · exact ⟨4 * Q k, hckD,
           x - 4 * Q k, hI (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h4 : x ≤ 8 * Q k
  · exact ⟨5 * Q k, hB (by simp only [Bk, mem_Icc]; omega),
           x - 5 * Q k, hI (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h5 : x ≤ 9 * Q k - 1
  · exact ⟨3 * Q k, hI (by simp only [mem_Icc]; omega),
           x - 3 * Q k, hB (by simp only [Bk, mem_Icc]; omega), by omega⟩
  by_cases h6 : x ≤ 10 * Q k - 1
  · exact ⟨4 * Q k, hckD,
           x - 4 * Q k, hB (by simp only [Bk, mem_Icc]; omega), by omega⟩
  by_cases h7 : x ≤ 11 * Q k - 1
  · exact ⟨5 * Q k, hB (by simp only [Bk, mem_Icc]; omega),
           x - 5 * Q k, hB (by simp only [Bk, mem_Icc]; omega), by omega⟩
  by_cases h8 : x ≤ 12 * Q k - 2
  · exact ⟨6 * Q k - 1, hB (by simp only [Bk, mem_Icc]; omega),
           x - (6 * Q k - 1), hB (by simp only [Bk, mem_Icc]; omega), by omega⟩
  by_cases h9 : x ≤ 13 * Q k - 1
  · exact ⟨10 * Q k - 1, hF (by simp only [Fk, mem_Icc]; omega),
           x - (10 * Q k - 1), hI (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h10 : x ≤ 18 * Q k
  · exact ⟨3 * Q k, hI (by simp only [mem_Icc]; omega),
           x - 3 * Q k, hF (by simp only [Fk, mem_Icc]; omega), by omega⟩
  by_cases h11 : x ≤ 20 * Q k
  · exact ⟨5 * Q k, hB (by simp only [Bk, mem_Icc]; omega),
           x - 5 * Q k, hF (by simp only [Fk, mem_Icc]; omega), by omega⟩
  by_cases h12 : x ≤ 25 * Q k - 1
  · exact ⟨10 * Q k - 1, hF (by simp only [Fk, mem_Icc]; omega),
           x - (10 * Q k - 1), hF (by simp only [Fk, mem_Icc]; omega), by omega⟩
  · exact ⟨15 * Q k, hF (by simp only [Fk, mem_Icc]; omega),
           x - 15 * Q k, hF (by simp only [Fk, mem_Icc]; omega), by omega⟩

/-! ## Basis -/

lemma basis_aux : ∀ k, ∀ x, x ∈ Icc 4 (30 * Q k) →
    ∃ a ∈ Akn (k+1), ∃ b ∈ Akn (k+1), a + b = x := by
  intro k
  induction k with
  | zero =>
    intro x hx
    have hx' : x ∈ Icc (4 * Q 0) (30 * Q 0) := by simpa [Q] using hx
    refine cover_pairs 0 (Akn 1) ?_ (ck_mem_akn 0) (Bk_sub_akn 0) (Fk_sub_akn 0) x hx'
    intro y hy
    simp only [Q, pow_zero, mul_one, mem_Icc] at hy
    refine akn_succ_sub 0 ?_
    simp only [Akn, mem_insert_iff, mem_singleton_iff]
    omega
  | succ k ih =>
    intro x hx
    simp only [mem_Icc] at hx
    obtain ⟨hxl, hxr⟩ := hx
    have hqs : Q (k+1) = 5 * Q k := Q_succ k
    by_cases hsplit : x ≤ 6 * Q (k+1)
    · have hxk : x ∈ Icc 4 (30 * Q k) := by
        simp only [mem_Icc]; exact ⟨hxl, by omega⟩
      obtain ⟨a, ha, b, hb, hab⟩ := ih x hxk
      exact ⟨a, akn_succ_sub (k+1) ha, b, akn_succ_sub (k+1) hb, hab⟩
    · have hx' : x ∈ Icc (4 * Q (k+1)) (30 * Q (k+1)) := by
        simp only [mem_Icc]; exact ⟨by omega, hxr⟩
      refine cover_pairs (k+1) (Akn (k+2)) ?_ (ck_mem_akn (k+1))
        (Bk_sub_akn (k+1)) (Fk_sub_akn (k+1)) x hx'
      intro y hy
      simp only [mem_Icc] at hy
      refine akn_succ_sub (k+1) (Fk_sub_akn k ?_)
      simp only [Fk, mem_Icc]; omega

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have hlt : n < 5 ^ n := Nat.lt_pow_self (by norm_num)
  have hmem : n ∈ Icc 4 (30 * Q n) := by
    simp only [mem_Icc, Q]
    exact ⟨hn, by omega⟩
  obtain ⟨a, ha, b, hb, hab⟩ := basis_aux n n hmem
  exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩

/-! ## Rigidity -/

lemma rigidity (k : ℕ) (hk : 1 ≤ k) (n : ℕ) (hn : n ∈ Jk k)
    (a : ℕ) (ha : a ∈ setA) (b : ℕ) (hb : b ∈ setA) (hab : a + b = n) :
    a = ck k ∨ b = ck k := by
  have hQ := Q_pos k
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have ha_le : a < 10 * Q k := by omega
  have hb_le : b < 10 * Q k := by omega
  have hca := classify k hk a ha ha_le
  have hcb := classify k hk b hb hb_le
  rcases hca with hS | hck | hB | hF
  · rcases hcb with hS' | hck' | hB' | hF'
    · exfalso; omega
    · exact Or.inr hck'
    · exfalso; simp only [Bk, mem_Icc] at hB'; omega
    · exfalso; simp only [Fk, mem_Icc] at hF'; omega
  · exact Or.inl hck
  · simp only [Bk, mem_Icc] at hB
    rcases hcb with hS' | hck' | hB' | hF'
    · exfalso; omega
    · exact Or.inr hck'
    · exfalso; simp only [Bk, mem_Icc] at hB'; omega
    · exfalso; simp only [Fk, mem_Icc] at hF'; omega
  · simp only [Fk, mem_Icc] at hF
    rcases hcb with hS' | hck' | hB' | hF'
    · exfalso; omega
    · exact Or.inr hck'
    · exfalso; simp only [Bk, mem_Icc] at hB'; omega
    · exfalso; simp only [Fk, mem_Icc] at hF'; omega

/-! ## Gap -/

lemma gap_lem (k : ℕ) (hk : 1 ≤ k) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ m, m ∈ Jk k → m ∉ T + T := by
  intro m hm hmem
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmem
  have ha' := hT ha
  have hb' := hT hb
  rcases rigidity k hk m hm a ha' b hb' hab with h | h
  · exact hck (h ▸ ha)
  · exact hck (h ▸ hb)

/-! ## Main theorem -/

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, basis_lem, ?_⟩
  rintro A₁ A₂ h1 h2 hcov hdisj ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
  -- pick k with Q k > C₁, C₂ and k ≥ 1
  set k := C₁ + C₂ + 1 with hk_def
  have hk1 : 1 ≤ k := by omega
  have hkbig : C₁ + C₂ + 1 < Q k := by
    have : k < 5 ^ k := Nat.lt_pow_self (by norm_num)
    simpa [Q, hk_def] using this
  have hC1lt : C₁ < Q k := by omega
  have hC2lt : C₂ < Q k := by omega
  -- ck k is in A, goes to one side
  have hckA : ck k ∈ setA := ck_mem_setA k
  rcases hcov (ck k) hckA with hc1 | hc2
  · -- ck k ∈ A₁, so ck k ∉ A₂
    have hcknot : ck k ∉ A₂ := by
      intro h
      have : ck k ∈ A₁ ∩ A₂ := ⟨hc1, h⟩
      rw [hdisj] at this
      exact this
    -- A₂ + A₂ syndetic with C₂ misses Jk k
    obtain ⟨m, hmem, hmIcc⟩ := hC₂ (9 * Q k)
    have hmJ : m ∈ Jk k := by
      simp only [mem_Icc] at hmIcc
      simp only [Jk, mem_Ico]
      exact ⟨hmIcc.1, by omega⟩
    exact gap_lem k hk1 A₂ h2 hcknot m hmJ hmem
  · -- ck k ∈ A₂, so ck k ∉ A₁
    have hcknot : ck k ∉ A₁ := by
      intro h
      have : ck k ∈ A₁ ∩ A₂ := ⟨h, hc2⟩
      rw [hdisj] at this
      exact this
    obtain ⟨m, hmem, hmIcc⟩ := hC₁ (9 * Q k)
    have hmJ : m ∈ Jk k := by
      simp only [mem_Icc] at hmIcc
      simp only [Jk, mem_Ico]
      exact ⟨hmIcc.1, by omega⟩
    exact gap_lem k hk1 A₁ h1 hcknot m hmJ hmem

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Q k = 5^k -/
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def Stg (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, Stg k

/-! ### Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {m n : ℕ} (h : m ≤ n) : Q m ≤ Q n :=
  Nat.pow_le_pow_right (by norm_num) h

/-- For `j < k`, the next power is already ≤ Q k, giving geometric separation. -/
lemma Q_5_le {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have : Q (j + 1) ≤ Q k := Q_mono h
  rwa [Q_succ] at this

/-! ### Membership helpers -/

lemma ck_in_stg (k : ℕ) : ck k ∈ Stg k := by
  unfold Stg; exact Or.inl (Or.inl rfl)

lemma Bk_in_stg {x k : ℕ} (h : x ∈ Bk k) : x ∈ Stg k := by
  unfold Stg; exact Or.inl (Or.inr h)

lemma Fk_in_stg {x k : ℕ} (h : x ∈ Fk k) : x ∈ Stg k := by
  unfold Stg; exact Or.inr h

lemma stg_mem_setA {x k : ℕ} (h : x ∈ Stg k) : x ∈ setA := by
  unfold setA; exact Or.inr (mem_iUnion.mpr ⟨k, h⟩)

lemma two_mem : (2 : ℕ) ∈ setA := by
  unfold setA; exact Or.inl (by simp)

lemma three_mem : (3 : ℕ) ∈ setA := by
  unfold setA; exact Or.inl (by simp)

lemma ck_mem (k : ℕ) : ck k ∈ setA := stg_mem_setA (ck_in_stg k)

lemma Bk_mem {x k : ℕ} (h : x ∈ Bk k) : x ∈ setA := stg_mem_setA (Bk_in_stg h)

lemma Fk_mem {x k : ℕ} (h : x ∈ Fk k) : x ∈ setA := stg_mem_setA (Fk_in_stg h)

/-- Every stage-element lies in `[4 Q j, 15 Q j]`. -/
lemma stg_bound {x j : ℕ} (h : x ∈ Stg j) : 4 * Q j ≤ x ∧ x ≤ 15 * Q j := by
  have hq := Q_pos j
  simp only [Stg, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at h
  omega

/-- Decompose membership in `setA`. -/
lemma mem_setA {x : ℕ} (h : x ∈ setA) : x = 2 ∨ x = 3 ∨ ∃ j, x ∈ Stg j := by
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff] at h
  tauto

/-! ### The basis property -/

def repr (n : ℕ) : Prop := ∃ a ∈ setA, ∃ b ∈ setA, a + b = n

/-- Generic "covering by a pair of blocks" lemma. -/
lemma cover_pair {xlo xhi ylo yhi n : ℕ}
    (hX : ∀ z, xlo ≤ z → z ≤ xhi → z ∈ setA)
    (hY : ∀ z, ylo ≤ z → z ≤ yhi → z ∈ setA)
    (hxx : xlo ≤ xhi) (hyy : ylo ≤ yhi)
    (h1 : xlo + ylo ≤ n) (h2 : n ≤ xhi + yhi) : repr n := by
  by_cases hc : n ≤ xhi + ylo
  · exact ⟨n - ylo, hX _ (by omega) (by omega), ylo, hY _ (le_refl _) hyy, by omega⟩
  · push_neg at hc
    exact ⟨xhi, hX _ hxx (le_refl _), n - xhi, hY _ (by omega) (by omega), by omega⟩

lemma ck_prov (k : ℕ) : ∀ z, 4 * Q k ≤ z → z ≤ 4 * Q k → z ∈ setA :=
  fun z h1 h2 => by
    have : z = ck k := by unfold ck; omega
    rw [this]; exact ck_mem k

lemma Bk_prov (k : ℕ) : ∀ z, 5 * Q k ≤ z → z ≤ 6 * Q k - 1 → z ∈ setA :=
  fun z h1 h2 => Bk_mem (mem_Icc.mpr ⟨h1, h2⟩)

lemma Fk_prov (k : ℕ) : ∀ z, 10 * Q k - 1 ≤ z → z ≤ 15 * Q k → z ∈ setA :=
  fun z h1 h2 => Fk_mem (mem_Icc.mpr ⟨h1, h2⟩)

lemma twothree_prov : ∀ z, 2 ≤ z → z ≤ 3 → z ∈ setA :=
  fun z h1 h2 => by
    interval_cases z
    · exact two_mem
    · exact three_mem

/-- Base layer: `[4,30]` is covered. -/
lemma base_cover (n : ℕ) (h1 : 4 ≤ n) (h2 : n ≤ 30) : repr n := by
  have hq0 : Q 0 = 1 := by norm_num [Q]
  by_cases c1 : n ≤ 6
  · exact cover_pair twothree_prov twothree_prov (by omega) (by omega) (by omega) (by omega)
  push_neg at c1
  by_cases c2 : n ≤ 7
  · exact cover_pair twothree_prov (ck_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c2
  by_cases c3 : n ≤ 8
  · exact cover_pair twothree_prov (Bk_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c3
  by_cases c4 : n ≤ 9
  · exact cover_pair (ck_prov 0) (Bk_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c4
  by_cases c5 : n ≤ 10
  · exact cover_pair (Bk_prov 0) (Bk_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c5
  by_cases c6 : n ≤ 18
  · exact cover_pair twothree_prov (Fk_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c6
  by_cases c7 : n ≤ 20
  · exact cover_pair (Bk_prov 0) (Fk_prov 0) (by omega) (by omega) (by omega) (by omega)
  push_neg at c7
  exact cover_pair (Fk_prov 0) (Fk_prov 0) (by omega) (by omega) (by omega) (by omega)

/-- Inductive layer: `[4 Q(k+1), 30 Q(k+1)]` covered using I = Fk k and level-(k+1) blocks. -/
lemma level_cover (k : ℕ) (n : ℕ) (h1 : 4 * Q (k+1) ≤ n) (h2 : n ≤ 30 * Q (k+1)) :
    repr n := by
  have hs := Q_succ k
  have hq := Q_pos k
  by_cases c1 : n ≤ 30 * Q k
  · exact cover_pair (Fk_prov k) (Fk_prov k) (by omega) (by omega) (by omega) (by omega)
  push_neg at c1
  by_cases c2 : n ≤ 35 * Q k
  · exact cover_pair (Fk_prov k) (ck_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c2
  by_cases c3 : n ≤ 45 * Q k - 1
  · exact cover_pair (Fk_prov k) (Bk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c3
  by_cases c4 : n ≤ 50 * Q k - 1
  · exact cover_pair (ck_prov (k+1)) (Bk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c4
  by_cases c5 : n ≤ 60 * Q k - 2
  · exact cover_pair (Bk_prov (k+1)) (Bk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c5
  by_cases c6 : n ≤ 90 * Q k
  · exact cover_pair (Fk_prov k) (Fk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c6
  by_cases c7 : n ≤ 105 * Q k - 1
  · exact cover_pair (Bk_prov (k+1)) (Fk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)
  push_neg at c7
  exact cover_pair (Fk_prov (k+1)) (Fk_prov (k+1)) (by omega) (by omega) (by omega) (by omega)

lemma basis_cover (k : ℕ) : ∀ n, 4 ≤ n → n ≤ 6 * Q (k+1) → repr n := by
  induction k with
  | zero =>
    intro n h1 h2
    have hq1 : Q (0 + 1) = 5 := by norm_num [Q]
    exact base_cover n h1 (by omega)
  | succ k ih =>
    intro n h1 h2
    by_cases hc : n ≤ 6 * Q (k+1)
    · exact ih n h1 hc
    · push_neg at hc
      have hs := Q_succ (k+1)
      exact level_cover k n (by omega) (by omega)

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ k ih =>
    have hs := Q_succ k
    have hq := Q_pos k
    omega

lemma basis (n : ℕ) (hn : 4 ≤ n) : repr n := by
  have h1 := n_le_Q n
  have h2 : Q n ≤ Q (n + 1) := Q_mono (Nat.le_succ n)
  exact basis_cover n n hn (by omega)

/-! ### Rigidity and the gap lemma -/

lemma stg_cases {x k : ℕ} (h : x ∈ Stg k) :
    x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ (10 * Q k - 1 ≤ x ∧ x ≤ 15 * Q k) := by
  simp only [Stg, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at h
  tauto

/-- Any representation `a + b = n` with `n ∈ Jk k` forces one summand to be `ck k`. -/
lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    a = ck k ∨ b = ck k := by
  rw [show ck k = 4 * Q k from rfl]
  have hq := Q_pos k
  rw [Jk, mem_Ico] at hn
  obtain ⟨h9, h10⟩ := hn
  rcases mem_setA ha with rfl | rfl | ⟨i, hi⟩
  · -- a = 2
    rcases mem_setA hb with rfl | rfl | ⟨j, hj⟩
    · omega
    · omega
    · obtain ⟨hbl, hbh⟩ := stg_bound hj
      rcases lt_trichotomy j k with hlt | hje | hgt
      · have := Q_5_le hlt; omega
      · rw [hje] at hj; have hc := stg_cases hj; omega
      · have := Q_5_le hgt; omega
  · -- a = 3
    rcases mem_setA hb with rfl | rfl | ⟨j, hj⟩
    · omega
    · omega
    · obtain ⟨hbl, hbh⟩ := stg_bound hj
      rcases lt_trichotomy j k with hlt | hje | hgt
      · have := Q_5_le hlt; omega
      · rw [hje] at hj; have hc := stg_cases hj; omega
      · have := Q_5_le hgt; omega
  · -- a staged i
    obtain ⟨hal, hah⟩ := stg_bound hi
    rcases mem_setA hb with rfl | rfl | ⟨j, hj⟩
    · -- b = 2
      rcases lt_trichotomy i k with hlt | hie | hgt
      · have := Q_5_le hlt; omega
      · rw [hie] at hi; have hc := stg_cases hi; omega
      · have := Q_5_le hgt; omega
    · -- b = 3
      rcases lt_trichotomy i k with hlt | hie | hgt
      · have := Q_5_le hlt; omega
      · rw [hie] at hi; have hc := stg_cases hi; omega
      · have := Q_5_le hgt; omega
    · -- both staged
      obtain ⟨hbl, hbh⟩ := stg_bound hj
      have hqi := Q_pos i; have hqj := Q_pos j
      rcases lt_trichotomy i k with hilt | hije | higt
      · rcases lt_trichotomy j k with hjlt | hjje | hjgt
        · have := Q_5_le hilt; have := Q_5_le hjlt; omega
        · have hi5 := Q_5_le hilt; rw [hjje] at hj; have hc := stg_cases hj; omega
        · have := Q_5_le hjgt; omega
      · rw [hije] at hi; have hca := stg_cases hi
        rcases lt_trichotomy j k with hjlt | hjje | hjgt
        · have hj5 := Q_5_le hjlt; omega
        · rw [hjje] at hj; have hcb := stg_cases hj; omega
        · have := Q_5_le hjgt; omega
      · have := Q_5_le higt; omega

/-- If `ck k ∉ T ⊆ setA`, then the gap zone `Jk k` is disjoint from `T + T`. -/
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n ∈ Jk k, n ∉ T + T := by
  intro n hn hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  rcases rigidity k n hn a b (hT ha) (hT hb) hab with h | h
  · rw [h] at ha; exact hck ha
  · rw [h] at hb; exact hck hb

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn; exact basis n hn
  · intro A₁ A₂ h1sub h2sub hcover hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    set k := C₁ + C₂ + 1 with hk
    have hkle := n_le_Q k
    have hC1k : C₁ < Q k := by omega
    have hC2k : C₂ < Q k := by omega
    have hckA : ck k ∈ setA := ck_mem k
    rcases hcover (ck k) hckA with hA1 | hA2
    · have hnotA2 : ck k ∉ A₂ := by
        intro hcon
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hA1, hcon⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmAdd, hmIcc⟩ := hC₂ (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by rw [Jk, mem_Ico]; omega
      exact gap_lem k A₂ h2sub hnotA2 m hmJ hmAdd
    · have hnotA1 : ck k ∉ A₁ := by
        intro hcon
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hcon, hA2⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmAdd, hmIcc⟩ := hC₁ (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by rw [Jk, mem_Ico]; omega
      exact gap_lem k A₁ h1sub hnotA1 m hmJ hmAdd

end Erdos741OAI

import Mathlib
set_option maxHeartbeats 800000
set_option maxRecDepth 1000
open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5^k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by unfold Q; ring

lemma akn_mono {j k : ℕ} (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction h with
  | refl => exact Subset.refl _
  | @step n _ ih => exact Subset.trans ih (fun x hx => Or.inl (Or.inl (Or.inl hx)))

lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with (((h | h) | h) | h)
    · exact ih h
    · exact Or.inr ⟨k, Or.inl (Or.inl h)⟩
    · exact Or.inr ⟨k, Or.inl (Or.inr h)⟩
    · exact Or.inr ⟨k, Or.inr h⟩

-- I interval is in Akn(k+1)
lemma Ik_subset (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, Nat.mul_one, mem_Icc] at hx
    simp only [Akn, mem_insert_iff, mem_singleton_iff]
    omega
  | succ k =>
    intro x hx
    simp only [mem_Icc, Q_succ] at hx
    apply @akn_mono (k + 1) (k + 2) (by omega)
    exact Or.inr (by
      simp only [Fk, mem_Icc, Q_succ]
      constructor <;> linarith [Q_pos k])

-- Stage bounds
lemma stage_upper {j x : ℕ} (hx : x ∈ ({ck j} : Set ℕ) ∪ Bk j ∪ Fk j) : x ≤ 15 * Q j := by
  rcases hx with ((hx | hx) | hx)
  · simp only [mem_singleton_iff] at hx; rw [hx, ck]; linarith [Q_pos j]
  · simp only [Bk, mem_Icc] at hx; linarith [hx.2, Q_pos j]
  · simp only [Fk, mem_Icc] at hx; exact hx.2

lemma stage_lower {j x : ℕ} (hx : x ∈ ({ck j} : Set ℕ) ∪ Bk j ∪ Fk j) : 4 * Q j ≤ x := by
  rcases hx with ((hx | hx) | hx)
  · simp only [mem_singleton_iff] at hx; rw [hx, ck]
  · simp only [Bk, mem_Icc] at hx; linarith [hx.1]
  · simp only [Fk, mem_Icc] at hx; linarith [hx.1, Q_pos j]

lemma stage_upper_small {j k : ℕ} (hjk : j < k) : 15 * Q j ≤ 3 * Q k := by
  have h : Q (j + 1) ≤ Q k := Nat.pow_le_pow_right (by norm_num) hjk
  rw [Q_succ] at h; linarith

lemma stage_lower_large {j k : ℕ} (hjk : k < j) : 20 * Q k ≤ 4 * Q j := by
  have h : Q (k + 1) ≤ Q j := Nat.pow_le_pow_right (by norm_num) hjk
  rw [Q_succ] at h; linarith

-- Q grows faster than n
lemma n_lt_Q_succ (n : ℕ) : n < Q (n + 1) := by
  induction n with
  | zero => simp [Q]
  | succ n ih =>
    have hq := Q_succ (n + 1)
    linarith [Q_pos (n + 1)]

-- Pairs lemma: [4*Qk, 30*Qk] ⊆ Akn(k+1) + Akn(k+1)
lemma pairs_lem (k : ℕ) (x : ℕ) (hx1 : 4 * Q k ≤ x) (hx2 : x ≤ 30 * Q k) :
    x ∈ Akn (k + 1) + Akn (k + 1) := by
  have hQpos : 0 < Q k := Q_pos k
  set Qk := Q k with hQk
  -- Key subsets of Akn(k+1)
  have hI : ∀ y, 2 * Qk ≤ y → y ≤ 3 * Qk → y ∈ Akn (k + 1) :=
    fun y h1 h2 => Ik_subset k (mem_Icc.mpr ⟨h1, h2⟩)
  have hck : ck k ∈ Akn (k + 1) := Or.inl (Or.inl (Or.inr rfl))
  have hBk : ∀ y, 5 * Qk ≤ y → y ≤ 6 * Qk - 1 → y ∈ Akn (k + 1) := fun y h1 h2 =>
    Or.inl (Or.inr (by simp only [Bk, mem_Icc]; omega))
  have hFk : ∀ y, 10 * Qk - 1 ≤ y → y ≤ 15 * Qk → y ∈ Akn (k + 1) := fun y h1 h2 =>
    Or.inr (by simp only [Fk, mem_Icc]; omega)
  have hck_val : ck k = 4 * Qk := rfl
  -- 13-band case split
  simp only [Set.mem_add]
  by_cases h1 : x ≤ 5 * Qk
  · exact ⟨x - 2 * Qk, hI _ (by omega) (by omega), 2 * Qk, hI _ (by omega) (by omega), by omega⟩
  push_neg at h1
  by_cases h2 : x ≤ 6 * Qk
  · exact ⟨x - 3 * Qk, hI _ (by omega) (by omega), 3 * Qk, hI _ (by omega) (by omega), by omega⟩
  push_neg at h2
  by_cases h3 : x ≤ 7 * Qk
  · -- I + ck: a = x - 4Qk ∈ I, b = ck k
    exact ⟨x - 4 * Qk, hI _ (by omega) (by omega), ck k,
      hck, by rw [hck_val]; omega⟩
  push_neg at h3
  by_cases h4 : x ≤ 8 * Qk
  · exact ⟨x - 5 * Qk, hI _ (by omega) (by omega), 5 * Qk, hBk _ (by omega) (by omega), by omega⟩
  push_neg at h4
  by_cases h5 : x ≤ 9 * Qk - 1
  · exact ⟨x - (6 * Qk - 1), hI _ (by omega) (by omega), 6 * Qk - 1, hBk _ (by omega) (by omega), by omega⟩
  push_neg at h5
  by_cases h6 : x ≤ 10 * Qk - 1
  · -- ck + Bk: a = ck k, b = x - 4Qk ∈ Bk
    exact ⟨ck k, hck, x - 4 * Qk, hBk _ (by rw [hck_val] at *; omega) (by rw [hck_val] at *; omega),
      by rw [hck_val]; omega⟩
  push_neg at h6
  by_cases h7 : x ≤ 11 * Qk - 1
  · exact ⟨5 * Qk, hBk _ (by omega) (by omega), x - 5 * Qk, hBk _ (by omega) (by omega), by omega⟩
  push_neg at h7
  by_cases h8 : x ≤ 12 * Qk - 2
  · exact ⟨x - (6 * Qk - 1), hBk _ (by omega) (by omega), 6 * Qk - 1, hBk _ (by omega) (by omega), by omega⟩
  push_neg at h8
  by_cases h9 : x ≤ 13 * Qk - 1
  · exact ⟨2 * Qk, hI _ (by omega) (by omega), x - 2 * Qk, hFk _ (by omega) (by omega), by omega⟩
  push_neg at h9
  by_cases h10 : x ≤ 18 * Qk
  · exact ⟨3 * Qk, hI _ (by omega) (by omega), x - 3 * Qk, hFk _ (by omega) (by omega), by omega⟩
  push_neg at h10
  by_cases h11 : x ≤ 20 * Qk
  · exact ⟨5 * Qk, hBk _ (by omega) (by omega), x - 5 * Qk, hFk _ (by omega) (by omega), by omega⟩
  push_neg at h11
  by_cases h12 : x ≤ 21 * Qk - 1
  · exact ⟨6 * Qk - 1, hBk _ (by omega) (by omega), x - (6 * Qk - 1), hFk _ (by omega) (by omega), by omega⟩
  push_neg at h12
  by_cases h13 : x ≤ 25 * Qk
  · exact ⟨x - 10 * Qk, hFk _ (by omega) (by omega), 10 * Qk, hFk _ (by omega) (by omega), by omega⟩
  push_neg at h13
  · exact ⟨x - 15 * Qk, hFk _ (by omega) (by omega), 15 * Qk, hFk _ (by omega) (by omega), by omega⟩

-- Basis lemma: Icc 4 (6*Qk) ⊆ Akn(k+1) + Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, Nat.mul_one, mem_Icc] at hx
    -- x ∈ [4,6], Akn 1 ⊇ {2,3,4,5}
    have h2 : (2 : ℕ) ∈ Akn 1 := @akn_mono 0 1 (by omega) (by simp [Akn])
    have h3 : (3 : ℕ) ∈ Akn 1 := @akn_mono 0 1 (by omega) (by simp [Akn]; right; rfl)
    simp only [Set.mem_add]
    interval_cases x
    · exact ⟨2, h2, 2, h2, rfl⟩
    · exact ⟨2, h2, 3, h3, rfl⟩
    · exact ⟨3, h3, 3, h3, rfl⟩
  | succ k ih =>
    intro x hx
    simp only [mem_Icc, Q_succ] at hx
    have hmono : Akn (k + 1) + Akn (k + 1) ⊆ Akn (k + 2) + Akn (k + 2) :=
      Set.add_subset_add (@akn_mono (k+1) (k+2) (by omega)) (@akn_mono (k+1) (k+2) (by omega))
    by_cases hle : x ≤ 6 * Q k
    · exact hmono (ih (mem_Icc.mpr ⟨hx.1, hle⟩))
    · push_neg at hle
      exact hmono (pairs_lem k x (by linarith [Q_pos k]) (by linarith [hx.2]))

-- Every n ≥ 4 is in setA + setA
lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  -- Pick k = n; n < Q(n+1) ≤ 6*Q(n+1) gives n ∈ Icc 4 (6*Q n)
  have hlt : n < Q (n + 1) := n_lt_Q_succ n
  have hn_in : n ∈ Icc 4 (6 * Q n) := by
    simp only [mem_Icc]
    constructor
    · exact hn
    · linarith [Q_succ n, Q_pos n]
  have := basis_lem n hn_in
  simp only [Set.mem_add] at this
  obtain ⟨a, ha, b, hb, hab⟩ := this
  exact ⟨a, akn_subset_setA (n+1) ha, b, akn_subset_setA (n+1) hb, hab⟩

-- Rigidity lemma: if n ∈ Jk k and a+b=n with a,b ∈ setA, then one is ck k and other in Bk k
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  have hQpos : 0 < Q k := Q_pos k
  have ha_le_n : a ≤ n := Nat.le_of_add_le_add_right (by linarith [hab.symm.le])
  -- Actually a ≤ n since a + b = n and b ≥ 0 (nat)
  have ha_le_n : a ≤ n := by omega
  simp only [setA, mem_union, mem_iUnion] at ha hb
  rcases ha with ha | ⟨j, haj⟩ <;> rcases hb with hb | ⟨i, hbi⟩
  · -- Both in {2,3}: sum ≤ 6 < 9 ≤ 9*Qk. Contradiction.
    simp only [mem_insert_iff, mem_singleton_iff] at ha hb
    linarith [hn.1]
  · -- a ∈ {2,3}, b ∈ stage i
    simp only [mem_insert_iff, mem_singleton_iff] at ha
    have ha3 : a ≤ 3 := by omega
    rcases lt_trichotomy i k with hlt | hje | hgt
    · -- i < k: b ≤ 3*Qk, so a+b ≤ 3+3Qk < 9Qk. Contradiction.
      have hbup : b ≤ 3 * Q k := le_trans (stage_upper hbi) (stage_upper_small hlt)
      linarith [hn.1]
    · -- i = k: b ∈ stage k
      rw [hje] at hbi
      rcases hbi with ((hbi | hbi) | hbi)
      · simp only [mem_singleton_iff] at hbi
        -- b = ck k = 4*Qk, a ≤ 3: a+b = a+4Qk ≤ 3+4Qk < 9Qk. Contradiction.
        rw [hbi] at hab; simp only [ck] at hab; linarith [hn.1]
      · simp only [Bk, mem_Icc] at hbi; linarith [hbi.2, hn.1]
      · simp only [Fk, mem_Icc] at hbi
        -- b ≥ 10Qk-1, n < 10Qk → n ≤ 10Qk-1 → b ≤ n ≤ 10Qk-1
        -- a + b = n ≥ 9Qk, a ≤ 3, b ≥ 10Qk-1 → b ≤ n < 10Qk
        -- b ≥ 10Qk-1 and b ≤ n ≤ 10Qk-1, so b = 10Qk-1, a = 0. But a ≥ 2. Contradiction.
        have hble : b ≤ n := by omega
        linarith [hbi.1, hn.2, hn.1]
    · -- i > k: b ≥ 20*Qk > n. But b ≤ n. Contradiction.
      have hbdown : 20 * Q k ≤ b := le_trans (stage_lower_large hgt) (stage_lower hbi)
      linarith [hn.2]
  · -- a ∈ stage j, b ∈ {2,3}: symmetric to case 2
    simp only [mem_insert_iff, mem_singleton_iff] at hb
    have hb3 : b ≤ 3 := by omega
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have haup : a ≤ 3 * Q k := le_trans (stage_upper haj) (stage_upper_small hlt)
      linarith [hn.1]
    · rw [hje] at haj
      rcases haj with ((haj | haj) | haj)
      · simp only [mem_singleton_iff] at haj
        rw [haj] at hab; simp only [ck] at hab; linarith [hn.1]
      · simp only [Bk, mem_Icc] at haj; linarith [haj.2, hn.1]
      · simp only [Fk, mem_Icc] at haj
        have hale : a ≤ n := by omega
        linarith [haj.1, hn.2, hn.1]
    · have hadown : 20 * Q k ≤ a := le_trans (stage_lower_large hgt) (stage_lower haj)
      linarith [hn.2, ha_le_n]
  · -- a ∈ stage j, b ∈ stage i
    rcases lt_trichotomy j k with hjlt | hjje | hjgt
    · -- j < k: a ≤ 3Qk, b ≥ 6Qk
      have haup : a ≤ 3 * Q k := le_trans (stage_upper haj) (stage_upper_small hjlt)
      have hbdown_val : 6 * Q k ≤ b := by linarith [hn.1]
      rcases lt_trichotomy i k with hilt | hije | higt
      · -- i < k: b ≤ 3Qk < 6Qk. Contradiction.
        have hbup : b ≤ 3 * Q k := le_trans (stage_upper hbi) (stage_upper_small hilt)
        linarith
      · -- i = k
        rw [hije] at hbi
        rcases hbi with ((hbi | hbi) | hbi)
        · simp only [mem_singleton_iff] at hbi
          -- b = ck k = 4Qk < 6Qk ≤ b. Contradiction.
          rw [hbi]; simp only [ck]; linarith
        · simp only [Bk, mem_Icc] at hbi
          -- b ≤ 6Qk-1 < 6Qk ≤ b. Contradiction.
          linarith [hbi.2]
        · simp only [Fk, mem_Icc] at hbi
          -- b ≥ 10Qk-1 and b ≤ n < 10Qk, so b ≤ 10Qk-1
          -- a+b=n, a ≥ 4*Qj ≥ 4, b ≥ 10Qk-1. a+b ≥ 4+10Qk-1 > n < 10Qk. Need 4+10Qk-1 > n?
          -- Actually: b ≥ 10Qk-1 and n < 10Qk → b ≤ n (from a+b=n, a ≥ 0) → b ≤ n ≤ 10Qk-1
          -- So a = n - b ≤ n - (10Qk-1) ≤ 0. But a ≥ 4*Qj ≥ 4. Contradiction.
          have hbnn : b ≤ n := by omega
          have halow : 4 * Q j ≤ a := stage_lower haj
          linarith [hbi.1, hn.2, Q_pos j]
      · -- i > k: b ≥ 20Qk. But b ≤ n < 10Qk. Contradiction.
        have hbdown2 : 20 * Q k ≤ b := le_trans (stage_lower_large higt) (stage_lower hbi)
        linarith [hn.2]
    · -- j = k: a ∈ stage k
      rw [hjje] at haj
      rcases haj with ((haj | haj) | haj)
      · -- a = ck k
        simp only [mem_singleton_iff] at haj
        -- a = ck k, b = n - ck k ∈ [5Qk, 6Qk-1] = Bk k
        left
        refine ⟨haj, ?_⟩
        simp only [Bk, mem_Icc, ck]
        rw [haj, ck] at hab
        constructor <;> omega
      · -- a ∈ Bk k = [5Qk, 6Qk-1]
        simp only [Bk, mem_Icc] at haj
        -- b ∈ (3Qk, 5Qk-1], so b = ck k = 4Qk
        rcases lt_trichotomy i k with hilt | hije | higt
        · have hbup : b ≤ 3 * Q k := le_trans (stage_upper hbi) (stage_upper_small hilt)
          linarith [haj.1, hn.1]
        · rw [hije] at hbi
          rcases hbi with ((hbi | hbi) | hbi)
          · -- b = ck k
            simp only [mem_singleton_iff] at hbi
            right
            exact ⟨hbi, by simp only [Bk, mem_Icc]; exact haj⟩
          · -- b ∈ Bk k: a+b ≥ 5Qk+5Qk = 10Qk > n < 10Qk. Contradiction.
            simp only [Bk, mem_Icc] at hbi
            linarith [haj.1, hbi.1, hn.2]
          · -- b ∈ Fk k: a+b ≥ 5Qk+10Qk-1 = 15Qk-1 > n < 10Qk. Contradiction.
            simp only [Fk, mem_Icc] at hbi
            linarith [haj.1, hbi.1, hn.2]
        · have hbdown : 20 * Q k ≤ b := le_trans (stage_lower_large higt) (stage_lower hbi)
          linarith [hn.2]
      · -- a ∈ Fk k: a ≥ 10Qk-1 ≥ n (since n < 10Qk → n ≤ 10Qk-1). So a = n, b = 0.
        -- But b ∈ stage i → b ≥ 4 > 0. Contradiction.
        simp only [Fk, mem_Icc] at haj
        have hble : b ≤ n := by omega
        have hbpos : 0 < b := by
          have := stage_lower hbi
          linarith [Q_pos i]
        linarith [haj.1, hn.2]
    · -- j > k: a ≥ 20Qk > n. But a ≤ n. Contradiction.
      have hadown : 20 * Q k ≤ a := le_trans (stage_lower_large hjgt) (stage_lower haj)
      linarith [hn.2, ha_le_n]

-- Gap lemma: if ck k ∉ T (T ⊆ setA), then Jk k ∩ (T+T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck_not : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, Set.mem_add, mem_empty_iff_false, iff_false, not_and]
  intro hn ⟨a, ha, b, hb, hab⟩
  have ha_setA := hT ha
  have hb_setA := hT hb
  rcases rigidity_lem k n hn a b ha_setA hb_setA hab with ⟨rfl, _⟩ | ⟨rfl, _⟩
  · exact hck_not ha
  · exact hck_not hb

-- Main theorem
theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, fun n hn => basis n hn, ?_⟩
  intro A₁ A₂ hA₁ hA₂ hcover hdisj ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
  -- Pick k large enough: Qk > C₁ + C₂
  -- Use k = C₁ + C₂ + 1, then Q k > C₁ + C₂ from n_lt_Q_succ
  set k := C₁ + C₂ + 1 with hk_def
  have hQk_large : C₁ + C₂ < Q k := by
    have := n_lt_Q_succ (C₁ + C₂)
    simp [hk_def, Q_succ] at *
    linarith [Q_pos (C₁ + C₂)]
  have hQk_C1 : C₁ < Q k := by linarith
  have hQk_C2 : C₂ < Q k := by linarith
  -- ck k ∈ setA, so it's in A₁ or A₂
  have hck_setA : ck k ∈ setA := by
    exact Or.inr ⟨k, Or.inl (Or.inl rfl)⟩
  rcases hcover (ck k) hck_setA with hck1 | hck2
  · -- ck k ∈ A₁. By gap_lem (T = A₂), Jk k ∩ (A₂+A₂) = ∅.
    have hck2_not : ck k ∉ A₂ := by
      intro h
      have : ck k ∈ A₁ ∩ A₂ := ⟨hck1, h⟩
      rw [hdisj] at this; exact this
    have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck2_not
    -- hC₂: ∀ x, ∃ m ∈ A₂+A₂, m ∈ Icc x (x+C₂)
    -- Apply at x = 9*Qk: get m ∈ A₂+A₂ with 9Qk ≤ m ≤ 9Qk+C₂
    obtain ⟨m, hm_sum, hm_icc⟩ := hC₂ (9 * Q k)
    simp only [mem_Icc] at hm_icc
    -- m ∈ Jk k since 9Qk ≤ m and m ≤ 9Qk+C₂ < 10Qk (from C₂ < Qk)
    have hm_Jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico]
      exact ⟨hm_icc.1, by linarith [hm_icc.2, hQk_C2]⟩
    have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_Jk, hm_sum⟩
    rw [hgap] at this; exact this
  · -- ck k ∈ A₂. By gap_lem (T = A₁), Jk k ∩ (A₁+A₁) = ∅.
    have hck1_not : ck k ∉ A₁ := by
      intro h
      have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck2⟩
      rw [hdisj] at this; exact this
    have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck1_not
    obtain ⟨m, hm_sum, hm_icc⟩ := hC₁ (9 * Q k)
    simp only [mem_Icc] at hm_icc
    have hm_Jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico]
      exact ⟨hm_icc.1, by linarith [hm_icc.2, hQk_C1]⟩
    have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_Jk, hm_sum⟩
    rw [hgap] at this; exact this

end Erdos741OAI

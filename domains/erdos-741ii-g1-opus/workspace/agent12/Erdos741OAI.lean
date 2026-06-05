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

/-! ## Construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {i j : ℕ} (h : i ≤ j) : Q i ≤ Q j := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

/-! ## Membership helpers -/

lemma mem_setA_of_mem_stage {x k : ℕ} (h : x ∈ stage k) : x ∈ setA := by
  simp only [setA, Set.mem_union]
  exact Or.inr (Set.mem_iUnion.mpr ⟨k, h⟩)

lemma ck_mem_stage (k : ℕ) : ck k ∈ stage k := Or.inl (Or.inl rfl)

lemma setA_ge2 {x : ℕ} (h : x ∈ setA) : 2 ≤ x := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_insert_iff,
    Set.mem_singleton_iff] at h
  rcases h with (rfl | rfl) | ⟨j, hj⟩
  · omega
  · omega
  · have hb := Q_pos j
    simp only [stage, Set.mem_union, Set.mem_singleton_iff, ck, Bk, Fk, Set.mem_Icc] at hj
    rcases hj with (rfl | ⟨h1, h2⟩) | ⟨h1, h2⟩ <;> omega

lemma stage_bounds {x j : ℕ} (h : x ∈ stage j) : 4 * Q j ≤ x ∧ x ≤ 15 * Q j := by
  have hp := Q_pos j
  simp only [stage, Set.mem_union, Set.mem_singleton_iff, ck, Bk, Fk, Set.mem_Icc] at h
  rcases h with (rfl | ⟨h1, h2⟩) | ⟨h1, h2⟩ <;> omega

/-! ## Classification of elements below 10*Q k -/

lemma classify (k y : ℕ) (hy : y ∈ setA) (hlt : y < 10 * Q k) :
    y ≤ 3 * Q k ∨ y = 4 * Q k ∨ (5 * Q k ≤ y ∧ y ≤ 6 * Q k - 1) ∨ y = 10 * Q k - 1 := by
  have hp := Q_pos k
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_insert_iff,
    Set.mem_singleton_iff] at hy
  rcases hy with (rfl | rfl) | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt2 | heq | hgt
    · -- j < k : bounded by 15 * Q (k-1) = 3 * Q k
      left
      have hb := stage_bounds hj
      have hk1 : k = (k - 1) + 1 := by omega
      have hs : Q k = 5 * Q (k - 1) := by rw [hk1]; exact Q_succ (k - 1)
      have hjm : Q j ≤ Q (k - 1) := Q_mono (by omega)
      omega
    · -- j = k
      rw [heq] at hj
      simp only [stage, Set.mem_union, Set.mem_singleton_iff, ck, Bk, Fk, Set.mem_Icc] at hj
      rcases hj with (rfl | ⟨h1, h2⟩) | ⟨h1, h2⟩
      · right; left; rfl
      · right; right; left; exact ⟨h1, h2⟩
      · right; right; right; omega
    · -- j > k : bounded below by 4 * Q (k+1) = 20 * Q k > y
      exfalso
      have hb := stage_bounds hj
      have hkj : Q (k + 1) ≤ Q j := Q_mono (by omega)
      have hs : Q (k + 1) = 5 * Q k := Q_succ k
      omega

/-! ## Rigidity: every representation of n ∈ [9Qk,10Qk) uses ck k -/

lemma rigidity (k a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hlo : 9 * Q k ≤ a + b) (hhi : a + b < 10 * Q k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hp := Q_pos k
  have ga := setA_ge2 ha
  have gb := setA_ge2 hb
  have ca := classify k a ha (by omega)
  have cb := classify k b hb (by omega)
  rcases ca with ca | ca | ca | ca <;> rcases cb with cb | cb | cb | cb <;>
    first
      | (exfalso; omega)
      | (exact Or.inl ⟨by simp only [ck]; omega, Set.mem_Icc.mpr ⟨by omega, by omega⟩⟩)
      | (exact Or.inr ⟨by simp only [ck]; omega, Set.mem_Icc.mpr ⟨by omega, by omega⟩⟩)

/-! ## Gap lemma -/

lemma gap_lem (T : Set ℕ) (k : ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [Set.eq_empty_iff_forall_notMem]
  intro x hx
  rw [Set.mem_inter_iff] at hx
  obtain ⟨hxJ, hxTT⟩ := hx
  rw [Set.mem_add] at hxTT
  obtain ⟨a, ha, b, hb, hab⟩ := hxTT
  rw [Jk, Set.mem_Ico] at hxJ
  obtain ⟨hJlo, hJhi⟩ := hxJ
  have ha' := hT ha
  have hb' := hT hb
  have hr := rigidity k a b ha' hb' (by omega) (by omega)
  rcases hr with ⟨rfl, _⟩ | ⟨rfl, _⟩
  · exact hck ha
  · exact hck hb

/-! ## Basis lemma -/

lemma cover_exists (n : ℕ) (hn : 4 ≤ n) : ∃ k, 4 * Q k ≤ n ∧ n ≤ 30 * Q k := by
  have hex : ∃ j, n < 4 * Q j := by
    refine ⟨n, ?_⟩
    have h1 : n < Q n := by
      have h2 : n < 5 ^ n := Nat.lt_pow_self (by norm_num)
      simpa [Q] using h2
    have := Q_pos n
    omega
  classical
  let k0 := Nat.find hex
  have hP : n < 4 * Q k0 := Nat.find_spec hex
  have hk0pos : 0 < k0 := by
    by_contra h
    push_neg at h
    have : k0 = 0 := by omega
    rw [this] at hP
    simp only [Q, pow_zero, mul_one] at hP
    omega
  refine ⟨k0 - 1, ?_, ?_⟩
  · have hmin := Nat.find_min hex (m := k0 - 1) (by omega)
    push_neg at hmin
    exact hmin
  · have hk : k0 = (k0 - 1) + 1 := by omega
    rw [hk, Q_succ] at hP
    have := Q_pos (k0 - 1)
    omega

lemma mem2 : (2 : ℕ) ∈ setA := by simp [setA]
lemma mem3 : (3 : ℕ) ∈ setA := by simp [setA]

lemma mem4 : (4 : ℕ) ∈ setA := by
  apply mem_setA_of_mem_stage (k := 0)
  exact Or.inl (Or.inl rfl)

lemma mem5 : (5 : ℕ) ∈ setA := by
  apply mem_setA_of_mem_stage (k := 0)
  refine Or.inl (Or.inr ?_)
  simp only [Bk, Q, pow_zero, mul_one, Set.mem_Icc]; omega

lemma memIcc9_15 {x : ℕ} (h1 : 9 ≤ x) (h2 : x ≤ 15) : x ∈ setA := by
  apply mem_setA_of_mem_stage (k := 0)
  refine Or.inr ?_
  simp only [Fk, Q, pow_zero, mul_one, Set.mem_Icc]; omega

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  obtain ⟨k, hk1, hk2⟩ := cover_exists n hn
  rcases Nat.eq_zero_or_pos k with hk0 | hkpos
  · -- k = 0 : n ∈ [4,30], explicit witnesses
    subst hk0
    simp only [Q, pow_zero, mul_one] at hk1 hk2
    interval_cases n
    · exact ⟨2, mem2, 2, mem2, rfl⟩
    · exact ⟨2, mem2, 3, mem3, rfl⟩
    · exact ⟨3, mem3, 3, mem3, rfl⟩
    · exact ⟨2, mem2, 5, mem5, rfl⟩
    · exact ⟨3, mem3, 5, mem5, rfl⟩
    · exact ⟨4, mem4, 5, mem5, rfl⟩
    · exact ⟨5, mem5, 5, mem5, rfl⟩
    · exact ⟨2, mem2, 9, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨3, mem3, 9, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨4, mem4, 9, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨5, mem5, 9, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨2, mem2, 13, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨2, mem2, 14, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨2, mem2, 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨3, mem3, 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨4, mem4, 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨5, mem5, 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨9, memIcc9_15 (by omega) (by omega), 12, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨9, memIcc9_15 (by omega) (by omega), 13, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨9, memIcc9_15 (by omega) (by omega), 14, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨9, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨10, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨11, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨12, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨13, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨14, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
    · exact ⟨15, memIcc9_15 (by omega) (by omega), 15, memIcc9_15 (by omega) (by omega), rfl⟩
  · -- k = m+1 : eight pair-type cover of [20*Qm, 150*Qm]
    obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k ≠ 0)
    have hq : Q (m + 1) = 5 * Q m := Q_succ m
    have hqpos : 0 < Q m := Q_pos m
    rw [hq] at hk1 hk2
    -- membership providers
    have memI : ∀ x, 10 * Q m - 1 ≤ x → x ≤ 15 * Q m → x ∈ setA := by
      intro x h1 h2
      apply mem_setA_of_mem_stage (k := m)
      refine Or.inr ?_
      simp only [Fk, Set.mem_Icc]; omega
    have memC : (20 * Q m) ∈ setA := by
      apply mem_setA_of_mem_stage (k := m + 1)
      have he : (20 * Q m) = ck (m + 1) := by rw [ck, hq]; ring
      rw [he]; exact ck_mem_stage (m + 1)
    have memB : ∀ x, 25 * Q m ≤ x → x ≤ 30 * Q m - 1 → x ∈ setA := by
      intro x h1 h2
      apply mem_setA_of_mem_stage (k := m + 1)
      refine Or.inl (Or.inr ?_)
      simp only [Bk, Set.mem_Icc]; rw [hq]; omega
    have memF : ∀ x, 50 * Q m - 1 ≤ x → x ≤ 75 * Q m → x ∈ setA := by
      intro x h1 h2
      apply mem_setA_of_mem_stage (k := m + 1)
      refine Or.inr ?_
      simp only [Fk, Set.mem_Icc]; rw [hq]; omega
    by_cases hc1 : n ≤ 30 * Q m
    · by_cases h : n ≤ 25 * Q m - 1
      · exact ⟨n - (10 * Q m - 1), memI _ (by omega) (by omega),
          10 * Q m - 1, memI _ (by omega) (by omega), by omega⟩
      · exact ⟨15 * Q m, memI _ (by omega) (by omega),
          n - 15 * Q m, memI _ (by omega) (by omega), by omega⟩
    · by_cases hc2 : n ≤ 35 * Q m
      · exact ⟨n - 20 * Q m, memI _ (by omega) (by omega), 20 * Q m, memC, by omega⟩
      · by_cases hc3 : n ≤ 45 * Q m - 1
        · by_cases h : n ≤ 40 * Q m
          · exact ⟨n - 25 * Q m, memI _ (by omega) (by omega),
              25 * Q m, memB _ (by omega) (by omega), by omega⟩
          · exact ⟨15 * Q m, memI _ (by omega) (by omega),
              n - 15 * Q m, memB _ (by omega) (by omega), by omega⟩
        · by_cases hc4 : n ≤ 50 * Q m - 1
          · exact ⟨20 * Q m, memC, n - 20 * Q m, memB _ (by omega) (by omega), by omega⟩
          · by_cases hc5 : n ≤ 60 * Q m - 2
            · by_cases h : n ≤ 55 * Q m - 1
              · exact ⟨n - 25 * Q m, memB _ (by omega) (by omega),
                  25 * Q m, memB _ (by omega) (by omega), by omega⟩
              · exact ⟨30 * Q m - 1, memB _ (by omega) (by omega),
                  n - (30 * Q m - 1), memB _ (by omega) (by omega), by omega⟩
            · by_cases hc6 : n ≤ 90 * Q m
              · by_cases h : n ≤ 65 * Q m - 1
                · exact ⟨n - (50 * Q m - 1), memI _ (by omega) (by omega),
                    50 * Q m - 1, memF _ (by omega) (by omega), by omega⟩
                · exact ⟨15 * Q m, memI _ (by omega) (by omega),
                    n - 15 * Q m, memF _ (by omega) (by omega), by omega⟩
              · by_cases hc7 : n ≤ 105 * Q m - 1
                · by_cases h : n ≤ 80 * Q m - 2
                  · exact ⟨n - (50 * Q m - 1), memB _ (by omega) (by omega),
                      50 * Q m - 1, memF _ (by omega) (by omega), by omega⟩
                  · exact ⟨30 * Q m - 1, memB _ (by omega) (by omega),
                      n - (30 * Q m - 1), memF _ (by omega) (by omega), by omega⟩
                · by_cases h : n ≤ 125 * Q m - 1
                  · exact ⟨n - (50 * Q m - 1), memF _ (by omega) (by omega),
                      50 * Q m - 1, memF _ (by omega) (by omega), by omega⟩
                  · exact ⟨75 * Q m, memF _ (by omega) (by omega),
                      n - 75 * Q m, memF _ (by omega) (by omega), by omega⟩

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
    exact basis_lem n hn
  · rintro A₁ A₂ h1 h2 hcov hdisj ⟨⟨C₁, hs1⟩, ⟨C₂, hs2⟩⟩
    set C := max C₁ C₂ with hC
    set k := C + 1 with hk
    have hQk : C < Q k := by
      have h2C : C < 5 ^ C := Nat.lt_pow_self (by norm_num)
      have hle : (5 : ℕ) ^ C ≤ 5 ^ (C + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
      have : Q k = 5 ^ (C + 1) := by rw [hk, Q]
      omega
    have hckA : ck k ∈ setA := mem_setA_of_mem_stage (ck_mem_stage k)
    rcases hcov (ck k) hckA with hin1 | hin2
    · have hnotin : ck k ∉ A₂ := by
        intro h
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hin1, h⟩
        rw [hdisj] at hmem
        simpa using hmem
      have hgap := gap_lem A₂ k h2 hnotin
      obtain ⟨m, hmS, hmIcc⟩ := hs2 (9 * Q k)
      rw [Set.mem_Icc] at hmIcc
      have hC2 : C₂ ≤ C := le_max_right _ _
      have hmJ : m ∈ Jk k := by
        rw [Jk, Set.mem_Ico]
        exact ⟨hmIcc.1, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra
      simpa using hcontra
    · have hnotin : ck k ∉ A₁ := by
        intro h
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h, hin2⟩
        rw [hdisj] at hmem
        simpa using hmem
      have hgap := gap_lem A₁ k h1 hnotin
      obtain ⟨m, hmS, hmIcc⟩ := hs1 (9 * Q k)
      rw [Set.mem_Icc] at hmIcc
      have hC1 : C₁ ≤ C := le_max_left _ _
      have hmJ : m ∈ Jk k := by
        rw [Jk, Set.mem_Ico]
        exact ⟨hmIcc.1, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra
      simpa using hcontra

end Erdos741OAI

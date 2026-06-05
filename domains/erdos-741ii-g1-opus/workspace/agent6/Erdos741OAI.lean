import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- `Q k = 5^k`. -/
def Q (k : ℕ) : ℕ := 5 ^ k

/-- The "gap zone" interval. -/
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

/-- The construction, written directly as a membership predicate. -/
def setA : Set ℕ :=
  {x | x = 2 ∨ x = 3 ∨
        ∃ j, x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨
              (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j)}

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b :=
  Nat.pow_le_pow_right (by norm_num) h

lemma lt_Q (k : ℕ) : k < Q k := by
  induction k with
  | zero => norm_num [Q]
  | succ m ih =>
    have hQs := Q_succ m
    have hp := Q_pos m
    omega

lemma two_le (x : ℕ) (hx : x ∈ setA) : 2 ≤ x := by
  simp only [setA, Set.mem_setOf_eq] at hx
  rcases hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have := Q_pos j
    rcases hj with h | ⟨h, _⟩ | ⟨h, _⟩ <;> omega

/-- Every element of `setA` is small (≤ 3Qk), or `4Qk`, or in the body interval, or large. -/
lemma size_class (k x : ℕ) (hx : x ∈ setA) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ x := by
  have qpos := Q_pos k
  simp only [setA, Set.mem_setOf_eq] at hx
  rcases hx with h | h | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · have hmono : Q (j + 1) ≤ Q k := Q_mono (by omega)
      have hQs : Q (j + 1) = 5 * Q j := Q_succ j
      have h5 : 5 * Q j ≤ Q k := by omega
      left
      rcases hj with he | ⟨hlo, hhi⟩ | ⟨hlo, hhi⟩ <;> omega
    · rw [hje] at hj
      rcases hj with he | ⟨hlo, hhi⟩ | ⟨hlo, hhi⟩
      · right; left; exact he
      · right; right; left; exact ⟨hlo, hhi⟩
      · right; right; right; omega
    · have hmono : Q (k + 1) ≤ Q j := Q_mono (by omega)
      have hQs : Q (k + 1) = 5 * Q k := Q_succ k
      have h5 : 5 * Q k ≤ Q j := by omega
      right; right; right
      rcases hj with he | ⟨hlo, hhi⟩ | ⟨hlo, hhi⟩ <;> omega

/-- The interval `[2Qk, 3Qk]` is contained in `setA` (it is `{2,3}` at level 0, or `F(k-1)`). -/
lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ setA := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  cases k with
  | zero =>
    have hQ0 : Q 0 = 1 := rfl
    have hx2 : x = 2 ∨ x = 3 := by omega
    rcases hx2 with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
  | succ m =>
    have hQs : Q (m + 1) = 5 * Q m := Q_succ m
    exact Or.inr (Or.inr ⟨m, Or.inr (Or.inr ⟨by omega, by omega⟩)⟩)

/-- Rigidity: any representation of `n ∈ Jk k` uses `4Qk` (the connector) plus a body element. -/
lemma rigidity (k a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hlo : 9 * Q k ≤ a + b) (hhi : a + b < 10 * Q k) :
    (a = 4 * Q k ∧ 5 * Q k ≤ b ∧ b ≤ 6 * Q k - 1) ∨
    (b = 4 * Q k ∧ 5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1) := by
  have qpos := Q_pos k
  have ha2 := two_le a ha
  have hb2 := two_le b hb
  rcases size_class k a ha with sa | sa | sa | sa
  · rcases size_class k b hb with sb | sb | sb | sb
    · exfalso; omega
    · exfalso; omega
    · exfalso; obtain ⟨hb1, hb2'⟩ := sb; omega
    · exfalso; omega
  · rcases size_class k b hb with sb | sb | sb | sb
    · exfalso; omega
    · exfalso; omega
    · left; exact ⟨sa, sb.1, sb.2⟩
    · exfalso; omega
  · rcases size_class k b hb with sb | sb | sb | sb
    · exfalso; obtain ⟨ha1, ha2'⟩ := sa; omega
    · right; exact ⟨sb, sa.1, sa.2⟩
    · exfalso; obtain ⟨ha1, _⟩ := sa; obtain ⟨hb1, _⟩ := sb; omega
    · exfalso; obtain ⟨ha1, _⟩ := sa; omega
  · exfalso; omega

/-- Gap lemma: if the connector `4Qk` is not in `T`, then `T+T` misses the whole gap zone. -/
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : (4 * Q k) ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [eq_empty_iff_forall_notMem]
  intro n hn
  rw [mem_inter_iff] at hn
  obtain ⟨hnJ, hnTT⟩ := hn
  simp only [Jk, mem_Ico] at hnJ
  rw [Set.mem_add] at hnTT
  obtain ⟨a, haT, b, hbT, hab⟩ := hnTT
  rcases rigidity k a b (hT haT) (hT hbT) (by omega) (by omega) with ⟨hae, _, _⟩ | ⟨hbe, _, _⟩
  · exact hck (hae ▸ haT)
  · exact hck (hbe ▸ hbT)

/-- Basis lemma: `setA + setA` covers `[4, 6Qk]` for every `k`. -/
lemma basis_cover (k : ℕ) :
    ∀ n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  induction k with
  | zero =>
    intro n h4 hn
    have hQ0 : Q 0 = 1 := rfl
    have two_mem : (2 : ℕ) ∈ setA := Or.inl rfl
    have three_mem : (3 : ℕ) ∈ setA := Or.inr (Or.inl rfl)
    rcases (by omega : n = 4 ∨ n = 5 ∨ n = 6) with rfl | rfl | rfl
    · exact ⟨2, two_mem, 2, two_mem, by norm_num⟩
    · exact ⟨2, two_mem, 3, three_mem, by norm_num⟩
    · exact ⟨3, three_mem, 3, three_mem, by norm_num⟩
  | succ k ih =>
    intro n h4 hn
    have hq := Q_pos k
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    by_cases hsmall : n ≤ 6 * Q k
    · exact ih n h4 hsmall
    · have ck_mem : (4 * Q k) ∈ setA := Or.inr (Or.inr ⟨k, Or.inl rfl⟩)
      have Bmem : ∀ x, 5 * Q k ≤ x → x ≤ 6 * Q k - 1 → x ∈ setA :=
        fun x h1 h2 => Or.inr (Or.inr ⟨k, Or.inr (Or.inl ⟨h1, h2⟩)⟩)
      have Fmem : ∀ x, 10 * Q k - 1 ≤ x → x ≤ 15 * Q k → x ∈ setA :=
        fun x h1 h2 => Or.inr (Or.inr ⟨k, Or.inr (Or.inr ⟨h1, h2⟩)⟩)
      have Imem : ∀ x, 2 * Q k ≤ x → x ≤ 3 * Q k → x ∈ setA := by
        intro x h1 h2
        exact I_sub k (mem_Icc.mpr ⟨h1, h2⟩)
      by_cases hA : n ≤ 7 * Q k
      · exact ⟨4 * Q k, ck_mem, n - 4 * Q k, Imem _ (by omega) (by omega), by omega⟩
      · by_cases hB : n ≤ 9 * Q k - 1
        · by_cases hB1 : n ≤ 8 * Q k - 1
          · exact ⟨n - 2 * Q k, Bmem _ (by omega) (by omega), 2 * Q k,
                   Imem _ (by omega) (by omega), by omega⟩
          · exact ⟨6 * Q k - 1, Bmem _ (by omega) (by omega), n - (6 * Q k - 1),
                   Imem _ (by omega) (by omega), by omega⟩
        · by_cases hC : n ≤ 10 * Q k - 1
          · exact ⟨4 * Q k, ck_mem, n - 4 * Q k, Bmem _ (by omega) (by omega), by omega⟩
          · by_cases hD : n ≤ 12 * Q k - 2
            · by_cases hD1 : n ≤ 11 * Q k - 1
              · exact ⟨n - 5 * Q k, Bmem _ (by omega) (by omega), 5 * Q k,
                       Bmem _ (by omega) (by omega), by omega⟩
              · exact ⟨6 * Q k - 1, Bmem _ (by omega) (by omega), n - (6 * Q k - 1),
                       Bmem _ (by omega) (by omega), by omega⟩
            · by_cases hE : n ≤ 18 * Q k
              · by_cases hE1 : n ≤ 17 * Q k
                · exact ⟨n - 2 * Q k, Fmem _ (by omega) (by omega), 2 * Q k,
                         Imem _ (by omega) (by omega), by omega⟩
                · exact ⟨15 * Q k, Fmem _ (by omega) (by omega), n - 15 * Q k,
                         Imem _ (by omega) (by omega), by omega⟩
              · by_cases hF : n ≤ 21 * Q k - 1
                · exact ⟨6 * Q k - 1, Bmem _ (by omega) (by omega), n - (6 * Q k - 1),
                         Fmem _ (by omega) (by omega), by omega⟩
                · by_cases hG1 : n ≤ 25 * Q k - 1
                  · exact ⟨n - (10 * Q k - 1), Fmem _ (by omega) (by omega), 10 * Q k - 1,
                           Fmem _ (by omega) (by omega), by omega⟩
                  · exact ⟨15 * Q k, Fmem _ (by omega) (by omega), n - 15 * Q k,
                           Fmem _ (by omega) (by omega), by omega⟩

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n h4
    exact basis_cover n n h4 (by have := lt_Q n; omega)
  · intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    set k := C₁ + C₂ + 1 with hk
    have hlt : k < Q k := lt_Q k
    have hckmem : (4 * Q k) ∈ setA := Or.inr (Or.inr ⟨k, Or.inl rfl⟩)
    rcases hcov (4 * Q k) hckmem with hin1 | hin2
    · have hnotA2 : (4 * Q k) ∉ A₂ := by
        intro hh
        have hmem : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hin1, hh⟩
        rw [hdisj] at hmem; simpa using hmem
      have hgap := gap_lem k A₂ h2 hnotA2
      obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      have hmem : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hmem; simpa using hmem
    · have hnotA1 : (4 * Q k) ∉ A₁ := by
        intro hh
        have hmem : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hh, hin2⟩
        rw [hdisj] at hmem; simpa using hmem
      have hgap := gap_lem k A₁ h1 hnotA1
      obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      have hmem : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hmem; simpa using hmem

end Erdos741OAI

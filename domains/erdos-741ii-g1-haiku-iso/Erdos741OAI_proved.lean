import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def IsAddBasis2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ (A ∪ {0}), ∃ b ∈ (A ∪ {0}), a + b = n

def Q (k : ℕ) : ℕ := 5 ^ k

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k
lemma Q_ge_one (k : ℕ) : 1 ≤ Q k := Q_pos k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by simp [Q, pow_succ, mul_comm]

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | n + 1 => Akn n ∪ {ck n} ∪ Bk n ∪ Fk n

lemma akn_mono {m n : ℕ} (h : m ≤ n) : Akn m ⊆ Akn n := by
  induction h with
  | refl => rfl
  | @step k _ ih =>
    exact ih.trans (fun x hx => by simp only [Akn]; left; left; left; exact hx)

lemma akn_bound {k : ℕ} {x : ℕ} (hx : x ∈ Akn k) : x ≤ 3 * Q k := by
  induction k with
  | zero => simp only [Akn, mem_Icc] at hx; simp [Q]; exact hx.2
  | succ k ih =>
    simp only [Akn, mem_union, mem_singleton_iff, Bk, Fk, mem_Icc] at hx
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    rcases hx with (((h | h) | ⟨h1, h2⟩) | ⟨h1, h2⟩)
    · have := ih h; rw [hQs]; linarith
    · rw [h]; simp only [ck]; rw [hQs]; linarith
    · rw [hQs]; omega
    · rw [hQs]; omega

lemma ik_sub_akn (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx; simp only [Q, pow_zero, mul_one] at hx
    exact Or.inl (Or.inl (Or.inl hx))
  | succ k =>
    intro x hx
    simp only [mem_Icc] at hx
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    have hx_fk : x ∈ Fk k := by
      simp only [Fk, mem_Icc]; rw [hQs] at hx; constructor <;> omega
    exact akn_mono (Nat.le_succ _)
      (Or.inr hx_fk : x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)

lemma icc_add_icc_ge {a b c d : ℕ} (h1 : a ≤ b) (h2 : c ≤ d) :
    Icc (a + c) (b + d) ⊆ (Icc a b + Icc c d : Set ℕ) := by
  intro x hx
  simp only [mem_Icc] at hx
  simp only [Set.mem_add, mem_Icc]
  by_cases h : x ≤ a + d
  · exact ⟨a, ⟨le_refl _, h1⟩, x - a, ⟨by omega, by omega⟩, by omega⟩
  · push_neg at h
    exact ⟨x - d, ⟨by omega, by omega⟩, d, ⟨h2, le_refl _⟩, by omega⟩

lemma singleton_add_icc {a b c : ℕ} (h : a ≤ b) :
    ({c} : Set ℕ) + Icc a b = Icc (c + a) (c + b) := by
  ext x; simp only [Set.mem_add, mem_singleton_iff, mem_Icc]
  constructor
  · rintro ⟨y, rfl, z, ⟨hz1, hz2⟩, rfl⟩; omega
  · intro ⟨hx1, hx2⟩
    exact ⟨c, rfl, x - c, ⟨by omega, by omega⟩, Nat.add_sub_cancel' (by omega)⟩

lemma icc_add_singleton {a b c : ℕ} (h : a ≤ b) :
    (Icc a b + {c} : Set ℕ) = Icc (a + c) (b + c) := by
  ext x; simp only [Set.mem_add, mem_singleton_iff, mem_Icc]
  constructor
  · rintro ⟨y, ⟨hy1, hy2⟩, z, rfl, rfl⟩; omega
  · intro ⟨hx1, hx2⟩
    exact ⟨x - c, ⟨by omega, by omega⟩, c, rfl, Nat.sub_add_cancel (by omega)⟩

private lemma pair_eq' (x b : ℕ) (h : b ≤ x) : b + (x - b) = x := Nat.add_sub_cancel' h

-- Coverage: I=[2Q,3Q], ck=4Q, Bk=[5Q,6Q-1], Fk=[10Q-1,15Q]
-- 8 pair types cover [4Q,30Q]: I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one] at hx
    have hbase : Icc 2 3 ⊆ Akn 1 := fun y hy =>
      (Or.inl (Or.inl (Or.inl hy)) : y ∈ Akn 0 ∪ {ck 0} ∪ Bk 0 ∪ Fk 0)
    exact Set.add_subset_add hbase hbase
      (icc_add_icc_ge (by norm_num) (by norm_num) (by simpa using hx))
  | succ k ih =>
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    have hQge : 1 ≤ Q k := Q_ge_one k
    rw [hQs]
    show Icc 4 (6 * (5 * Q k)) ⊆ Akn (k + 1 + 1) + Akn (k + 1 + 1)
    have ih' : Icc 4 (6 * Q k) ⊆ Akn (k + 1 + 1) + Akn (k + 1 + 1) :=
      ih.trans (Set.add_subset_add (akn_mono (Nat.le_succ _)) (akn_mono (Nat.le_succ _)))
    have hI : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 2) :=
      (ik_sub_akn k).trans (akn_mono (Nat.le_succ _))
    have hBk_sub : Bk k ⊆ Akn (k + 2) := fun x hx => akn_mono (Nat.le_succ _)
      (Or.inl (Or.inr hx) : x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
    have hFk_sub : Fk k ⊆ Akn (k + 2) := fun x hx => akn_mono (Nat.le_succ _)
      (Or.inr hx : x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
    have inI : ∀ y, 2*Q k ≤ y → y ≤ 3*Q k → y ∈ Akn (k+2) :=
      fun y h1 h2 => hI (mem_Icc.mpr ⟨h1, h2⟩)
    have inCk : 4 * Q k ∈ Akn (k + 2) := akn_mono (Nat.le_succ _)
      (Or.inl (Or.inl (Or.inr rfl)) : ck k ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
    have inBk : ∀ y, 5*Q k ≤ y → y ≤ 6*Q k - 1 → y ∈ Akn (k+2) :=
      fun y h1 h2 => hBk_sub (mem_Icc.mpr ⟨h1, h2⟩)
    have inFk : ∀ y, 10*Q k - 1 ≤ y → y ≤ 15*Q k → y ∈ Akn (k+2) :=
      fun y h1 h2 => hFk_sub (mem_Icc.mpr ⟨h1, h2⟩)
    have new_cov : Icc (4 * Q k) (30 * Q k) ⊆ Akn (k + 1 + 1) + Akn (k + 1 + 1) := by
      intro x hx
      obtain ⟨hlo, hhi⟩ := hx
      by_cases h1 : x ≤ 5 * Q k
      · exact ⟨x-2*Q k, inI _ (by omega) (by omega), 2*Q k, inI _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h2 : x ≤ 6 * Q k
      · exact ⟨x-3*Q k, inI _ (by omega) (by omega), 3*Q k, inI _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h3 : x ≤ 7 * Q k
      · exact ⟨x-4*Q k, inI _ (by omega) (by omega), 4*Q k, inCk,
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h4 : x ≤ 8 * Q k
      · exact ⟨x-5*Q k, inI _ (by omega) (by omega), 5*Q k, inBk _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h5 : x ≤ 9 * Q k - 1
      · have hle : 6 * Q k - 1 ≤ x := by omega
        exact ⟨x-(6*Q k-1), inI _ (by omega) (by omega), 6*Q k-1, inBk _ (by omega) (by omega),
               Nat.sub_add_cancel hle⟩
      by_cases h6 : x ≤ 10 * Q k - 1
      · exact ⟨4*Q k, inCk, x-4*Q k, inBk _ (by omega) (by omega),
               Nat.add_sub_cancel' (by linarith)⟩
      by_cases h7 : x ≤ 11 * Q k - 1
      · exact ⟨x-5*Q k, inBk _ (by omega) (by omega), 5*Q k, inBk _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h8 : x ≤ 12 * Q k - 2
      · have hle : 6 * Q k - 1 ≤ x := by omega
        exact ⟨6*Q k-1, inBk _ (by omega) (by omega), x-(6*Q k-1), inBk _ (by omega) (by omega),
               Nat.add_sub_cancel' hle⟩
      by_cases h9 : x ≤ 17 * Q k
      · exact ⟨2*Q k, inI _ (by omega) (by omega), x-2*Q k, inFk _ (by omega) (by omega),
               Nat.add_sub_cancel' (by linarith)⟩
      by_cases h10 : x ≤ 18 * Q k
      · exact ⟨3*Q k, inI _ (by omega) (by omega), x-3*Q k, inFk _ (by omega) (by omega),
               Nat.add_sub_cancel' (by linarith)⟩
      by_cases h11 : x ≤ 20 * Q k
      · exact ⟨5*Q k, inBk _ (by omega) (by omega), x-5*Q k, inFk _ (by omega) (by omega),
               Nat.add_sub_cancel' (by linarith)⟩
      by_cases h12 : x ≤ 21 * Q k - 1
      · exact ⟨x-15*Q k, inBk _ (by omega) (by omega), 15*Q k, inFk _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
      by_cases h13 : x ≤ 25 * Q k - 2
      · have hle : 10 * Q k - 1 ≤ x := by omega
        exact ⟨10*Q k-1, inFk _ (by omega) (by omega), x-(10*Q k-1), inFk _ (by omega) (by omega),
               Nat.add_sub_cancel' hle⟩
      · exact ⟨x-15*Q k, inFk _ (by omega) (by omega), 15*Q k, inFk _ (by omega) (by omega),
               Nat.sub_add_cancel (by linarith)⟩
    intro x hx
    simp only [mem_Icc] at hx
    by_cases h : x ≤ 6 * Q k
    · exact ih' (mem_Icc.mpr ⟨by linarith, by linarith⟩)
    · push_neg at h
      exact new_cov (mem_Icc.mpr ⟨by linarith, by linarith⟩)

-- ─────────────────────────────────────────────
-- Part 2: setA is a basis and no partition is both-syndetic
-- ─────────────────────────────────────────────

-- Q k grows without bound: n ≤ Q n
lemma n_le_Qn (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ n ih => rw [Q_succ]; linarith [Q_pos n]

-- Akn k ⊆ setA
lemma akn_sub_setA {k : ℕ} : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    simp only [Akn, mem_union, mem_singleton_iff] at hx
    rcases hx with (((h | h) | h) | h)
    · exact ih h
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl (Set.mem_singleton_iff.mpr h))⟩)
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inr h)⟩)
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inr h⟩)

-- Every n ≥ 4 is a sum of two setA elements
lemma setA_covers : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  obtain ⟨a, ha, b, hb, hab⟩ := basis_lem n (mem_Icc.mpr ⟨hn, by linarith [n_le_Qn n]⟩)
  exact ⟨a, akn_sub_setA ha, b, akn_sub_setA hb, hab⟩

-- ck k ∈ setA
lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl (Set.mem_singleton_iff.mpr rfl))⟩)

-- Rigidity: n ∈ Jk=[9Qk,10Qk) can only be written as ck k + Bk k.
-- Stage j < k: elements ≤ 3Qk; stage j > k: elements ≥ 20Qk > n.
-- Only stage k pair that sums into [9Qk,10Qk) is 4Qk + [5Qk,6Qk-1].
lemma rigidity (k : ℕ) {n : ℕ} (hn : n ∈ Jk k)
    {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hlo, hhi⟩ := hn
  have hQk := Q_pos k
  have stage_lo : ∀ j x, x ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ) → 4 * Q j ≤ x := by
    intro j x hx; simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
    have := Q_pos j; rcases hx with ((rfl | ⟨h, _⟩) | ⟨h, _⟩) <;> omega
  have stage_hi : ∀ j x, x ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ) → x ≤ 15 * Q j := by
    intro j x hx; simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
    have := Q_pos j; rcases hx with ((rfl | ⟨_, h⟩) | ⟨_, h⟩) <;> omega
  have small_stage : ∀ j x, j < k → x ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ) → x ≤ 3 * Q k := by
    intro j x hj hx
    have h1 : Q (j + 1) = 5 * Q j := Q_succ j
    have h2 : Q (j + 1) ≤ Q k := Nat.pow_le_pow_right (by norm_num) hj
    linarith [stage_hi j x hx]
  have large_stage : ∀ j x, k < j → x ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ) → 20 * Q k ≤ x := by
    intro j x hj hx
    have h1 : Q (k + 1) = 5 * Q k := Q_succ k
    have h2 : Q (k + 1) ≤ Q j := Nat.pow_le_pow_right (by norm_num) hj
    linarith [stage_lo j x hx]
  -- Tactic helper: close setA membership contradiction for b in a given stage
  -- (avoids repeating the 3-way split + omega)
  simp only [setA, mem_union, mem_iUnion] at ha hb
  -- Inline helper: b in stage j', close contradiction with context k-bounds
  -- (k always stays in scope; use rw not subst to avoid renaming k)
  -- ── Case 1: a ∈ Icc 2 3 ──────────────────────────────────────────────
  rcases ha with ha23 | ⟨j, haj⟩
  · obtain ⟨_, ha_hi⟩ := mem_Icc.mp ha23
    rcases hb with hb23 | ⟨j', hbj'⟩
    · obtain ⟨hb_lo, _⟩ := mem_Icc.mp hb23; linarith
    · rcases lt_trichotomy j' k with hlt' | hje' | hgt'
      · linarith [small_stage j' b hlt' hbj']
      · rw [hje'] at hbj'  -- rewrite j' to k in hbj'; k stays in scope
        simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hbj'
        rcases hbj' with ((rfl | ⟨_, hb2⟩) | ⟨hb1, _⟩) <;> [linarith; omega; omega]
      · linarith [large_stage j' b hgt' hbj']
  -- ── Case 2: a ∈ stage j ──────────────────────────────────────────────
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · -- j < k: a ≤ 3Qk; k always in scope (no subst)
      have ha_bd := small_stage j a hlt haj
      have ha_lo := stage_lo j a haj
      have hQj := Q_pos j
      rcases hb with hb23 | ⟨j', hbj'⟩
      · obtain ⟨_, hb_hi⟩ := mem_Icc.mp hb23; linarith
      · rcases lt_trichotomy j' k with hlt' | hje' | hgt'
        · linarith [small_stage j' b hlt' hbj']
        · rw [hje'] at hbj'
          simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hbj'
          rcases hbj' with ((rfl | ⟨_, hb2⟩) | ⟨hb1, _⟩) <;> [linarith; omega; omega]
        · linarith [large_stage j' b hgt' hbj']
    · -- j = k: rewrite haj to use k, keep both j and k in scope
      rw [hje] at haj
      simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at haj
      rcases haj with ((rfl | ⟨ha1, ha2⟩) | ⟨ha1, _⟩)
      · -- a = 4*Qk = ck k
        exact Or.inl ⟨by simp [ck], mem_Icc.mpr ⟨by omega, by omega⟩⟩
      · -- a ∈ Bk k: only b = ck k = 4*Qk works
        rcases hb with hb23 | ⟨j', hbj'⟩
        · obtain ⟨_, hb_hi⟩ := mem_Icc.mp hb23; omega
        · rcases lt_trichotomy j' k with hlt' | hje' | hgt'
          · have := small_stage j' b hlt' hbj'; omega
          · rw [hje'] at hbj'
            simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hbj'
            rcases hbj' with ((rfl | ⟨hb1', _⟩) | ⟨hb1', _⟩)
            · exact Or.inr ⟨by simp [ck], mem_Icc.mpr ⟨ha1, ha2⟩⟩
            · linarith   -- 5*Qk ≤ a ∧ 5*Qk ≤ b → a+b ≥ 10*Qk > n
            · omega      -- 10*Qk-1 ≤ b, 5*Qk ≤ a, a+b < 10*Qk
          · linarith [large_stage j' b hgt' hbj']
      · -- a ∈ Fk k: b forced to 0 (∉ setA)
        simp only [setA, mem_union, mem_iUnion, mem_Icc, mem_singleton_iff, ck, Bk, Fk] at hb
        rcases hb with ⟨hb1, _⟩ | ⟨j', hbj'⟩
        · omega
        · have := Q_pos j'
          rcases hbj' with (h2 | ⟨h2, _⟩ | ⟨h2, _⟩) <;> omega
    · -- j > k: a ≥ 20*Qk > n
      linarith [large_stage j a hgt haj]

-- Gap lemma: if ck k ∉ T (T ⊆ setA), then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, Set.mem_add, mem_empty_iff_false, iff_false, not_and]
  intro hn
  rintro ⟨a, ha, b, hb, hab⟩
  rcases rigidity k hn (hT ha) (hT hb) hab with ⟨rfl, _⟩ | ⟨rfl, _⟩
  · exact hck ha
  · exact hck hb

-- Main theorem: setA is an additive basis of order 2 (for n ≥ 4) and
-- no 2-partition of setA has both summsets syndetic.
theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, setA_covers, ?_⟩
  intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨⟨C₁, hC₁⟩, C₂, hC₂⟩
  -- Pick k with Q k > max(C₁, C₂)
  obtain ⟨k, hk⟩ : ∃ k, C₁ + C₂ < Q k :=
    ⟨C₁ + C₂ + 1, lt_of_lt_of_le (Nat.lt_succ_self _) (n_le_Qn _)⟩
  have hC₁k : C₁ < Q k := by linarith
  have hC₂k : C₂ < Q k := by linarith
  -- ck k ∈ setA, so it belongs to A₁ or A₂
  rcases hpart (ck k) (ck_mem_setA k) with h₁ | h₂
  · -- ck k ∈ A₁ → ck k ∉ A₂ → Jk ∩ (A₂+A₂) = ∅
    have hck₂ : ck k ∉ A₂ := fun h => by
      have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h₁, h⟩; simp [hdisj] at hmem
    have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck₂
    obtain ⟨m, hm_sum, hm_range⟩ := hC₂ (9 * Q k)
    have hm_jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico, mem_Icc] at *
      exact ⟨hm_range.1, by linarith [hm_range.2]⟩
    have hmem : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_jk, hm_sum⟩
    simp [hgap] at hmem
  · -- ck k ∈ A₂ → ck k ∉ A₁ → Jk ∩ (A₁+A₁) = ∅
    have hck₁ : ck k ∉ A₁ := fun h => by
      have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h, h₂⟩; simp [hdisj] at hmem
    have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck₁
    obtain ⟨m, hm_sum, hm_range⟩ := hC₁ (9 * Q k)
    have hm_jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico, mem_Icc] at *
      exact ⟨hm_range.1, by linarith [hm_range.2]⟩
    have hmem : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_jk, hm_sum⟩
    simp [hgap] at hmem

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Scale function: Q k = 5^k. -/
def Q (k : ℕ) : ℕ := 5 ^ k

/-- Stage k block: marker 4·Q k, narrow middle interval, and high interval. -/
def stage (k : ℕ) : Set ℕ :=
  {4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k)

/-- The construction. -/
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

/-! ### Basic facts about Q -/

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {i j : ℕ} (h : i ≤ j) : Q i ≤ Q j := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma Q_pos (k : ℕ) : 1 ≤ Q k := by
  unfold Q; exact Nat.one_le_pow _ _ (by norm_num)

lemma Q_step_le {i k : ℕ} (h : i < k) : 5 * Q i ≤ Q k := by
  have hh := Q_mono (show i + 1 ≤ k by omega)
  rw [Q_succ] at hh; exact hh

lemma Q_step_ge {i k : ℕ} (h : k < i) : 5 * Q k ≤ Q i := by
  have hh := Q_mono (show k + 1 ≤ i by omega)
  rw [Q_succ] at hh; exact hh

lemma lt_Q (n : ℕ) : n < Q n := by
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : 2 ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  unfold Q; omega

/-! ### Membership analysis -/

lemma setA_ge_two {z : ℕ} (hz : z ∈ setA) : 2 ≤ z := by
  simp only [setA, stage, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff,
    Set.mem_insert_iff, Set.mem_Icc] at hz
  obtain h | ⟨i, hi⟩ := hz
  · omega
  · have hp := Q_pos i
    omega

/-- Every element of setA is binned into 4 bands relative to scale k. -/
lemma classify {z : ℕ} (hz : z ∈ setA) (k : ℕ) :
    z ≤ 3 * Q k ∨ z = 4 * Q k ∨ (5 * Q k ≤ z ∧ z ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ z := by
  have hp := Q_pos k
  simp only [setA, stage, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff,
    Set.mem_insert_iff, Set.mem_Icc] at hz
  obtain h | ⟨i, hi⟩ := hz
  · omega
  · have hpi := Q_pos i
    rcases lt_trichotomy i k with hik | hik | hik
    · have hstep := Q_step_le hik
      omega
    · subst hik
      omega
    · have hstep := Q_step_ge hik
      omega

/-! ### Interval subsets of setA -/

lemma lo_sub : Icc 2 3 ⊆ setA := by
  intro x hx
  rw [Set.mem_Icc] at hx
  have hmem : x ∈ ({2, 3} : Set ℕ) := by
    rw [Set.mem_insert_iff, Set.mem_singleton_iff]; omega
  exact Set.mem_union_left _ hmem

lemma mid_sub (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ setA := by
  intro x hx
  have hs : x ∈ stage k := Set.mem_union_left _ (Set.mem_union_right _ hx)
  exact Set.mem_union_right _ (Set.mem_iUnion.2 ⟨k, hs⟩)

lemma high_sub (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ setA := by
  intro x hx
  have hs : x ∈ stage k := Set.mem_union_right _ hx
  exact Set.mem_union_right _ (Set.mem_iUnion.2 ⟨k, hs⟩)

lemma marker_mem (k : ℕ) : 4 * Q k ∈ setA := by
  have hs : 4 * Q k ∈ stage k :=
    Set.mem_union_left _ (Set.mem_union_left _ (Set.mem_singleton_iff.mpr rfl))
  exact Set.mem_union_right _ (Set.mem_iUnion.2 ⟨k, hs⟩)

lemma marker_sub (k : ℕ) : Icc (4 * Q k) (4 * Q k) ⊆ setA := by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : x = 4 * Q k := by omega
  rw [this]; exact marker_mem k

/-! ### Minkowski-sum covering helper -/

lemma cover (a b c d y : ℕ) (sa : Icc a b ⊆ setA) (sc : Icc c d ⊆ setA)
    (hab : a ≤ b) (hcd : c ≤ d) (h1 : a + c ≤ y) (h2 : y ≤ b + d) :
    y ∈ setA + setA := by
  rw [Set.mem_add]
  refine ⟨min b (y - c), sa ?_, y - min b (y - c), sc ?_, by omega⟩
  · rw [Set.mem_Icc]; omega
  · rw [Set.mem_Icc]; omega

/-! ### Basis: every n ≥ 4 is a sum of two elements of setA -/

/-- Stage k+1 together with stage k covers `[4·Q(k+1), 30·Q(k+1)]`. -/
lemma stageCover (m : ℕ) : Icc (4 * Q (m + 1)) (30 * Q (m + 1)) ⊆ setA + setA := by
  intro y hy
  rw [Set.mem_Icc] at hy
  have hs : Q (m + 1) = 5 * Q m := Q_succ m
  have hp : 1 ≤ Q m := Q_pos m
  rcases (show
      (20 * Q m ≤ y ∧ y ≤ 30 * Q m) ∨ (30 * Q m ≤ y ∧ y ≤ 35 * Q m) ∨
      (35 * Q m ≤ y ∧ y ≤ 45 * Q m - 1) ∨ (45 * Q m ≤ y ∧ y ≤ 50 * Q m - 1) ∨
      (50 * Q m ≤ y ∧ y ≤ 60 * Q m - 2) ∨ (60 * Q m - 1 ≤ y ∧ y ≤ 75 * Q m) ∨
      (75 * Q m ≤ y ∧ y ≤ 105 * Q m - 1) ∨ (105 * Q m ≤ y ∧ y ≤ 150 * Q m)
      from by omega) with h | h | h | h | h | h | h | h
  · exact cover (10 * Q m - 1) (15 * Q m) (10 * Q m - 1) (15 * Q m) y
      (high_sub m) (high_sub m) (by omega) (by omega) (by omega) (by omega)
  · exact cover (4 * Q (m + 1)) (4 * Q (m + 1)) (10 * Q m - 1) (15 * Q m) y
      (marker_sub (m + 1)) (high_sub m) (by omega) (by omega) (by omega) (by omega)
  · exact cover (5 * Q (m + 1)) (6 * Q (m + 1) - 1) (10 * Q m - 1) (15 * Q m) y
      (mid_sub (m + 1)) (high_sub m) (by omega) (by omega) (by omega) (by omega)
  · exact cover (4 * Q (m + 1)) (4 * Q (m + 1)) (5 * Q (m + 1)) (6 * Q (m + 1) - 1) y
      (marker_sub (m + 1)) (mid_sub (m + 1)) (by omega) (by omega) (by omega) (by omega)
  · exact cover (5 * Q (m + 1)) (6 * Q (m + 1) - 1) (5 * Q (m + 1)) (6 * Q (m + 1) - 1) y
      (mid_sub (m + 1)) (mid_sub (m + 1)) (by omega) (by omega) (by omega) (by omega)
  · exact cover 2 3 (10 * Q (m + 1) - 1) (15 * Q (m + 1)) y
      lo_sub (high_sub (m + 1)) (by omega) (by omega) (by omega) (by omega)
  · exact cover (5 * Q (m + 1)) (6 * Q (m + 1) - 1) (10 * Q (m + 1) - 1) (15 * Q (m + 1)) y
      (mid_sub (m + 1)) (high_sub (m + 1)) (by omega) (by omega) (by omega) (by omega)
  · exact cover (10 * Q (m + 1) - 1) (15 * Q (m + 1)) (10 * Q (m + 1) - 1) (15 * Q (m + 1)) y
      (high_sub (m + 1)) (high_sub (m + 1)) (by omega) (by omega) (by omega) (by omega)

lemma basis_cover : ∀ k, Icc 4 (30 * Q k) ⊆ setA + setA := by
  intro k
  induction k with
  | zero =>
    intro y hy
    rw [Set.mem_Icc] at hy
    have hq0 : Q 0 = 1 := by norm_num [Q]
    rcases (show
        (4 ≤ y ∧ y ≤ 6) ∨ (6 ≤ y ∧ y ≤ 7) ∨ (7 ≤ y ∧ y ≤ 8) ∨
        (y = 9) ∨ (y = 10) ∨ (11 ≤ y ∧ y ≤ 18) ∨ (18 ≤ y ∧ y ≤ 30)
        from by omega) with h | h | h | h | h | h | h
    · exact cover 2 3 2 3 y lo_sub lo_sub (by omega) (by omega) (by omega) (by omega)
    · exact cover 2 3 (4 * Q 0) (4 * Q 0) y lo_sub (marker_sub 0)
        (by omega) (by omega) (by omega) (by omega)
    · exact cover 2 3 (5 * Q 0) (6 * Q 0 - 1) y lo_sub (mid_sub 0)
        (by omega) (by omega) (by omega) (by omega)
    · exact cover (4 * Q 0) (4 * Q 0) (5 * Q 0) (6 * Q 0 - 1) y (marker_sub 0) (mid_sub 0)
        (by omega) (by omega) (by omega) (by omega)
    · exact cover (5 * Q 0) (6 * Q 0 - 1) (5 * Q 0) (6 * Q 0 - 1) y (mid_sub 0) (mid_sub 0)
        (by omega) (by omega) (by omega) (by omega)
    · exact cover 2 3 (10 * Q 0 - 1) (15 * Q 0) y lo_sub (high_sub 0)
        (by omega) (by omega) (by omega) (by omega)
    · exact cover (10 * Q 0 - 1) (15 * Q 0) (10 * Q 0 - 1) (15 * Q 0) y (high_sub 0) (high_sub 0)
        (by omega) (by omega) (by omega) (by omega)
  | succ m IH =>
    intro y hy
    rw [Set.mem_Icc] at hy
    have hs : Q (m + 1) = 5 * Q m := Q_succ m
    have hp : 1 ≤ Q m := Q_pos m
    by_cases hc : y ≤ 30 * Q m
    · exact IH (Set.mem_Icc.2 ⟨hy.1, hc⟩)
    · exact stageCover m (Set.mem_Icc.2 ⟨by omega, hy.2⟩)

/-! ### Rigidity: unique representation in the gap window -/

/-- Any setA-representation of a number in `[9·Q k, 10·Q k)` must use the marker `4·Q k`. -/
lemma uniq (k a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (h1 : 9 * Q k ≤ a + b) (h2 : a + b < 10 * Q k) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have hp := Q_pos k
  have ca := classify ha k
  have cb := classify hb k
  rcases ca with c | c | c | c <;> rcases cb with d | d | d | d <;> omega

/-- If the marker `4·Q k` is not in `T`, then `T+T` misses the entire window `[9·Q k, 10·Q k)`. -/
lemma gap (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hk : 4 * Q k ∉ T)
    (y : ℕ) (hy1 : 9 * Q k ≤ y) (hy2 : y < 10 * Q k) : y ∉ T + T := by
  intro hyT
  rw [Set.mem_add] at hyT
  obtain ⟨a, ha, b, hb, hab⟩ := hyT
  have ha' := hT ha
  have hb' := hT hb
  rcases uniq k a b ha' hb' (by omega) (by omega) with h | h
  · exact hk (h ▸ ha)
  · exact hk (h ▸ hb)

/-! ### Main theorem -/

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · -- Basis
    intro n hn
    have hp := Q_pos n
    have hlt := lt_Q n
    have hmem : n ∈ setA + setA :=
      basis_cover n (Set.mem_Icc.2 ⟨hn, by omega⟩)
    rw [Set.mem_add] at hmem
    obtain ⟨a, ha, b, hb, hab⟩ := hmem
    exact ⟨a, ha, b, hb, hab⟩
  · -- Rigidity
    intro A₁ A₂ hA1 hA2 hpart hdisj
    rintro ⟨⟨C₁, h₁⟩, ⟨C₂, h₂⟩⟩
    set k := C₁ + C₂ + 1 with hk_def
    have hlt := lt_Q k
    have hC1 : C₁ < Q k := by omega
    have hC2 : C₂ < Q k := by omega
    have hmark : 4 * Q k ∈ setA := marker_mem k
    rcases hpart _ hmark with hcase | hcase
    · -- marker in A₁, so not in A₂
      have hnot : 4 * Q k ∉ A₂ := by
        intro hh
        have hmem : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hcase, hh⟩
        rw [hdisj] at hmem
        exact hmem
      obtain ⟨m, hmS, hmI⟩ := h₂ (9 * Q k)
      rw [Set.mem_Icc] at hmI
      exact gap k A₂ hA2 hnot m (by omega) (by omega) hmS
    · -- marker in A₂, so not in A₁
      have hnot : 4 * Q k ∉ A₁ := by
        intro hh
        have hmem : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hh, hcase⟩
        rw [hdisj] at hmem
        exact hmem
      obtain ⟨m, hmS, hmI⟩ := h₁ (9 * Q k)
      rw [Set.mem_Icc] at hmI
      exact gap k A₁ hA1 hnot m (by omega) (by omega) hmS

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## Construction

`Q k = 5^k`. Stage `k` contributes a connector `4*Q k`, a block `B_k = [5Qk, 6Qk-1]`,
and a fat block `F_k = [10Qk-1, 15Qk]`. `setA = {2,3} ∪ ⋃ k stage k`.

Basis: `[4, 6*Q k] ⊆ A+A` by induction; the interval `I_k = [2Qk,3Qk]` sits inside
`F_{k-1}` (or `{2,3}` for k=0), and the seven pair-types
`c+I, I+B, c+B, B+B, I+F, B+F, F+F` tile `(6Qk, 30Qk]`.

Partition: any sum landing in the window `[9Qk, 10Qk)` must use the connector `4*Q k`
as one summand (rigidity). If both halves of a partition had syndetic sumsets with
constants `C₁, C₂`, take `k = C₁+C₂+1` so `Q k > C₁, C₂`: each half's sumset must meet
`[9Qk, 9Qk+Cᵢ] ⊆ [9Qk, 10Qk)`, forcing `4*Q k` into BOTH halves — contradicting
disjointness. -/

def Q (k : ℕ) : ℕ := 5 ^ k

def stage (k : ℕ) : Set ℕ :=
  {4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

lemma Q_pos (k : ℕ) : 1 ≤ Q k := Nat.one_le_pow k 5 (by norm_num)

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

lemma Q_step {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  calc 5 * Q j = Q (j + 1) := (Q_succ j).symm
    _ ≤ Q k := Q_mono h

lemma Q_gt (k : ℕ) : k < Q k := by
  have h1 : k < 2 ^ k := Nat.lt_two_pow_self
  have h2 : 2 ^ k ≤ 5 ^ k := Nat.pow_le_pow_left (by norm_num) k
  unfold Q; omega

/-! ## Membership helpers -/

lemma stage_subset (k : ℕ) : stage k ⊆ setA :=
  fun _ hx => Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, hx⟩)

lemma two_mem : (2 : ℕ) ∈ setA := Set.mem_union_left _ (Set.mem_insert 2 {3})

lemma three_mem : (3 : ℕ) ∈ setA :=
  Set.mem_union_left _ (Set.mem_insert_of_mem 2 rfl)

lemma c_mem (k : ℕ) : 4 * Q k ∈ setA :=
  stage_subset k (Set.mem_union_left _ (Set.mem_union_left _ rfl))

lemma B_mem {k x : ℕ} (h1 : 5 * Q k ≤ x) (h2 : x ≤ 6 * Q k - 1) : x ∈ setA :=
  stage_subset k (Set.mem_union_left _ (Set.mem_union_right _ (Set.mem_Icc.mpr ⟨h1, h2⟩)))

lemma F_mem {k x : ℕ} (h1 : 10 * Q k - 1 ≤ x) (h2 : x ≤ 15 * Q k) : x ∈ setA :=
  stage_subset k (Set.mem_union_right _ (Set.mem_Icc.mpr ⟨h1, h2⟩))

lemma I_mem {k x : ℕ} (h1 : 2 * Q k ≤ x) (h2 : x ≤ 3 * Q k) : x ∈ setA := by
  cases k with
  | zero =>
      have h0 : Q 0 = 1 := by norm_num [Q]
      have hx : x = 2 ∨ x = 3 := by omega
      rcases hx with h | h
      · rw [h]; exact two_mem
      · rw [h]; exact three_mem
  | succ j =>
      have hQ : Q (j + 1) = 5 * Q j := Q_succ j
      have hq := Q_pos j
      exact F_mem (k := j) (by omega) (by omega)

/-! ## Structure lemmas -/

lemma setA_elt {x : ℕ} (h : x ∈ setA) : x = 2 ∨ x = 3 ∨ ∃ j, x ∈ stage j := by
  simp only [setA, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_iUnion] at h
  tauto

lemma stage_bound {j x : ℕ} (h : x ∈ stage j) : 4 * Q j ≤ x ∧ x ≤ 15 * Q j := by
  have hq := Q_pos j
  simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at h
  rcases h with (h | h) | h <;> omega

lemma setA_ge_two {x : ℕ} (h : x ∈ setA) : 2 ≤ x := by
  rcases setA_elt h with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have hb := stage_bound hj
    have hq := Q_pos j
    omega

/-! ## Basis half -/

lemma basis_lem (k : ℕ) :
    ∀ x : ℕ, 4 ≤ x → x ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = x := by
  induction k with
  | zero =>
      intro x h1 h2
      have h0 : Q 0 = 1 := by norm_num [Q]
      have hx : x = 4 ∨ x = 5 ∨ x = 6 := by omega
      rcases hx with h | h | h
      · exact ⟨2, two_mem, 2, two_mem, by omega⟩
      · exact ⟨2, two_mem, 3, three_mem, by omega⟩
      · exact ⟨3, three_mem, 3, three_mem, by omega⟩
  | succ k ih =>
      intro x h1 h2
      have hQ : Q (k + 1) = 5 * Q k := Q_succ k
      have hq := Q_pos k
      by_cases c0 : x ≤ 6 * Q k
      · exact ih x h1 c0
      · by_cases c1 : x ≤ 7 * Q k
        · exact ⟨4 * Q k, c_mem k, x - 4 * Q k,
            I_mem (k := k) (by omega) (by omega), by omega⟩
        · by_cases c2 : x ≤ 9 * Q k - 1
          · exact ⟨max (2 * Q k) (x - (6 * Q k - 1)),
              I_mem (k := k) (by omega) (by omega),
              x - max (2 * Q k) (x - (6 * Q k - 1)),
              B_mem (k := k) (by omega) (by omega), by omega⟩
          · by_cases c3 : x ≤ 10 * Q k - 1
            · exact ⟨4 * Q k, c_mem k, x - 4 * Q k,
                B_mem (k := k) (by omega) (by omega), by omega⟩
            · by_cases c4 : x ≤ 12 * Q k - 2
              · exact ⟨max (5 * Q k) (x - (6 * Q k - 1)),
                  B_mem (k := k) (by omega) (by omega),
                  x - max (5 * Q k) (x - (6 * Q k - 1)),
                  B_mem (k := k) (by omega) (by omega), by omega⟩
              · by_cases c5 : x ≤ 18 * Q k
                · exact ⟨max (2 * Q k) (x - 15 * Q k),
                    I_mem (k := k) (by omega) (by omega),
                    x - max (2 * Q k) (x - 15 * Q k),
                    F_mem (k := k) (by omega) (by omega), by omega⟩
                · by_cases c6 : x ≤ 21 * Q k - 1
                  · exact ⟨max (5 * Q k) (x - 15 * Q k),
                      B_mem (k := k) (by omega) (by omega),
                      x - max (5 * Q k) (x - 15 * Q k),
                      F_mem (k := k) (by omega) (by omega), by omega⟩
                  · exact ⟨max (10 * Q k - 1) (x - 15 * Q k),
                      F_mem (k := k) (by omega) (by omega),
                      x - max (10 * Q k - 1) (x - 15 * Q k),
                      F_mem (k := k) (by omega) (by omega), by omega⟩

/-! ## Rigidity half -/

lemma classify {k x : ℕ} (h : x ∈ setA) (hx : x < 10 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨
      x = 10 * Q k - 1 := by
  have hqk := Q_pos k
  rcases setA_elt h with h2 | h3 | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases Nat.lt_trichotomy j k with hjk | hjk | hjk
    · have hb := stage_bound hj
      have h5 : 5 * Q j ≤ Q k := Q_step hjk
      left; omega
    · subst hjk
      simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at hj
      rcases hj with (h | h) | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr (Or.inl h))
      · right; right; right; omega
    · exfalso
      have hb := stage_bound hj
      have h5 : 5 * Q k ≤ Q j := Q_step hjk
      omega

lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (h1 : 9 * Q k ≤ a + b) (h2 : a + b < 10 * Q k) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hq := Q_pos k
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have hca := classify (k := k) ha (by omega)
  have hcb := classify (k := k) hb (by omega)
  rcases hca with h | h | h | h
  · rcases hcb with h' | h' | h' | h'
    · exfalso; omega
    · exact Or.inr h'
    · exfalso; omega
    · exfalso; omega
  · exact Or.inl h
  · rcases hcb with h' | h' | h' | h'
    · exfalso; omega
    · exact Or.inr h'
    · exfalso; omega
    · exfalso; omega
  · exfalso; omega

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
    have h := Q_gt n
    exact basis_lem n n hn (by omega)
  · intro A₁ A₂ hA₁ hA₂ _hcover hdisj
    rintro ⟨hS₁, hS₂⟩
    unfold IsSyndetic at hS₁ hS₂
    obtain ⟨C₁, hC₁⟩ := hS₁
    obtain ⟨C₂, hC₂⟩ := hS₂
    have hk : C₁ + C₂ + 1 < Q (C₁ + C₂ + 1) := Q_gt _
    have key : ∀ (T : Set ℕ) (C : ℕ), T ⊆ setA → C < Q (C₁ + C₂ + 1) →
        (∀ x : ℕ, ∃ m ∈ T + T, m ∈ Icc x (x + C)) →
        4 * Q (C₁ + C₂ + 1) ∈ T := by
      intro T C hT hC hsyn
      obtain ⟨m, hm, hmI⟩ := hsyn (9 * Q (C₁ + C₂ + 1))
      rw [Set.mem_Icc] at hmI
      rw [Set.mem_add] at hm
      obtain ⟨a, haT, b, hbT, hab⟩ := hm
      rcases rigidity (k := C₁ + C₂ + 1) (hT haT) (hT hbT)
        (by omega) (by omega) with h | h
      · rw [← h]; exact haT
      · rw [← h]; exact hbT
    have h1 : 4 * Q (C₁ + C₂ + 1) ∈ A₁ := key A₁ C₁ hA₁ (by omega) hC₁
    have h2 : 4 * Q (C₁ + C₂ + 1) ∈ A₂ := key A₂ C₂ hA₂ (by omega) hC₂
    have hmem : 4 * Q (C₁ + C₂ + 1) ∈ A₁ ∩ A₂ := ⟨h1, h2⟩
    rw [hdisj] at hmem
    exact hmem

end Erdos741OAI

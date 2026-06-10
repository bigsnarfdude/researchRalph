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

def S (k : ℕ) : Set ℕ :=
  ({4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1)) ∪ Icc (10 * Q k - 1) (15 * Q k)

def A : Set ℕ := {2, 3} ∪ ⋃ k, S k

/-! ## Power facts -/

lemma Q_pos (k : ℕ) : 1 ≤ Q k := by
  simp only [Q]; exact Nat.one_le_pow k 5 (by norm_num)

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ]; ring

lemma Q_step_le {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  simp only [Q]
  have h1 : (5 : ℕ) ^ (j + 1) ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) h
  rw [pow_succ] at h1; omega

lemma Q_ge {k j : ℕ} (h : k < j) : Q (k + 1) ≤ Q j := by
  simp only [Q]; exact Nat.pow_le_pow_right (by norm_num) h

/-! ## Membership helpers -/

lemma hp (k : ℕ) : 4 * Q k ∈ A := by
  refine Or.inr (Set.mem_iUnion.2 ⟨k, ?_⟩)
  simp only [S, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc]
  exact Or.inl (Or.inl rfl)

lemma hLmem (k x : ℕ) (h1 : 5 * Q k ≤ x) (h2 : x ≤ 6 * Q k - 1) : x ∈ A := by
  refine Or.inr (Set.mem_iUnion.2 ⟨k, ?_⟩)
  simp only [S, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc]
  exact Or.inl (Or.inr ⟨h1, h2⟩)

lemma hHmem (k x : ℕ) (h1 : 10 * Q k - 1 ≤ x) (h2 : x ≤ 15 * Q k) : x ∈ A := by
  refine Or.inr (Set.mem_iUnion.2 ⟨k, ?_⟩)
  simp only [S, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc]
  exact Or.inr ⟨h1, h2⟩

lemma h2A : (2 : ℕ) ∈ A := by refine Or.inl ?_; simp
lemma h3A : (3 : ℕ) ∈ A := by refine Or.inl ?_; simp

/-! ## Lower bound and band classification -/

lemma two_le (z : ℕ) (hz : z ∈ A) : 2 ≤ z := by
  rcases hz with h | hU
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h; omega
  · rw [Set.mem_iUnion] at hU
    obtain ⟨j, hj⟩ := hU
    have hq : 1 ≤ Q j := Q_pos j
    simp only [S, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at hj
    rcases hj with (h | h) | h <;> omega

lemma classify (k z : ℕ) (hz : z ∈ A) (hlt : z < 10 * Q k) :
    z ≤ 3 * Q k ∨ z = 4 * Q k ∨ (5 * Q k ≤ z ∧ z ≤ 6 * Q k - 1) ∨ z = 10 * Q k - 1 := by
  have hqk : 1 ≤ Q k := Q_pos k
  rcases hz with h | hU
  · left
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h; omega
  · rw [Set.mem_iUnion] at hU
    obtain ⟨j, hj⟩ := hU
    have hqj : 1 ≤ Q j := Q_pos j
    simp only [S, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at hj
    rcases lt_trichotomy j k with hlt' | rfl | hgt
    · left
      have h5 : 5 * Q j ≤ Q k := Q_step_le hlt'
      rcases hj with (h | h) | h <;> omega
    · rcases hj with (h | h) | h
      · right; left; omega
      · right; right; left; omega
      · right; right; right; omega
    · exfalso
      have hge : Q (k + 1) ≤ Q j := Q_ge hgt
      have hr : Q (k + 1) = 5 * Q k := Q_succ k
      have hz4 : 4 * Q j ≤ z := by rcases hj with (h | h) | h <;> omega
      omega

/-! ## Interval Minkowski-sum cover -/

lemma icc_add (a b c d n : ℕ) (h1 : a + c ≤ n) (h2 : n ≤ b + d)
    (hab : a ≤ b) (hcd : c ≤ d) :
    ∃ x y, a ≤ x ∧ x ≤ b ∧ c ≤ y ∧ y ≤ d ∧ x + y = n := by
  refine ⟨max a (n - d), n - max a (n - d), ?_, ?_, ?_, ?_, ?_⟩ <;> omega

lemma pair_to_A (x1 x2 y1 y2 n : ℕ)
    (hx : ∀ x, x1 ≤ x → x ≤ x2 → x ∈ A)
    (hy : ∀ y, y1 ≤ y → y ≤ y2 → y ∈ A)
    (h1 : x1 + y1 ≤ n) (h2 : n ≤ x2 + y2) (hx12 : x1 ≤ x2) (hy12 : y1 ≤ y2) :
    n ∈ A + A := by
  obtain ⟨x, y, ha, hb, hc, hd, hxy⟩ := icc_add x1 x2 y1 y2 n h1 h2 hx12 hy12
  exact Set.mem_add.2 ⟨x, hx x ha hb, y, hy y hc hd, hxy⟩

lemma hsmall (x : ℕ) (h1 : 2 ≤ x) (h2 : x ≤ 5) : x ∈ A := by
  have hq0 : Q 0 = 1 := by simp [Q]
  interval_cases x
  · exact h2A
  · exact h3A
  · simpa [hq0] using hp 0
  · exact hLmem 0 5 (by rw [hq0]; omega) (by rw [hq0]; omega)

/-! ## Basis: every n ≥ 4 is a sum of two elements -/

lemma basis_up_to (k : ℕ) : ∀ n, 4 ≤ n → n ≤ 30 * Q k → n ∈ A + A := by
  induction k with
  | zero =>
    intro n hn4 hn30
    have hq0 : Q 0 = 1 := by simp [Q]
    rw [hq0] at hn30
    by_cases c1 : n ≤ 10
    · exact pair_to_A 2 5 2 5 n hsmall hsmall (by omega) (by omega) (by omega) (by omega)
    · by_cases c2 : n ≤ 20
      · exact pair_to_A 2 5 9 15 n hsmall
          (fun y _ _ => hHmem 0 y (by rw [hq0]; omega) (by rw [hq0]; omega))
          (by omega) (by omega) (by omega) (by omega)
      · exact pair_to_A 9 15 9 15 n
          (fun x _ _ => hHmem 0 x (by rw [hq0]; omega) (by rw [hq0]; omega))
          (fun y _ _ => hHmem 0 y (by rw [hq0]; omega) (by rw [hq0]; omega))
          (by omega) (by omega) (by omega) (by omega)
  | succ k ih =>
    intro n hn4 hn30
    by_cases hle : n ≤ 30 * Q k
    · exact ih n hn4 hle
    · push_neg at hle
      have hr : Q (k + 1) = 5 * Q k := Q_succ k
      have hqk : 1 ≤ Q k := Q_pos k
      have hpt : ∀ x, 4 * Q (k + 1) ≤ x → x ≤ 4 * Q (k + 1) → x ∈ A :=
        fun x ha hb => by
          have hx : x = 4 * Q (k + 1) := le_antisymm hb ha
          rw [hx]; exact hp (k + 1)
      by_cases ca : n ≤ 7 * Q (k + 1)
      · exact pair_to_A (4 * Q (k + 1)) (4 * Q (k + 1)) (10 * Q k - 1) (15 * Q k) n
          hpt (hHmem k) (by omega) (by omega) (by omega) (by omega)
      · by_cases cb : n ≤ 9 * Q (k + 1) - 1
        · exact pair_to_A (5 * Q (k + 1)) (6 * Q (k + 1) - 1) (10 * Q k - 1) (15 * Q k) n
            (hLmem (k + 1)) (hHmem k) (by omega) (by omega) (by omega) (by omega)
        · by_cases cc : n ≤ 10 * Q (k + 1) - 1
          · exact pair_to_A (4 * Q (k + 1)) (4 * Q (k + 1)) (5 * Q (k + 1)) (6 * Q (k + 1) - 1) n
              hpt (hLmem (k + 1)) (by omega) (by omega) (by omega) (by omega)
          · by_cases cd : n ≤ 12 * Q (k + 1) - 2
            · exact pair_to_A (5 * Q (k + 1)) (6 * Q (k + 1) - 1) (5 * Q (k + 1)) (6 * Q (k + 1) - 1) n
                (hLmem (k + 1)) (hLmem (k + 1)) (by omega) (by omega) (by omega) (by omega)
            · by_cases ce : n ≤ 18 * Q (k + 1)
              · exact pair_to_A (10 * Q k - 1) (15 * Q k) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) n
                  (hHmem k) (hHmem (k + 1)) (by omega) (by omega) (by omega) (by omega)
              · by_cases cf : n ≤ 19 * Q (k + 1)
                · exact pair_to_A (4 * Q (k + 1)) (4 * Q (k + 1)) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) n
                    hpt (hHmem (k + 1)) (by omega) (by omega) (by omega) (by omega)
                · by_cases cg : n ≤ 21 * Q (k + 1) - 1
                  · exact pair_to_A (5 * Q (k + 1)) (6 * Q (k + 1) - 1) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) n
                      (hLmem (k + 1)) (hHmem (k + 1)) (by omega) (by omega) (by omega) (by omega)
                  · exact pair_to_A (10 * Q (k + 1) - 1) (15 * Q (k + 1)) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) n
                      (hHmem (k + 1)) (hHmem (k + 1)) (by omega) (by omega) (by omega) (by omega)

/-! ## Rigidity: sums in [9Qk, 10Qk-1] force the isolated point 4Qk -/

lemma rigidity (k n : ℕ) (hn1 : 9 * Q k ≤ n) (hn2 : n ≤ 10 * Q k - 1)
    (x y : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hxy : x + y = n) :
    x = 4 * Q k ∨ y = 4 * Q k := by
  have hqk : 1 ≤ Q k := Q_pos k
  have hx2 := two_le x hx
  have hy2 := two_le y hy
  have hxlt : x < 10 * Q k := by omega
  have hylt : y < 10 * Q k := by omega
  rcases classify k x hx hxlt with hxb | hxb | hxb | hxb <;>
    rcases classify k y hy hylt with hyb | hyb | hyb | hyb <;> omega

/-! ## Main theorem -/

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨A, ?_, ?_⟩
  · intro n hn4
    have hk : n ≤ 30 * Q n := by
      have h1 : n < 2 ^ n := Nat.lt_two_pow_self
      have h2 : (2 : ℕ) ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
      have hlt : n < Q n := by simp only [Q]; omega
      omega
    exact Set.mem_add.1 (basis_up_to n n hn4 hk)
  · intro A₁ A₂ hsub1 hsub2 hcov hdisj
    rintro ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    set k := C₁ + C₂ + 1 with hkdef
    have hqk : 1 ≤ Q k := Q_pos k
    have hkk : k < Q k := by
      have h1 : k < 2 ^ k := Nat.lt_two_pow_self
      have h2 : (2 : ℕ) ^ k ≤ 5 ^ k := Nat.pow_le_pow_left (by norm_num) k
      simp only [Q]; omega
    have hpk : 4 * Q k ∈ A := hp k
    rcases hcov _ hpk with hin1 | hin2
    · obtain ⟨m, hmem, hIcc⟩ := hC2 (9 * Q k)
      rw [Set.mem_add] at hmem
      obtain ⟨x, hx, y, hy, hxy⟩ := hmem
      rw [Set.mem_Icc] at hIcc
      have hm1 : 9 * Q k ≤ m := hIcc.1
      have hm2 : m ≤ 10 * Q k - 1 := by omega
      rcases rigidity k m hm1 hm2 x y (hsub2 hx) (hsub2 hy) hxy with h | h
      · have hmem2 : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hin1, h ▸ hx⟩
        rw [hdisj] at hmem2; exact hmem2
      · have hmem2 : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hin1, h ▸ hy⟩
        rw [hdisj] at hmem2; exact hmem2
    · obtain ⟨m, hmem, hIcc⟩ := hC1 (9 * Q k)
      rw [Set.mem_add] at hmem
      obtain ⟨x, hx, y, hy, hxy⟩ := hmem
      rw [Set.mem_Icc] at hIcc
      have hm1 : 9 * Q k ≤ m := hIcc.1
      have hm2 : m ≤ 10 * Q k - 1 := by omega
      rcases rigidity k m hm1 hm2 x y (hsub1 hx) (hsub1 hy) hxy with h | h
      · have hmem2 : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h ▸ hx, hin2⟩
        rw [hdisj] at hmem2; exact hmem2
      · have hmem2 : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h ▸ hy, hin2⟩
        rw [hdisj] at hmem2; exact hmem2

end Erdos741OAI

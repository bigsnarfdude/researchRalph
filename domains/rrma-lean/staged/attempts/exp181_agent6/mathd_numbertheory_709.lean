import Mathlib
set_option maxHeartbeats 200000000
open BigOperators Real Nat Topology Rat

lemma factorization_eq_zero_of_not_mem_primeFactors {m p : ℕ} (hm : m ≠ 0)
    (h : p ∉ m.primeFactors) : m.factorization p = 0 := by
  rw [Nat.factorization_eq_zero_iff]
  by_contra h'; push_neg at h'
  exact h (Nat.mem_primeFactors.mpr ⟨h'.1, h'.2.1, hm⟩)

lemma prod_primeFactors_union_singleton (p : ℕ) (m : ℕ) (hm : m ≠ 0) :
    ∏ q ∈ m.primeFactors, (m.factorization q + 1) =
    ∏ q ∈ m.primeFactors ∪ {p}, (m.factorization q + 1) := by
  apply Finset.prod_subset (Finset.subset_union_left (s₂ := {p}))
  intro q hq hqn
  simp only [Finset.mem_union, Finset.mem_singleton] at hq
  rcases hq with h | rfl
  · exact absurd h hqn
  · simp [factorization_eq_zero_of_not_mem_primeFactors hm hqn]

lemma card_divisors_prime_mul (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : m ≠ 0) :
    (p * m).divisors.card * (m.factorization p + 1) =
    m.divisors.card * (m.factorization p + 2) := by
  have hpm : p * m ≠ 0 := mul_ne_zero hp.ne_zero hm
  rw [Nat.card_divisors hpm, Nat.card_divisors hm]
  have hpf : (p * m).primeFactors = m.primeFactors ∪ {p} := by
    rw [Nat.primeFactors_mul hp.ne_zero hm, Nat.Prime.primeFactors hp, Finset.union_comm]
  have hf_p : (p * m).factorization p = m.factorization p + 1 := by
    rw [Nat.factorization_mul hp.ne_zero hm, Finsupp.add_apply,
        Nat.Prime.factorization_self hp]; omega
  have hf_eq (q : ℕ) (hq : q ≠ p) : (p * m).factorization q = m.factorization q := by
    rw [Nat.factorization_mul hp.ne_zero hm, Finsupp.add_apply,
        Nat.Prime.factorization hp]
    simp [Finsupp.single_apply, hq]
  rw [hpf, prod_primeFactors_union_singleton p m hm]
  have hp_mem : p ∈ m.primeFactors ∪ {p} :=
    Finset.mem_union_right _ (Finset.mem_singleton.mpr rfl)
  rw [← Finset.mul_prod_erase _ _ hp_mem, ← Finset.mul_prod_erase _ _ hp_mem, hf_p]
  -- Goal: (m.factorization p + 1 + 1) * ∏ ... * (m.factorization p + 1) =
  --       (m.factorization p + 1) * ∏ ... * (m.factorization p + 2)
  -- The products over the erased set are equal since factorizations agree for q ≠ p
  have hrest : ∀ q ∈ (m.primeFactors ∪ {p}).erase p,
      (p * m).factorization q + 1 = m.factorization q + 1 := by
    intro q hq
    rw [hf_eq q (by intro h; rw [h] at hq; simp at hq)]
  rw [Finset.prod_congr rfl hrest]
  ring

theorem mathd_numbertheory_709 (n : ℕ) (h₀ : 0 < n) (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30) : Finset.card (Nat.divisors (6 * n)) = 35 := by
  have hn : n ≠ 0 := by omega
  set a := n.factorization 2 with ha_def
  set b := n.factorization 3 with hb_def
  set d6 := (6 * n).divisors.card with hd6_def
  set dn := n.divisors.card with hdn_def
  -- Relation 1: 28 * (a+1) = d(n) * (a+2)
  have h_2n : 28 * (a + 1) = dn * (a + 2) := by
    rw [hdn_def, ← h₁]; exact card_divisors_prime_mul 2 (by norm_num) n hn
  -- Relation 2: 30 * (b+1) = d(n) * (b+2)
  have h_3n : 30 * (b + 1) = dn * (b + 2) := by
    rw [hdn_def, ← h₂]; exact card_divisors_prime_mul 3 (by norm_num) n hn
  -- Relation 3: d(6n) * (a+1) = 30 * (a+2)
  have hv2_3n : (3 * n).factorization 2 = a := by
    rw [Nat.factorization_mul (by norm_num) hn, Finsupp.add_apply, ha_def,
        Nat.Prime.factorization (by norm_num : Nat.Prime 3)]
    simp [Finsupp.single_apply]
  have h_6n_a : d6 * (a + 1) = 30 * (a + 2) := by
    have := card_divisors_prime_mul 2 (by norm_num) (3 * n) (by omega)
    rw [hv2_3n, h₂] at this
    rw [hd6_def, show 6 * n = 2 * (3 * n) from by ring]; exact this
  -- Relation 4: d(6n) * (b+1) = 28 * (b+2)
  have hv3_2n : (2 * n).factorization 3 = b := by
    rw [Nat.factorization_mul (by norm_num) hn, Finsupp.add_apply, hb_def,
        Nat.Prime.factorization (by norm_num : Nat.Prime 2)]
    simp [Finsupp.single_apply]
  have h_6n_b : d6 * (b + 1) = 28 * (b + 2) := by
    have := card_divisors_prime_mul 3 (by norm_num) (2 * n) (by omega)
    rw [hv3_2n, h₁] at this
    rw [hd6_def, show 6 * n = 3 * (2 * n) from by ring]; exact this
  -- Divisibility: (a+2) | 28
  have ha2_div : (a + 2) ∣ 28 := by
    have h1 : (a + 2) ∣ 28 * (a + 1) := ⟨dn, by linarith⟩
    have h2 : (a + 2) ∣ 28 * (a + 2) := dvd_mul_left _ _
    have := Nat.dvd_sub h2 h1
    rwa [show 28 * (a + 2) - 28 * (a + 1) = 28 from by omega] at this
  -- Divisibility: (b+2) | 30
  have hb2_div : (b + 2) ∣ 30 := by
    have h1 : (b + 2) ∣ 30 * (b + 1) := ⟨dn, by linarith⟩
    have h2 : (b + 2) ∣ 30 * (b + 2) := dvd_mul_left _ _
    have := Nat.dvd_sub h2 h1
    rwa [show 30 * (b + 2) - 30 * (b + 1) = 30 from by omega] at this
  -- Cross relation
  have h_cross : 30 * (a + 2) * (b + 1) = 28 * (b + 2) * (a + 1) := by nlinarith
  -- Bounds
  have ha_le : a ≤ 26 := by have := Nat.le_of_dvd (by norm_num) ha2_div; omega
  have hb_le : b ≤ 28 := by have := Nat.le_of_dvd (by norm_num) hb2_div; omega
  -- Enumerate a: (a+2) | 28 means a ∈ {0, 2, 5, 12, 26} (a+2 ∈ {2, 4, 7, 14, 28})
  -- For each a, h_cross determines b (linear in b after fixing a), then h_6n_a gives d6.
  -- omega handles linear arithmetic after interval_cases substitutes concrete a values.
  interval_cases a <;> omega

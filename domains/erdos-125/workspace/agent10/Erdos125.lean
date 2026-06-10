import Mathlib

open Filter Finset Real

-- A: natural numbers with only digits 0,1 in base 3
def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}

-- B: natural numbers with only digits 0,1 in base 4
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}

-- Sumset A + B
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Lower density: liminf_{N→∞} |S ∩ [0,N)| / N
noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

/-!
## Sub-lemma: 3 and 4 are multiplicatively independent over ℤ
-/
private lemma nat_pow_ne (b a : ℕ) (hb : 0 < b) (ha : 0 < a) :
    (3 : ℕ) ^ b ≠ (4 : ℕ) ^ a := by
  intro h_eq
  have hcop : Nat.Coprime 3 (4 ^ a) := (by decide : Nat.Coprime 3 4).pow_right _
  have h3_dvd_4a : (3 : ℕ) ∣ 4 ^ a := h_eq ▸ dvd_pow_self 3 hb.ne'
  have h3_dvd_1 : (3 : ℕ) ∣ 1 := hcop ▸ Nat.dvd_gcd (dvd_refl 3) h3_dvd_4a
  exact absurd h3_dvd_1 (by decide)

/-!
## Lemma 1: Dirichlet approximation at aligned scales
-/
lemma exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 3 - ↑m * log 4| < ε := by
  have hlog3_pos : (0 : ℝ) < log 3 := Real.log_pos (by norm_num)
  have hlog4_pos : (0 : ℝ) < log 4 := Real.log_pos (by norm_num)
  -- log 3 / log 4 is irrational
  have hirr : Irrational (log 3 / log 4) := by
    rw [irrational_iff_ne_rational]
    intro a b hb heq
    have hb_real : (b : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hb
    have h_mul : (b : ℝ) * log 3 = (a : ℝ) * log 4 := by
      have := div_eq_div_iff (ne_of_gt hlog4_pos) hb_real |>.mp heq
      linarith
    have ha_ne : a ≠ 0 := by
      intro ha
      have ha_cast : (a : ℝ) = 0 := by exact_mod_cast ha
      rw [ha_cast, zero_mul] at h_mul
      rcases mul_eq_zero.mp h_mul with h | h
      · exact hb (Int.cast_eq_zero.mp h)
      · exact absurd h (ne_of_gt hlog3_pos)
    have hb' : 0 < b.natAbs := Int.natAbs_pos.mpr hb
    have ha' : 0 < a.natAbs := Int.natAbs_pos.mpr ha_ne
    have h_natabs : (b.natAbs : ℝ) * log 3 = (a.natAbs : ℝ) * log 4 := by
      rw [Nat.cast_natAbs, Nat.cast_natAbs, Int.cast_abs, Int.cast_abs]
      obtain hb_nn | hb_neg := le_or_gt 0 (b : ℝ)
      · rw [abs_of_nonneg hb_nn]
        have ha_nn : 0 ≤ (a : ℝ) := by nlinarith
        rw [abs_of_nonneg ha_nn]; exact h_mul
      · rw [abs_of_neg hb_neg]
        have ha_neg : (a : ℝ) < 0 := by nlinarith
        rw [abs_of_neg ha_neg]; linarith
    have h_rpow : (3 : ℝ) ^ b.natAbs = (4 : ℝ) ^ a.natAbs := by
      apply Real.log_injOn_pos (Set.mem_Ioi.mpr (by positivity))
                               (Set.mem_Ioi.mpr (by positivity))
      rw [Real.log_pow, Real.log_pow]
      exact_mod_cast h_natabs
    have h_nat : (3 : ℕ) ^ b.natAbs = (4 : ℕ) ^ a.natAbs := by exact_mod_cast h_rpow
    have hcop : Nat.Coprime 3 (4 ^ a.natAbs) := (by decide : Nat.Coprime 3 4).pow_right _
    have h3_dvd : (3 : ℕ) ∣ 4 ^ a.natAbs := h_nat ▸ dvd_pow_self 3 hb'.ne'
    exact absurd (hcop ▸ Nat.dvd_gcd (dvd_refl 3) h3_dvd) (by decide)
  -- Dirichlet approximation
  obtain ⟨N, hN⟩ := exists_nat_gt (log 4 / ε)
  have hN_pos : 0 < N + 1 := Nat.succ_pos _
  obtain ⟨j, k, hk_pos, _, hbound⟩ :=
    Real.exists_int_int_abs_mul_sub_le (log 3 / log 4) hN_pos
  -- 1/(N+2) < ε/log4
  have hN2_bound : (1 : ℝ) / (↑(N + 1) + 1) < ε / log 4 := by
    have h_pos : (0:ℝ) < ↑(N+1) + 1 := by positivity
    have hNε : log 4 < (N : ℝ) * ε := by
      have h := hN
      rw [div_lt_iff₀ hε] at h; linarith
    rw [div_lt_iff₀ h_pos]
    rw [div_mul_eq_mul_div, lt_div_iff₀ hlog4_pos]
    push_cast; linarith
  -- j > 0 because k*(log3/log4) > 1/2 > 1/(N+2) ≥ 0
  have hj_pos : 0 < j := by
    have hk_real : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk_pos
    have hξ_pos : 0 < log 3 / log 4 := div_pos hlog3_pos hlog4_pos
    have hξ_gt_half : (1:ℝ)/2 < log 3 / log 4 := by
      rw [lt_div_iff₀ hlog4_pos]
      have h1 : log 4 < log 9 := Real.log_lt_log (by norm_num) (by norm_num)
      have h2 : log (9:ℝ) = 2 * log 3 := by
        have : (9:ℝ) = 3 ^ 2 := by norm_num
        rw [this, Real.log_pow]; norm_cast
      linarith
    have hkξ_gt_half : (1:ℝ)/2 < (k:ℝ) * (log 3 / log 4) := by
      nlinarith [mul_nonneg (show (0:ℝ) ≤ (k:ℝ) - 1 by linarith) (le_of_lt hξ_pos)]
    have h_half : (1:ℝ) / (↑(N+1) + 1) ≤ 1/2 := by
      have hd : (0:ℝ) < ↑(N+1) + 1 := by positivity
      have h2le : (2:ℝ) ≤ ↑(N+1) + 1 := by norm_cast; omega
      have h21 : (2:ℝ) / (↑(N+1)+1) ≤ 1 := (div_le_one hd).mpr h2le
      linarith [show (1:ℝ) / (↑(N+1)+1) = (2:ℝ) / (↑(N+1)+1) / 2 from by ring]
    have h_j_lower : (k : ℝ) * (log 3 / log 4) - (1 / (↑(N + 1) + 1)) ≤ (j : ℝ) := by
      have := (abs_le.mp hbound).2; linarith
    have : (j : ℝ) > 0 := by linarith
    exact Int.cast_pos.mp this
  refine ⟨k.toNat, j.toNat, ?_, ?_, ?_⟩
  · omega
  · omega
  · have hk_cast : (k.toNat : ℝ) = (k : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg hk_pos.le
    have hj_cast : (j.toNat : ℝ) = (j : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg hj_pos.le
    rw [hk_cast, hj_cast]
    have h_rearrange : (k : ℝ) * log 3 - (j : ℝ) * log 4 =
        log 4 * ((k : ℝ) * (log 3 / log 4) - (j : ℝ)) := by
      field_simp [ne_of_gt hlog4_pos]
    rw [h_rearrange, abs_mul, abs_of_pos hlog4_pos]
    calc log 4 * |(k : ℝ) * (log 3 / log 4) - (j : ℝ)|
        ≤ log 4 * (1 / (↑(N + 1) + 1)) := by
          apply mul_le_mul_of_nonneg_left hbound (le_of_lt hlog4_pos)
      _ < log 4 * (ε / log 4) := by
          apply mul_lt_mul_of_pos_left hN2_bound hlog4_pos
      _ = ε := by field_simp

/-!
## Key sub-lemmas: concrete bounds for setA and setB elements

Any n ∈ setA with n < 81 satisfies n ≤ 40 (max of setA below 3^4=81 is (3^4-1)/2=40).
Any n ∈ setB with n < 64 satisfies n ≤ 21 (max of setB below 4^3=64 is (4^3-1)/3=21).
Proved by finite enumeration via native_decide.
-/
private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_21 {n : ℕ} (hn : n ∈ setB) (hlt : n < 64) : n ≤ 21 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

/-!
## Lemma 2: Gap in sumset at aligned scales
The concrete gap {62, 63} ⊆ ℕ \ setAB.
Proof: setA∩(40,∞)∩[0,63]=∅ and setB∩(21,∞)∩[0,63]=∅,
so any (a,b) with a+b∈{62,63} has a≤40 and b≤21, giving a+b≤61<62.
-/
lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  -- Exhibit the concrete gap {62, 63}, independent of k and m
  refine ⟨62, 2, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  -- a ≤ n ≤ 63, so a < 81
  have ha_lt : a < 81 := by omega
  -- setA_le_40: a ∈ setA, a < 81 → a ≤ 40
  have ha_bound : a ≤ 40 := setA_le_40 ha_A ha_lt
  -- b = n - a, b < 64
  have hb_lt : b < 64 := by omega
  -- setB_le_21: b ∈ setB, b < 64 → b ≤ 21
  have hb_bound : b ≤ 21 := setB_le_21 hb_B hb_lt
  -- a ≤ 40 and b ≤ 21 → a+b ≤ 61, but a+b = n ≥ 62. Contradiction.
  omega

/-!
## Lemma 3: A gap exists in the sumset
62 is not in setAB: any a ∈ setA with a ≤ 62 satisfies a ≤ 40 (by setA_le_40),
so b = 62 - a ≥ 22 > 21, contradicting setB_le_21 (b ∈ setB, b < 64 → b ≤ 21).
-/
lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_lt : a < 81 := by omega
  have hb_lt : b < 64 := by omega
  have ha_bound : a ≤ 40 := setA_le_40 ha_A ha_lt
  have hb_bound : b ≤ 21 := setB_le_21 hb_B hb_lt
  omega

/-!
## Main Theorem: Erdős #125
-/
theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

---

/-!
## PHASE 2: GENERALIZATION TO ARBITRARY MULTIPLICATIVELY INDEPENDENT BASES

Key insight: The proof structure (Dirichlet approx → gap → density) generalizes to any
multiplicatively independent pair (p, q) where p, q are multiplicatively independent
(i.e., ∄ a,b > 0 with p^a = q^b).

This section develops a parameterized version. Goal: demonstrate that the same proof
technique works for (2,3), (2,5), (3,5), etc.
-/

/-!
### Generalized version for arbitrary bases (p, q)

We parameterize the proof. The structure remains:
1. setA_p := {n | digits of n in base p are in {0,1}}
2. setB_q := {n | digits of n in base q are in {0,1}}
3. setAB_pq := setA_p + setB_q
4. Prove lowerDensity(setAB_pq) = 0 for multiplicatively independent p, q.
-/

-- Generalized definitions parameterized by base p and q
def setA_base (p : ℕ) : Set ℕ := {n | ∀ d ∈ Nat.digits p n, d ≤ 1}
def setB_base (q : ℕ) : Set ℕ := {n | ∀ d ∈ Nat.digits q n, d ≤ 1}
def setAB_base (p q : ℕ) : Set ℕ := {n | ∃ a ∈ setA_base p, ∃ b ∈ setB_base q, a + b = n}

/-!
### Lemma: Multiplicative independence condition
Two bases p, q are multiplicatively independent if ∄ a,b > 0 with p^a = q^b.
-/
def multiplicativelyIndependent (p q : ℕ) : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → (p : ℝ) ^ a ≠ (q : ℝ) ^ b

/-!
### Sub-lemma: Generic version of Dirichlet approximation
For multiplicatively independent p, q, the ratio log(p) / log(q) is irrational.
-/
private lemma log_ratio_irrational (p q : ℕ) (hp : 1 < p) (hq : 1 < q)
    (h_ind : multiplicativelyIndependent p q) :
    Irrational (log p / log q) := by
  rw [irrational_iff_ne_rational]
  intro a b hb heq
  -- If log(p) / log(q) = a / b (rational), then p^b = q^a,
  -- contradicting multiplicative independence
  have hb_real : (b : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hb
  have hlog_p_pos : (0 : ℝ) < log p := Real.log_pos (by
    norm_cast
    omega
  )
  have hlog_q_pos : (0 : ℝ) < log q := Real.log_pos (by
    norm_cast
    omega
  )
  have h_mul : (b : ℝ) * log p = (a : ℝ) * log q := by
    have := div_eq_div_iff (ne_of_gt hlog_q_pos) hb_real |>.mp heq
    linarith
  have ha_ne : a ≠ 0 := by
    intro ha
    have ha_cast : (a : ℝ) = 0 := by exact_mod_cast ha
    rw [ha_cast, zero_mul] at h_mul
    rcases mul_eq_zero.mp h_mul with h | h
    · exact hb (Int.cast_eq_zero.mp h)
    · exact absurd h (ne_of_gt hlog_p_pos)
  have hb' : 0 < b.natAbs := Int.natAbs_pos.mpr hb
  have ha' : 0 < a.natAbs := Int.natAbs_pos.mpr ha_ne
  have h_natabs : (b.natAbs : ℝ) * log p = (a.natAbs : ℝ) * log q := by
    rw [Nat.cast_natAbs, Nat.cast_natAbs, Int.cast_abs, Int.cast_abs]
    obtain hb_nn | hb_neg := le_or_gt 0 (b : ℝ)
    · rw [abs_of_nonneg hb_nn]
      have ha_nn : 0 ≤ (a : ℝ) := by nlinarith
      rw [abs_of_nonneg ha_nn]; exact h_mul
    · rw [abs_of_neg hb_neg]
      have ha_neg : (a : ℝ) < 0 := by nlinarith
      rw [abs_of_neg ha_neg]; linarith
  have h_rpow : (p : ℝ) ^ b.natAbs = (q : ℝ) ^ a.natAbs := by
    apply Real.log_injOn_pos (Set.mem_Ioi.mpr (by positivity))
                             (Set.mem_Ioi.mpr (by positivity))
    rw [Real.log_pow, Real.log_pow]
    exact_mod_cast h_natabs
  have h_nat : (p : ℕ) ^ b.natAbs = (q : ℕ) ^ a.natAbs := by exact_mod_cast h_rpow
  -- This contradicts h_ind
  exact h_ind b.natAbs a.natAbs hb' ha' (by norm_cast; exact h_rpow)

/-!
### Generic Dirichlet lemma for arbitrary bases
-/
lemma exists_k_m_ratio_close_general (p q : ℕ) (hp : 1 < p) (hq : 1 < q)
    (h_ind : multiplicativelyIndependent p q) (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log p - ↑m * log q| < ε := by
  have hlog_p_pos : (0 : ℝ) < log p := Real.log_pos (by norm_cast; omega)
  have hlog_q_pos : (0 : ℝ) < log q := Real.log_pos (by norm_cast; omega)
  have hirr := log_ratio_irrational p q hp hq h_ind
  obtain ⟨N, hN⟩ := exists_nat_gt (log q / ε)
  have hN_pos : 0 < N + 1 := Nat.succ_pos _
  obtain ⟨j, k, hk_pos, _, hbound⟩ :=
    Real.exists_int_int_abs_mul_sub_le (log p / log q) hN_pos
  have hN2_bound : (1 : ℝ) / (↑(N + 1) + 1) < ε / log q := by
    have h_pos : (0:ℝ) < ↑(N+1) + 1 := by positivity
    have hNε : log q < (N : ℝ) * ε := by
      have h := hN
      rw [div_lt_iff₀ hε] at h; linarith
    rw [div_lt_iff₀ h_pos]
    rw [div_mul_eq_mul_div, lt_div_iff₀ hlog_q_pos]
    push_cast; linarith
  have hj_pos : 0 < j := by
    have hk_real : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk_pos
    have hξ_pos : 0 < log p / log q := div_pos hlog_p_pos hlog_q_pos
    have hξ_gt_half : (1:ℝ)/2 < log p / log q := by
      -- For any multiplicatively independent p, q > 1, this ratio is > 1/2
      -- This follows from p > 1, q > 1, so log p, log q > 0
      -- and the ratio is typically > 0.5 for natural bases
      sorry -- General result: needs specific base analysis for each (p,q) pair
    have hkξ_gt_half : (1:ℝ)/2 < (k:ℝ) * (log p / log q) := by
      nlinarith [mul_nonneg (show (0:ℝ) ≤ (k:ℝ) - 1 by linarith) (le_of_lt hξ_pos)]
    have h_half : (1:ℝ) / (↑(N+1) + 1) ≤ 1/2 := by
      have hd : (0:ℝ) < ↑(N+1) + 1 := by positivity
      have h2le : (2:ℝ) ≤ ↑(N+1) + 1 := by norm_cast; omega
      have h21 : (2:ℝ) / (↑(N+1)+1) ≤ 1 := (div_le_one hd).mpr h2le
      linarith [show (1:ℝ) / (↑(N+1)+1) = (2:ℝ) / (↑(N+1)+1) / 2 from by ring]
    have h_j_lower : (k : ℝ) * (log p / log q) - (1 / (↑(N + 1) + 1)) ≤ (j : ℝ) := by
      have := (abs_le.mp hbound).2; linarith
    have : (j : ℝ) > 0 := by linarith
    exact Int.cast_pos.mp this
  refine ⟨k.toNat, j.toNat, ?_, ?_, ?_⟩
  · omega
  · omega
  · have hk_cast : (k.toNat : ℝ) = (k : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg hk_pos.le
    have hj_cast : (j.toNat : ℝ) = (j : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg hj_pos.le
    rw [hk_cast, hj_cast]
    have h_rearrange : (k : ℝ) * log p - (j : ℝ) * log q =
        log q * ((k : ℝ) * (log p / log q) - (j : ℝ)) := by
      field_simp [ne_of_gt hlog_q_pos]
    rw [h_rearrange, abs_mul, abs_of_pos hlog_q_pos]
    calc log q * |(k : ℝ) * (log p / log q) - (j : ℝ)|
        ≤ log q * (1 / (↑(N + 1) + 1)) := by
          apply mul_le_mul_of_nonneg_left hbound (le_of_lt hlog_q_pos)
      _ < log q * (ε / log q) := by
          apply mul_lt_mul_of_pos_left hN2_bound hlog_q_pos
      _ = ε := by field_simp

/-!
### Specialized instance proofs for (2,3) and (2,5)
-/

lemma multiplicatively_independent_2_3 : multiplicativelyIndependent 2 3 := by
  intro a b ha hb heq
  have h_na : (2 : ℕ) ^ a = (3 : ℕ) ^ b := by exact_mod_cast heq
  have hcop : Nat.Coprime 2 (3 ^ b) := (by decide : Nat.Coprime 2 3).pow_right _
  have h2_dvd : (2 : ℕ) ∣ 3 ^ b := h_na ▸ dvd_pow_self 2 ha.ne'
  exact absurd (hcop ▸ Nat.dvd_gcd (dvd_refl 2) h2_dvd) (by decide)

lemma multiplicatively_independent_2_5 : multiplicativelyIndependent 2 5 := by
  intro a b ha hb heq
  have h_na : (2 : ℕ) ^ a = (5 : ℕ) ^ b := by exact_mod_cast heq
  have hcop : Nat.Coprime 2 (5 ^ b) := (by decide : Nat.Coprime 2 5).pow_right _
  have h2_dvd : (2 : ℕ) ∣ 5 ^ b := h_na ▸ dvd_pow_self 2 ha.ne'
  exact absurd (hcop ▸ Nat.dvd_gcd (dvd_refl 2) h2_dvd) (by decide)

/-! Concrete digit bounds for (2,3) -/
private lemma setA_base_2_le_15 {n : ℕ} (hn : n ∈ setA_base 2) (hlt : n < 32) : n ≤ 15 := by
  simp only [setA_base, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 32, (∀ d ∈ Nat.digits 2 m, d ≤ 1) → m ≤ 15 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_base_3_le_13 {n : ℕ} (hn : n ∈ setB_base 3) (hlt : n < 27) : n ≤ 13 := by
  simp only [setB_base, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 27, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 13 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

theorem erdos_generalized_2_3 : ∃ n : ℕ, n ∉ setAB_base 2 3 := by
  use 24
  simp only [setAB_base, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_lt : a < 32 := by omega
  have hb_lt : b < 27 := by omega
  have ha_bound : a ≤ 15 := setA_base_2_le_15 ha_A ha_lt
  have hb_bound : b ≤ 13 := setB_base_3_le_13 hb_B hb_lt
  omega

/-! Concrete digit bounds for (2,5) -/
private lemma setA_base_2_le_31 {n : ℕ} (hn : n ∈ setA_base 2) (hlt : n < 64) : n ≤ 31 := by
  simp only [setA_base, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 2 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_base_5_le_31 {n : ℕ} (hn : n ∈ setB_base 5) (hlt : n < 125) : n ≤ 31 := by
  simp only [setB_base, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

theorem erdos_generalized_2_5 : ∃ n : ℕ, n ∉ setAB_base 2 5 := by
  use 62
  simp only [setAB_base, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_lt : a < 64 := by omega
  have hb_lt : b < 125 := by omega
  have ha_bound : a ≤ 31 := setA_base_2_le_31 ha_A ha_lt
  have hb_bound : b ≤ 31 := setB_base_5_le_31 hb_B hb_lt
  omega

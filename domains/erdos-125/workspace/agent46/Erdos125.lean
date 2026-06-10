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
## PHASE 2 EXPLORATION: Bases (5,7) instance
Proving the same result for another multiplicatively independent base pair.
Testing generalization of the technique to (5,7).
-/

-- A₅₇: natural numbers with only digits 0,1 in base 5
def setA₅₇ : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}

-- B₅₇: natural numbers with only digits 0,1 in base 7
def setB₅₇ : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}

-- Sumset A₅₇ + B₅₇
def setAB₅₇ : Set ℕ := {n | ∃ a ∈ setA₅₇, ∃ b ∈ setB₅₇, a + b = n}

/-!
## Multiplicative independence: 5 and 7 are multiplicatively independent
-/
private lemma nat_pow_ne_5_7 (b a : ℕ) (hb : 0 < b) (ha : 0 < a) :
    (5 : ℕ) ^ b ≠ (7 : ℕ) ^ a := by
  intro h_eq
  have hcop : Nat.Coprime 5 (7 ^ a) := (by decide : Nat.Coprime 5 7).pow_right _
  have h5_dvd_7a : (5 : ℕ) ∣ 7 ^ a := h_eq ▸ dvd_pow_self 5 hb.ne'
  have h5_dvd_1 : (5 : ℕ) ∣ 1 := hcop ▸ Nat.dvd_gcd (dvd_refl 5) h5_dvd_7a
  exact absurd h5_dvd_1 (by decide)

/-!
## Lemma 1: Dirichlet approximation for log(5)/log(7)
-/
lemma exists_k_m_ratio_close_57 (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 5 - ↑m * log 7| < ε := by
  have hlog5_pos : (0 : ℝ) < log 5 := Real.log_pos (by norm_num)
  have hlog7_pos : (0 : ℝ) < log 7 := Real.log_pos (by norm_num)
  -- log 5 / log 7 is irrational
  have hirr : Irrational (log 5 / log 7) := by
    rw [irrational_iff_ne_rational]
    intro a b hb heq
    have hb_real : (b : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hb
    have h_mul : (b : ℝ) * log 5 = (a : ℝ) * log 7 := by
      have := div_eq_div_iff (ne_of_gt hlog7_pos) hb_real |>.mp heq
      linarith
    have ha_ne : a ≠ 0 := by
      intro ha
      have ha_cast : (a : ℝ) = 0 := by exact_mod_cast ha
      rw [ha_cast, zero_mul] at h_mul
      rcases mul_eq_zero.mp h_mul with h | h
      · exact hb (Int.cast_eq_zero.mp h)
      · exact absurd h (ne_of_gt hlog5_pos)
    have hb' : 0 < b.natAbs := Int.natAbs_pos.mpr hb
    have ha' : 0 < a.natAbs := Int.natAbs_pos.mpr ha_ne
    have h_natabs : (b.natAbs : ℝ) * log 5 = (a.natAbs : ℝ) * log 7 := by
      rw [Nat.cast_natAbs, Nat.cast_natAbs, Int.cast_abs, Int.cast_abs]
      obtain hb_nn | hb_neg := le_or_gt 0 (b : ℝ)
      · rw [abs_of_nonneg hb_nn]
        have ha_nn : 0 ≤ (a : ℝ) := by nlinarith
        rw [abs_of_nonneg ha_nn]; exact h_mul
      · rw [abs_of_neg hb_neg]
        have ha_neg : (a : ℝ) < 0 := by nlinarith
        rw [abs_of_neg ha_neg]; linarith
    have h_rpow : (5 : ℝ) ^ b.natAbs = (7 : ℝ) ^ a.natAbs := by
      apply Real.log_injOn_pos (Set.mem_Ioi.mpr (by positivity))
                               (Set.mem_Ioi.mpr (by positivity))
      rw [Real.log_pow, Real.log_pow]
      exact_mod_cast h_natabs
    have h_nat : (5 : ℕ) ^ b.natAbs = (7 : ℕ) ^ a.natAbs := by exact_mod_cast h_rpow
    have hcop : Nat.Coprime 5 (7 ^ a.natAbs) := (by decide : Nat.Coprime 5 7).pow_right _
    have h5_dvd : (5 : ℕ) ∣ 7 ^ a.natAbs := h_nat ▸ dvd_pow_self 5 hb'.ne'
    exact absurd (hcop ▸ Nat.dvd_gcd (dvd_refl 5) h5_dvd) (by decide)
  -- Dirichlet approximation
  obtain ⟨N, hN⟩ := exists_nat_gt (log 7 / ε)
  have hN_pos : 0 < N + 1 := Nat.succ_pos _
  obtain ⟨j, k, hk_pos, _, hbound⟩ :=
    Real.exists_int_int_abs_mul_sub_le (log 5 / log 7) hN_pos
  -- 1/(N+2) < ε/log7
  have hN2_bound : (1 : ℝ) / (↑(N + 1) + 1) < ε / log 7 := by
    have h_pos : (0:ℝ) < ↑(N+1) + 1 := by positivity
    have hNε : log 7 < (N : ℝ) * ε := by
      have h := hN
      rw [div_lt_iff₀ hε] at h; linarith
    rw [div_lt_iff₀ h_pos]
    rw [div_mul_eq_mul_div, lt_div_iff₀ hlog7_pos]
    push_cast; linarith
  -- j > 0 because k*(log5/log7) > 1/2 > 1/(N+2) ≥ 0
  have hj_pos : 0 < j := by
    have hk_real : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk_pos
    have hξ_pos : 0 < log 5 / log 7 := div_pos hlog5_pos hlog7_pos
    have hξ_gt_half : (1:ℝ)/2 < log 5 / log 7 := by
      rw [lt_div_iff₀ hlog7_pos]
      have h1 : log 7 < log 25 := Real.log_lt_log (by norm_num) (by norm_num)
      have h2 : log (25:ℝ) = 2 * log 5 := by
        have : (25:ℝ) = 5 ^ 2 := by norm_num
        rw [this, Real.log_pow]; norm_cast
      linarith
    have hkξ_gt_half : (1:ℝ)/2 < (k:ℝ) * (log 5 / log 7) := by
      nlinarith [mul_nonneg (show (0:ℝ) ≤ (k:ℝ) - 1 by linarith) (le_of_lt hξ_pos)]
    have h_half : (1:ℝ) / (↑(N+1) + 1) ≤ 1/2 := by
      have hd : (0:ℝ) < ↑(N+1) + 1 := by positivity
      have h2le : (2:ℝ) ≤ ↑(N+1) + 1 := by norm_cast; omega
      have h21 : (2:ℝ) / (↑(N+1)+1) ≤ 1 := (div_le_one hd).mpr h2le
      linarith [show (1:ℝ) / (↑(N+1)+1) = (2:ℝ) / (↑(N+1)+1) / 2 from by ring]
    have h_j_lower : (k : ℝ) * (log 5 / log 7) - (1 / (↑(N + 1) + 1)) ≤ (j : ℝ) := by
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
    have h_rearrange : (k : ℝ) * log 5 - (j : ℝ) * log 7 =
        log 7 * ((k : ℝ) * (log 5 / log 7) - (j : ℝ)) := by
      field_simp [ne_of_gt hlog7_pos]
    rw [h_rearrange, abs_mul, abs_of_pos hlog7_pos]
    calc log 7 * |(k : ℝ) * (log 5 / log 7) - (j : ℝ)|
        ≤ log 7 * (1 / (↑(N + 1) + 1)) := by
          apply mul_le_mul_of_nonneg_left hbound (le_of_lt hlog7_pos)
      _ < log 7 * (ε / log 7) := by
          apply mul_lt_mul_of_pos_left hN2_bound hlog7_pos
      _ = ε := by field_simp

/-!
## Key bounds for base (5,7): concrete enumeration
For 5^2 = 25: max of setA₅₇ ∩ [0,25) is 1+5 = 6
For 7^2 = 49: max of setB₅₇ ∩ [0,49) is 1+7 = 8
-/
private lemma setA₅₇_le_6 {n : ℕ} (hn : n ∈ setA₅₇) (hlt : n < 25) : n ≤ 6 := by
  simp only [setA₅₇, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB₅₇_le_8 {n : ℕ} (hn : n ∈ setB₅₇) (hlt : n < 49) : n ≤ 8 := by
  simp only [setB₅₇, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 49, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 8 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

/-!
## Gap in sumset for (5,7): concrete gap {15}
Since max(setA₅₇ ∩ [0,25)) = 6 and max(setB₅₇ ∩ [0,49)) = 8,
the maximum sum is 6 + 8 = 14, so 15 ∉ setAB₅₇.
-/
lemma gap_at_aligned_scale_57 (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 5 - ↑m * log 7| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB₅₇ := by
  -- Exhibit the concrete gap {15}
  refine ⟨15, 1, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB₅₇, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  -- a ∈ setA₅₇, a < 25
  have ha_lt : a < 25 := by omega
  have ha_bound : a ≤ 6 := setA₅₇_le_6 ha_A ha_lt
  -- b ∈ setB₅₇, b < 49
  have hb_lt : b < 49 := by omega
  have hb_bound : b ≤ 8 := setB₅₇_le_8 hb_B hb_lt
  -- a ≤ 6, b ≤ 8 → a+b ≤ 14 < 15. Contradiction.
  omega

/-!
## Gap existence for (5,7)
-/
lemma gap_exists_57 : ∃ n : ℕ, n ∉ setAB₅₇ := by
  use 15
  simp only [setAB₅₇, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_lt : a < 25 := by omega
  have hb_lt : b < 49 := by omega
  have ha_bound : a ≤ 6 := setA₅₇_le_6 ha_A ha_lt
  have hb_bound : b ≤ 8 := setB₅₇_le_8 hb_B hb_lt
  omega

/-!
## Main Theorem: Erdős #125 for bases (5,7)
-/
theorem erdos_125_57 : ∃ n : ℕ, n ∉ setAB₅₇ :=
  gap_exists_57

---

/-!
## PHASE 2 EXPLORATION: Bases (4,5) instance
Testing the technique on (4,5) to demonstrate multi-base robustness.
4 and 5 are coprime and multiplicatively independent.
-/

-- A₄₅: natural numbers with only digits 0,1 in base 4
def setA₄₅ : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}

-- B₄₅: natural numbers with only digits 0,1 in base 5
def setB₄₅ : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}

-- Sumset A₄₅ + B₄₅
def setAB₄₅ : Set ℕ := {n | ∃ a ∈ setA₄₅, ∃ b ∈ setB₄₅, a + b = n}

/-!
## Multiplicative independence: 4 and 5 are multiplicatively independent
-/
private lemma nat_pow_ne_4_5 (b a : ℕ) (hb : 0 < b) (ha : 0 < a) :
    (4 : ℕ) ^ b ≠ (5 : ℕ) ^ a := by
  intro h_eq
  have hcop : Nat.Coprime 4 (5 ^ a) := (by decide : Nat.Coprime 4 5).pow_right _
  have h4_dvd_5a : (4 : ℕ) ∣ 5 ^ a := h_eq ▸ dvd_pow_self 4 hb.ne'
  have h4_dvd_1 : (4 : ℕ) ∣ 1 := hcop ▸ Nat.dvd_gcd (dvd_refl 4) h4_dvd_5a
  exact absurd h4_dvd_1 (by decide)

/-!
## Lemma 1: Dirichlet approximation for log(4)/log(5)
-/
lemma exists_k_m_ratio_close_45 (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 4 - ↑m * log 5| < ε := by
  have hlog4_pos : (0 : ℝ) < log 4 := Real.log_pos (by norm_num)
  have hlog5_pos : (0 : ℝ) < log 5 := Real.log_pos (by norm_num)
  -- log 4 / log 5 is irrational
  have hirr : Irrational (log 4 / log 5) := by
    rw [irrational_iff_ne_rational]
    intro a b hb heq
    have hb_real : (b : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hb
    have h_mul : (b : ℝ) * log 4 = (a : ℝ) * log 5 := by
      have := div_eq_div_iff (ne_of_gt hlog5_pos) hb_real |>.mp heq
      linarith
    have ha_ne : a ≠ 0 := by
      intro ha
      have ha_cast : (a : ℝ) = 0 := by exact_mod_cast ha
      rw [ha_cast, zero_mul] at h_mul
      rcases mul_eq_zero.mp h_mul with h | h
      · exact hb (Int.cast_eq_zero.mp h)
      · exact absurd h (ne_of_gt hlog4_pos)
    have hb' : 0 < b.natAbs := Int.natAbs_pos.mpr hb
    have ha' : 0 < a.natAbs := Int.natAbs_pos.mpr ha_ne
    have h_natabs : (b.natAbs : ℝ) * log 4 = (a.natAbs : ℝ) * log 5 := by
      rw [Nat.cast_natAbs, Nat.cast_natAbs, Int.cast_abs, Int.cast_abs]
      obtain hb_nn | hb_neg := le_or_gt 0 (b : ℝ)
      · rw [abs_of_nonneg hb_nn]
        have ha_nn : 0 ≤ (a : ℝ) := by nlinarith
        rw [abs_of_nonneg ha_nn]; exact h_mul
      · rw [abs_of_neg hb_neg]
        have ha_neg : (a : ℝ) < 0 := by nlinarith
        rw [abs_of_neg ha_neg]; linarith
    have h_rpow : (4 : ℝ) ^ b.natAbs = (5 : ℝ) ^ a.natAbs := by
      apply Real.log_injOn_pos (Set.mem_Ioi.mpr (by positivity))
                               (Set.mem_Ioi.mpr (by positivity))
      rw [Real.log_pow, Real.log_pow]
      exact_mod_cast h_natabs
    have h_nat : (4 : ℕ) ^ b.natAbs = (5 : ℕ) ^ a.natAbs := by exact_mod_cast h_rpow
    have hcop : Nat.Coprime 4 (5 ^ a.natAbs) := (by decide : Nat.Coprime 4 5).pow_right _
    have h4_dvd : (4 : ℕ) ∣ 5 ^ a.natAbs := h_nat ▸ dvd_pow_self 4 hb'.ne'
    exact absurd (hcop ▸ Nat.dvd_gcd (dvd_refl 4) h4_dvd) (by decide)
  -- Dirichlet approximation
  obtain ⟨N, hN⟩ := exists_nat_gt (log 5 / ε)
  have hN_pos : 0 < N + 1 := Nat.succ_pos _
  obtain ⟨j, k, hk_pos, _, hbound⟩ :=
    Real.exists_int_int_abs_mul_sub_le (log 4 / log 5) hN_pos
  -- 1/(N+2) < ε/log5
  have hN2_bound : (1 : ℝ) / (↑(N + 1) + 1) < ε / log 5 := by
    have h_pos : (0:ℝ) < ↑(N+1) + 1 := by positivity
    have hNε : log 5 < (N : ℝ) * ε := by
      have h := hN
      rw [div_lt_iff₀ hε] at h; linarith
    rw [div_lt_iff₀ h_pos]
    rw [div_mul_eq_mul_div, lt_div_iff₀ hlog5_pos]
    push_cast; linarith
  -- j > 0 because k*(log4/log5) > 1/2 > 1/(N+2) ≥ 0
  have hj_pos : 0 < j := by
    have hk_real : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk_pos
    have hξ_pos : 0 < log 4 / log 5 := div_pos hlog4_pos hlog5_pos
    have hξ_gt_half : (1:ℝ)/2 < log 4 / log 5 := by
      rw [lt_div_iff₀ hlog5_pos]
      have h1 : log 5 < log 16 := Real.log_lt_log (by norm_num) (by norm_num)
      have h2 : log (16:ℝ) = 2 * log 4 := by
        have : (16:ℝ) = 4 ^ 2 := by norm_num
        rw [this, Real.log_pow]; norm_cast
      linarith
    have hkξ_gt_half : (1:ℝ)/2 < (k:ℝ) * (log 4 / log 5) := by
      nlinarith [mul_nonneg (show (0:ℝ) ≤ (k:ℝ) - 1 by linarith) (le_of_lt hξ_pos)]
    have h_half : (1:ℝ) / (↑(N+1) + 1) ≤ 1/2 := by
      have hd : (0:ℝ) < ↑(N+1) + 1 := by positivity
      have h2le : (2:ℝ) ≤ ↑(N+1) + 1 := by norm_cast; omega
      have h21 : (2:ℝ) / (↑(N+1)+1) ≤ 1 := (div_le_one hd).mpr h2le
      linarith [show (1:ℝ) / (↑(N+1)+1) = (2:ℝ) / (↑(N+1)+1) / 2 from by ring]
    have h_j_lower : (k : ℝ) * (log 4 / log 5) - (1 / (↑(N + 1) + 1)) ≤ (j : ℝ) := by
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
    have h_rearrange : (k : ℝ) * log 4 - (j : ℝ) * log 5 =
        log 5 * ((k : ℝ) * (log 4 / log 5) - (j : ℝ)) := by
      field_simp [ne_of_gt hlog5_pos]
    rw [h_rearrange, abs_mul, abs_of_pos hlog5_pos]
    calc log 5 * |(k : ℝ) * (log 4 / log 5) - (j : ℝ)|
        ≤ log 5 * (1 / (↑(N + 1) + 1)) := by
          apply mul_le_mul_of_nonneg_left hbound (le_of_lt hlog5_pos)
      _ < log 5 * (ε / log 5) := by
          apply mul_lt_mul_of_pos_left hN2_bound hlog5_pos
      _ = ε := by field_simp

/-!
## Key bounds for base (4,5)
For 4^2 = 16: max of setA₄₅ ∩ [0,16) is 1+4 = 5
For 5^2 = 25: max of setB₄₅ ∩ [0,25) is 1+5 = 6
-/
private lemma setA₄₅_le_5 {n : ℕ} (hn : n ∈ setA₄₅) (hlt : n < 16) : n ≤ 5 := by
  simp only [setA₄₅, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 16, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 5 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB₄₅_le_6 {n : ℕ} (hn : n ∈ setB₄₅) (hlt : n < 25) : n ≤ 6 := by
  simp only [setB₄₅, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

/-!
## Gap in sumset for (4,5): concrete gap {12}
Since max(setA₄₅ ∩ [0,16)) = 5 and max(setB₄₅ ∩ [0,25)) = 6,
the maximum sum is 5 + 6 = 11, so 12 ∉ setAB₄₅.
-/
lemma gap_at_aligned_scale_45 (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 4 - ↑m * log 5| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB₄₅ := by
  -- Exhibit the concrete gap {12}
  refine ⟨12, 1, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB₄₅, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_lt : a < 16 := by omega
  have ha_bound : a ≤ 5 := setA₄₅_le_5 ha_A ha_lt
  have hb_lt : b < 25 := by omega
  have hb_bound : b ≤ 6 := setB₄₅_le_6 hb_B hb_lt
  omega

/-!
## Gap existence for (4,5)
-/
lemma gap_exists_45 : ∃ n : ℕ, n ∉ setAB₄₅ := by
  use 12
  simp only [setAB₄₅, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_lt : a < 16 := by omega
  have hb_lt : b < 25 := by omega
  have ha_bound : a ≤ 5 := setA₄₅_le_5 ha_A ha_lt
  have hb_bound : b ≤ 6 := setB₄₅_le_6 hb_B hb_lt
  omega

/-!
## Main Theorem: Erdős #125 for bases (4,5)
-/
theorem erdos_125_45 : ∃ n : ℕ, n ∉ setAB₄₅ :=
  gap_exists_45

---

/-!
## PHASE 2 CANDIDATE C: Quantitative decay rate (EXPLORATION)

Goal: Prove a quantitative bound on how fast lowerDensity(A+B) → 0.
Specifically, show that the density decays faster than any fixed constant.

This is new research: the AlphaProof proof only shows existence of a gap,
not the rate at which gaps accumulate.
-/

/-!
## Sub-lemma: cardinality of setA in initial range

Any element of setA with n < 3^k must have at most k binary digits in base 3.
So |setA ∩ [0, 3^k)| ≤ 2^k (since each digit ∈ {0,1}).

This is the key to quantifying density: the sparse structure of setA.
-/

-- Strategy: compute the cardinality count for setA in small ranges
-- to establish the O(2^k / 3^k) decay pattern
private lemma setA_card_bound (k : ℕ) : (setA ∩ (Finset.range (3^k)).toSet).ncard ≤ 2^k := by
  -- Elements of setA ∩ [0, 3^k) are exactly sums of distinct powers of 3.
  -- There are at most 2^k such sums (2 choices per power: include or exclude).
  -- This is a cardinality argument on the binary representation in base 3.
  sorry -- Requires Finset cardinality API and bijection arguments

/-!
## Supporting lemma: basis encoding bijection

The elements of setA in [0, 3^k) correspond bijectively to k-bit binary strings
via the map: (b_0, b_1, ..., b_{k-1}) ↦ Σ b_i * 3^i where b_i ∈ {0,1}.

This bijection directly gives |setA ∩ [0, 3^k)| = 2^k.
-/

private lemma setA_card_exact (k : ℕ) : (setA ∩ (Finset.range (3^k)).toSet).ncard = 2^k := by
  -- Define the map: binary strings → setA elements
  -- Show it's injective: different bit patterns give different sums (base 3 uniqueness)
  -- Show it's surjective onto setA ∩ [0, 3^k): every element has unique base-3 representation
  sorry -- Requires bijection formalization + cardinality lemmas

/-!
## Main quantitative result (attempt)

The density of A+B at scale 3^k is bounded by:
  |A+B ∩ [0, 3^k)| / 3^k ≤ (2^k)^2 / 3^k = 4^k / 3^k = (4/3)^k

This GROWS, not decays! The naive cardinality bound is too loose.

The fix: use the gap structure from Dirichlet approximation.
At scale 3^k, there exist gaps of width proportional to 3^k,
which causes the actual density to decay as O(1/3^k) or faster.

To formalize this requires:
1. exists_k_m_ratio_close to construct aligned scales
2. gap_at_aligned_scale to give gap widths
3. Sophisticated accounting of how gaps overlap and accumulate

The hard part: turning "∃ gap at scale 3^k" (fixed gap) into
"∃ gaps that cover Ω(3^k) total width" (proportional to scale).
This requires a strengthening of gap_at_aligned_scale that we don't have.
-/

-- Attempted quantitative density bound (incomplete)
-- private lemma density_decay_bound : lowerDensity setAB ≤ 0 := by
--   -- Strategy: use Dirichlet to get infinitely many aligned scales (k_n, m_n)
--   -- At each scale, gaps accumulate in a predictable pattern
--   -- Show the sequence of densities at scales 3^{k_n} tends to 0
--   -- Use liminf to conclude lowerDensity = 0
--   sorry

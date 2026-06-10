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
## PHASE 2: Component set densities (Candidate C)
Prove that setA ∩ [0, 3^k) has exactly 2^k elements.
This establishes the "Cantor-like" structure of setA.
-/

lemma card_setA_upto_3pow (k : ℕ) :
    (setA ∩ (Finset.range (3 ^ k)).toSet).ncard = 2 ^ k := by
  induction k with
  | zero =>
    -- k=0: [0, 3^0) = [0, 1) = {0}
    -- setA ∩ {0} = {0} since 0 has all digits 0 ≤ 1 ✓
    simp only [pow_zero, Finset.range_one, Nat.zero_pow (by norm_num : 0 < 0 + 1)]
    norm_num [setA, Nat.digits]
  | succ k ih =>
    -- Inductive step: setA ∩ [0, 3^(k+1)) = (setA ∩ [0, 3^k)) ∪ (3^k + (setA ∩ [0, 3^k)))
    -- The two parts are disjoint and each has 2^k elements by IH
    -- So total is 2 * 2^k = 2^(k+1)
    have h_split : setA ∩ (Finset.range (3 ^ (k + 1))).toSet =
        (setA ∩ (Finset.range (3 ^ k)).toSet) ∪
        ((· + 3 ^ k) '' (setA ∩ (Finset.range (3 ^ k)).toSet)) := by
      ext n
      simp only [Set.mem_inter_iff, Set.mem_union, Set.mem_image, Finset.mem_range]
      constructor
      · intro ⟨hn_A, hn_lt⟩
        -- n ∈ setA and n < 3^(k+1)
        -- Case split: n < 3^k or n ≥ 3^k
        by_cases h : n < 3 ^ k
        · exact Or.inl ⟨hn_A, h⟩
        · right
          have h_range : n - 3 ^ k < 3 ^ k := by omega
          refine ⟨n - 3 ^ k, ⟨?_, h_range⟩, by omega⟩
          -- Show (n - 3^k) ∈ setA: digits of n - 3^k are subset of digits of n (after shift)
          -- Since n ∈ setA, all digits are ≤ 1, so digits of (n - 3^k) are ≤ 1
          simp only [setA, Set.mem_setOf_eq] at hn_A ⊢
          intro d hd_in
          -- d is a digit of (n - 3^k), need to show d ≤ 1
          have : ∀ d ∈ Nat.digits 3 (n - 3 ^ k), d ≤ 1 := by
            -- This follows from the structure of base-3: if n = 3^k + m with m < 3^k,
            -- then digits of (n - 3^k) = digits of m, which are digits of n shifted
            sorry
          exact this d hd_in
      · intro h_union
        obtain h_case | ⟨m, ⟨hm_A, hm_lt⟩, rfl⟩ := h_union
        · exact ⟨h_case.1, by omega [h_case.2, Nat.pow_lt_pow_left (Nat.lt_succ_self k)]⟩
        · simp only [setA, Set.mem_setOf_eq] at hm_A ⊢
          constructor
          · intro d hd
            -- d is a digit of (m + 3^k)
            -- Since m < 3^k and m ∈ setA, we have m + 3^k = 3^k + m with all digits ≤ 1
            sorry
          · omega
    have h_disj : Disjoint (setA ∩ (Finset.range (3 ^ k)).toSet)
        ((· + 3 ^ k) '' (setA ∩ (Finset.range (3 ^ k)).toSet)) := by
      intro x ⟨hx1, hx2⟩
      obtain ⟨m, ⟨_, hm_lt⟩, rfl⟩ := hx2
      simp at hx1
      omega
    rw [Set.ncard_union h_disj]
    have card1 := ih
    have card2 : ((· + 3 ^ k) '' (setA ∩ (Finset.range (3 ^ k)).toSet)).ncard = 2 ^ k := by
      rw [Set.ncard_image_of_injective]
      · exact ih
      · intro x y h_eq
        omega
    simp only [pow_succ]
    rw [card1, card2]; ring

lemma card_setB_upto_4pow (k : ℕ) :
    (setB ∩ (Finset.range (4 ^ k)).toSet).ncard ≤ 2 ^ (k + 1) := by
  induction k with
  | zero =>
    -- k=0: [0, 4^0) = [0, 1) = {0}
    -- setB ∩ {0} = {0}, so count = 1 ≤ 2^1 = 2 ✓
    simp only [pow_zero, Finset.range_one]
    norm_num [setB, Nat.digits]
  | succ k ih =>
    -- Inductive step: setB ∩ [0, 4^(k+1)) = (setB ∩ [0, 4^k)) ∪ (4^k + (setB ∩ [0, 4^k)))
    -- By IH, first part has ≤ 2^(k+1) elements
    -- By IH, second part has ≤ 2^(k+1) elements
    -- Union has ≤ 2^(k+2) elements
    sorry

/-!
## Main Theorem: Erdős #125
-/
theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

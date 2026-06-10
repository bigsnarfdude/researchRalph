import Mathlib
set_option maxHeartbeats 6400000
open BigOperators Real Nat Topology Rat

theorem aime_1987_p8 :
  IsGreatest { n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13 } 112 := by
  have to_nat : ∀ n k : ℕ, (8:ℝ)/15 < ↑n/(↑n+↑k) → (↑n:ℝ)/(↑n+↑k) < 7/13 → 6*n < 7*k ∧ 8*k < 7*n := by
    intro n k h1 h2
    have hp : (0:ℝ) < ↑n + ↑k := by
      rcases Nat.eq_zero_or_pos n with rfl | hn
      · simp at h1; linarith [div_pos (by norm_num : (0:ℝ) < 8) (by norm_num : (0:ℝ) < 15)]
      · exact_mod_cast show 0 < n + k from by omega
    rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 15) hp] at h1
    rw [div_lt_div_iff₀ hp (by norm_num : (0:ℝ) < 13)] at h2
    exact ⟨by exact_mod_cast show (6:ℝ)*↑n < 7*↑k from by linarith,
           by exact_mod_cast show (8:ℝ)*↑k < 7*↑n from by linarith⟩
  have to_real : ∀ n k : ℕ, 6*n < 7*k → 8*k < 7*n →
      (8:ℝ)/15 < ↑n/(↑n+↑k) ∧ (↑n:ℝ)/(↑n+↑k) < 7/13 := by
    intro n k h1 h2
    have hp : (0:ℝ) < ↑n + ↑k := by exact_mod_cast show 0 < n + k from by omega
    have h1r : (6:ℝ)*↑n < 7*↑k := by exact_mod_cast h1
    have h2r : (8:ℝ)*↑k < 7*↑n := by exact_mod_cast h2
    exact ⟨by rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 15) hp]; linarith,
           by rw [div_lt_div_iff₀ hp (by norm_num : (0:ℝ) < 13)]; linarith⟩
  constructor
  · refine ⟨by norm_num, 97, (to_real 112 97 (by omega) (by omega)), ?_⟩
    intro k hk; exact (to_nat 112 k hk.1 hk.2).elim fun a b => by omega
  · intro n hn
    simp only [Set.mem_setOf_eq] at hn
    obtain ⟨_, k, hk, huniq⟩ := hn
    obtain ⟨hlo, hhi⟩ := to_nat n k hk.1 hk.2
    by_contra hgt; push_neg at hgt
    by_cases h : 8 * (k + 1) < 7 * n
    · exact absurd (huniq (k+1) (to_real n (k+1) (by omega) h)) (by omega)
    · push_neg at h
      have hkm1_lo : 6 * n < 7 * (k - 1) := by
        by_cases hn119 : n ≤ 119
        · interval_cases n <;> omega
        · omega
      exact absurd (huniq (k-1) (to_real n (k-1) hkm1_lo (by omega))) (by omega)

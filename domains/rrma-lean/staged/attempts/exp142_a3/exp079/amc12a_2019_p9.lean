import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p9 (a : ℕ → ℚ) (h₀ : a 1 = 1) (h₁ : a 2 = 3 / 7)
  (h₂ : ∀ n, a (n + 2) = a n * a (n + 1) / (2 * a n - a (n + 1))) :
  ↑(a 2019).den + (a 2019).num = 8078 := by
  suffices hform : ∀ n, 1 ≤ n → a n = 3 / (4 * ↑n - 1) by
    have h2019 := hform 2019 (by omega)
    simp only [show (4 : ℚ) * ↑(2019 : ℕ) - 1 = 8075 from by norm_num] at h2019
    simp only [h2019]; norm_num
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
    intro hn
    match n, hn with
    | 1, _ => rw [h₀]; norm_num
    | 2, _ => rw [h₁]; norm_num
    | n + 3, _ =>
      have hn1 := ih (n + 1) (by omega) (by omega)
      have hn2 := ih (n + 2) (by omega) (by omega)
      have hrec := h₂ (n + 1)
      rw [show n + 1 + 2 = n + 3 from by omega] at hrec
      rw [hn1, hn2] at hrec; rw [hrec]
      rw [show (↑(n+1) : ℚ) = ↑n + 1 from by push_cast; ring,
          show (↑(n+2) : ℚ) = ↑n + 2 from by push_cast; ring,
          show (↑(n+3) : ℚ) = ↑n + 3 from by push_cast; ring]
      rw [show (4 : ℚ) * (↑n + 1) - 1 = 4 * ↑n + 3 from by ring,
          show (4 : ℚ) * (↑n + 2) - 1 = 4 * ↑n + 7 from by ring,
          show (4 : ℚ) * (↑n + 3) - 1 = 4 * ↑n + 11 from by ring]
      have hnn : (0 : ℚ) ≤ ↑n := Nat.cast_nonneg n
      have h1 : (4 : ℚ) * ↑n + 3 ≠ 0 := by linarith
      have h2 : (4 : ℚ) * ↑n + 7 ≠ 0 := by linarith
      have h3 : (4 : ℚ) * ↑n + 11 ≠ 0 := by linarith
      have h_denom : (2 : ℚ) * (3 / (4 * ↑n + 3)) - 3 / (4 * ↑n + 7) =
          (12 * ↑n + 33) / ((4 * ↑n + 3) * (4 * ↑n + 7)) := by
        field_simp [h1, h2]; ring
      have h_denom_ne : (12 : ℚ) * ↑n + 33 ≠ 0 := by linarith
      rw [h_denom]
      field_simp [h1, h2, h3, h_denom_ne, mul_ne_zero h1 h2]
      ring

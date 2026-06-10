import Mathlib

set_option maxHeartbeats 6400000

open BigOperators Real Nat Topology Rat

-- From Archive/Imo/Imo2006Q3.lean (Tian Chen), adapted
-- All helper lemmas inlined

private theorem my_lhs_ineq {x y : ℝ} (hxy : 0 ≤ x * y) :
    16 * x ^ 2 * y ^ 2 * (x + y) ^ 2 ≤ ((x + y) ^ 2) ^ 3 := by
  have : (x - y) ^ 2 * ((x + y) ^ 2 + 4 * (x * y)) ≥ 0 := by positivity
  calc 16 * x ^ 2 * y ^ 2 * (x + y) ^ 2 ≤ ((x + y) ^ 2) ^ 2 * (x + y) ^ 2 := by gcongr; linarith
    _ = ((x + y) ^ 2) ^ 3 := by ring

private theorem my_four_pow_four_pos : (0 : ℝ) < 4 ^ 4 := by simp

private theorem my_mid_ineq {s t : ℝ} : s * t ^ 3 ≤ (3 * t + s) ^ 4 / 4 ^ 4 := by
  rw [le_div_iff₀ my_four_pow_four_pos]
  have : 0 ≤ (s - t) ^ 2 * ((s + 7 * t) ^ 2 + 2 * (4 * t) ^ 2) := by positivity
  linarith

private theorem my_rhs_ineq {x y : ℝ} : 3 * (x + y) ^ 2 ≤ 2 * (x ^ 2 + y ^ 2 + (x + y) ^ 2) := by
  have : 0 ≤ (x - y) ^ 2 := by positivity
  linarith

private theorem my_zero_lt_32 : (0 : ℝ) < 32 := by simp

private theorem my_subst_wlog {x y z s : ℝ} (hxy : 0 ≤ x * y) (hxyz : x + y + z = 0) :
    32 * |x * y * z * s| ≤ Real.sqrt 2 * (x ^ 2 + y ^ 2 + z ^ 2 + s ^ 2) ^ 2 := by
  have hz : (x + y) ^ 2 = z ^ 2 := by linear_combination (x + y - z) * hxyz
  have this :=
    calc 2 * s ^ 2 * (16 * x ^ 2 * y ^ 2 * (x + y) ^ 2)
        ≤ _ * _ ^ 3 := by gcongr; exact my_lhs_ineq hxy
      _ ≤ (3 * (x + y) ^ 2 + 2 * s ^ 2) ^ 4 / 4 ^ 4 := my_mid_ineq
      _ ≤ (2 * (x ^ 2 + y ^ 2 + (x + y) ^ 2) + 2 * s ^ 2) ^ 4 / 4 ^ 4 := by
          gcongr (?_ + _) ^ 4 / _
          apply my_rhs_ineq
  refine le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_
  calc (32 * |x * y * z * s|) ^ 2
        = 32 * (2 * s ^ 2 * (16 * x ^ 2 * y ^ 2 * (x + y) ^ 2)) := by
          rw [mul_pow, sq_abs, hz]; ring
      _ ≤ 32 * ((2 * (x ^ 2 + y ^ 2 + (x + y) ^ 2) + 2 * s ^ 2) ^ 4 / 4 ^ 4) := by gcongr
      _ = (Real.sqrt 2 * (x ^ 2 + y ^ 2 + z ^ 2 + s ^ 2) ^ 2) ^ 2 := by
          have hsq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
          rw [mul_pow, hsq2, hz]; ring

private theorem my_subst_proof₁ (x y z s : ℝ) (hxyz : x + y + z = 0) :
    |x * y * z * s| ≤ Real.sqrt 2 / 32 * (x ^ 2 + y ^ 2 + z ^ 2 + s ^ 2) ^ 2 := by
  wlog h' : 0 ≤ x * y generalizing x y z; swap
  · rw [div_mul_eq_mul_div, le_div_iff₀' my_zero_lt_32]
    exact my_subst_wlog h' hxyz
  rcases (mul_nonneg_of_three x y z).resolve_left h' with h | h
  · convert this y z x _ h using 2 <;> linarith
  · convert this z x y _ h using 2 <;> linarith

private theorem my_proof₁ {a b c : ℝ} :
    |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)| ≤
      9 * Real.sqrt 2 / 32 * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2 :=
  calc _ = |(a - b) * (b - c) * (c - a) * -(a + b + c)| := by ring_nf
    _ ≤ _ := my_subst_proof₁ (a - b) (b - c) (c - a) (-(a + b + c)) (by ring)
    _ = _ := by ring

theorem imo_2006_p3 (a b c : ℝ) :
  a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2) ≤
  9 * Real.sqrt 2 / 32 * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2 :=
  le_trans (le_abs_self _) my_proof₁

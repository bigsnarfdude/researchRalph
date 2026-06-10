import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3 (x y z : ℤ)
  (h₀ : (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 = x * y * z) :
  x + y + z + 6 ∣ x ^ 3 + y ^ 3 + z ^ 3 := by
  first
  | solve | simp only [h₀]; omega
  | solve | omega
  | solve | norm_num
  | solve | simp; omega
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | linear_combination h₀
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; ring
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
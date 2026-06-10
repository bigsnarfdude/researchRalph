import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  try
  have hb : n ≤ 10 := by nlinarith [h₀, h₀]
  interval_cases n <;> omega
  try
  have hb : n ≤ 10 := by nlinarith [h₀, h₀]
  interval_cases n <;> simp_all <;> omega
  try
  have hb : n ≤ 20 := by nlinarith [h₀, h₀]
  interval_cases n <;> omega
  try
  have hb : n ≤ 20 := by nlinarith [h₀, h₀]
  interval_cases n <;> simp_all <;> omega
  try
  have hb : n ≤ 50 := by nlinarith [h₀, h₀]
  interval_cases n <;> omega
  try
  have hb : n ≤ 50 := by nlinarith [h₀, h₀]
  interval_cases n <;> simp_all <;> omega
  try
  have hb : n ≤ 100 := by nlinarith [h₀, h₀]
  interval_cases n <;> omega
  try
  have hb : n ≤ 100 := by nlinarith [h₀, h₀]
  interval_cases n <;> simp_all <;> omega
  first
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | nlinarith [sq_nonneg n, h₀, h₀]
  | solve | nlinarith [sq_nonneg (n - 1), h₀, h₀]
  | solve | simp only [h₀, h₀]; ring
  | solve | simp [h₀, h₀]; ring
  | solve | simp only [h₀, h₀]; norm_num
  | solve | simp [h₀, h₀]; norm_num
  | solve | simp only [h₀, h₀]; omega
  | solve | simp [h₀, h₀]; omega
  | solve | simp only [h₀, h₀]; linarith
  | solve | simp [h₀, h₀]; linarith
  | solve | simp only [h₀, h₀]; nlinarith
  | solve | simp [h₀, h₀]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₀ + h₀
  | solve | linear_combination h₀ + -h₀
  | solve | linear_combination -h₀ + h₀
  | solve | linear_combination 2 * h₀ + -h₀
  | solve | linear_combination -h₀ + 2 * h₀
  | solve | linear_combination 3 * h₀ + -h₀
  | solve | linear_combination -3 * h₀ + 2 * h₀
  | solve | push_cast; ring
  | solve | norm_cast; ring
  | solve | push_cast; omega
  | solve | norm_cast; omega
  | solve | push_cast; norm_num
  | solve | norm_cast; norm_num
  | solve | push_cast; linarith
  | solve | norm_cast; linarith
  | solve | ring_nf; omega
  | solve | ring_nf; norm_num
  | solve | ring_nf; ring
  | solve | ring_nf; linarith
  | solve | ring_nf; nlinarith
  | solve | ring_nf; simp
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; ring
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; simp
  | solve | push_cast; nlinarith
  | solve | push_cast; simp
  | solve | norm_cast; nlinarith
  | solve | norm_cast; simp
  | solve | field_simp; omega
  | solve | field_simp; norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
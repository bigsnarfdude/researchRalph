import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  try
  have := h₀ 0
  linarith
  try
  have := h₀ 0
  nlinarith
  try
  have := h₀ 0
  omega
  try
  have := h₀ 0
  norm_num
  try
  have := h₀ 0
  ring
  try
  have := h₀ 0
  simp
  try
  have := h₀ 0
  field_simp; ring
  try
  have := h₀ 1
  linarith
  try
  have := h₀ 1
  nlinarith
  try
  have := h₀ 1
  omega
  try
  have := h₀ 1
  norm_num
  try
  have := h₀ 1
  ring
  try
  have := h₀ 1
  simp
  try
  have := h₀ 1
  field_simp; ring
  try
  have := h₀ 2
  linarith
  try
  have := h₀ 2
  nlinarith
  try
  have := h₀ 2
  omega
  try
  have := h₀ 2
  norm_num
  try
  have := h₀ 2
  ring
  try
  have := h₀ 2
  simp
  try
  have := h₀ 2
  field_simp; ring
  try
  have := h₀ 3
  linarith
  try
  have := h₀ 3
  nlinarith
  try
  have := h₀ 3
  omega
  try
  have := h₀ 3
  norm_num
  try
  have := h₀ 3
  ring
  try
  have := h₀ 3
  simp
  try
  have := h₀ 3
  field_simp; ring
  try
  have := h₀ (-1)
  linarith
  try
  have := h₀ (-1)
  nlinarith
  try
  have := h₀ (-1)
  omega
  try
  have := h₀ (-1)
  norm_num
  try
  have := h₀ (-1)
  ring
  try
  have := h₀ (-1)
  simp
  try
  have := h₀ (-1)
  field_simp; ring
  try
  have := h₀ (-2)
  linarith
  try
  have := h₀ (-2)
  nlinarith
  try
  have := h₀ (-2)
  omega
  try
  have := h₀ (-2)
  norm_num
  try
  have := h₀ (-2)
  ring
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
  | solve | nlinarith [sq_nonneg p, h₀, h₀]
  | solve | nlinarith [sq_nonneg (p - 1), h₀, h₀]
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
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | field_simp; linarith [h₀, h₀]
  | solve | field_simp; nlinarith [h₀, h₀]
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
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
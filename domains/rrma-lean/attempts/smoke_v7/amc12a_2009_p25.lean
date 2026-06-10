import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p25 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1))) : abs (a 2009) = 0 := by
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  linarith
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  nlinarith
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  omega
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  norm_num
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  ring
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  field_simp; ring
  try
  have := h₂ 0 (⟨by norm_num, by norm_num⟩)
  field_simp; linarith
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 4 (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 4 (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 4 (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 4 (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  linarith
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  nlinarith
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  omega
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  norm_num
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  ring
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  field_simp; ring
  try
  have := h₂ 1 (⟨by norm_num, by norm_num⟩)
  field_simp; linarith
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 0 (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  omega [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ 2 (⟨by norm_num, by norm_num⟩)
  norm_num [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  linarith [h_s1, h_s2]
  try
  have h_s1 := h₂ 1 (⟨by norm_num, by norm_num⟩)
  have h_s2 := h₂ (-1) (⟨by norm_num, by norm_num⟩)
  nlinarith [h_s1, h_s2]
  first
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | nlinarith [sq_nonneg a, h₀, h₁, h₂]
  | solve | nlinarith [sq_nonneg (a - 1), h₀, h₁, h₂]
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
  | solve | simp [h₀, h₁, h₂]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ + -h₁
  | solve | linear_combination -h₀ + h₁
  | solve | linear_combination 2 * h₀ + -h₁
  | solve | linear_combination -h₀ + 2 * h₁
  | solve | linear_combination 3 * h₀ + -h₁
  | solve | linear_combination -3 * h₀ + 2 * h₁
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | field_simp; linarith [h₀, h₁, h₂]
  | solve | field_simp; nlinarith [h₀, h₁, h₂]
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
  | solve | push_cast; omega
  | solve | push_cast; norm_num
  | solve | push_cast; ring
  | solve | push_cast; linarith
  | solve | push_cast; nlinarith
  | solve | push_cast; simp
  | solve | norm_cast; omega
  | solve | norm_cast; norm_num
  | solve | norm_cast; ring
  | solve | norm_cast; linarith
  | solve | norm_cast; nlinarith
  | solve | norm_cast; simp
  | solve | field_simp; omega
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
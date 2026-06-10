import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1996_p5 (a b c r s t : ℝ) (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x ^ 3 + 3 * x ^ 2 + 4 * x - 11) (h₁ : ∀ x, g x = x ^ 3 + r * x ^ 2 + s * x + t)
  (h₂ : f a = 0) (h₃ : f b = 0) (h₄ : f c = 0) (h₅ : g (a + b) = 0) (h₆ : g (b + c) = 0)
  (h₇ : g (c + a) = 0) (h₈ : List.Pairwise (· ≠ ·) [a, b, c]) : t = 23 := by
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
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; ring
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; norm_num
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; omega
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; linarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; nlinarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; nlinarith
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
  | solve | field_simp; norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
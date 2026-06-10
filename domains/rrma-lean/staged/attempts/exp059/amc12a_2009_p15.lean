import Mathlib
set_option maxHeartbeats 64000000
open BigOperators Real Nat Topology Rat

private lemma I_pow_one : Complex.I ^ (1 : ℕ) = Complex.I := pow_one _
private lemma I_pow_two : Complex.I ^ (2 : ℕ) = -1 := Complex.I_sq
private lemma I_pow_three : Complex.I ^ (3 : ℕ) = -Complex.I := by
  rw [show (3 : ℕ) = 2 + 1 from rfl, pow_add, I_pow_two, I_pow_one]; ring
private lemma I_pow_four_mul (q : ℕ) : Complex.I ^ (4 * q) = 1 := by
  rw [pow_mul]; norm_num

private lemma block_sum (q : ℕ) :
    (∑ k ∈ Finset.Icc (4 * q + 1) (4 * q + 4), (k : ℂ) * Complex.I ^ k) =
    2 - 2 * Complex.I := by
  have hI4 := I_pow_four_mul q
  have hset : Finset.Icc (4 * q + 1) (4 * q + 4) =
    (Finset.range 4).image (· + (4 * q + 1)) := by
    ext x; simp only [Finset.mem_Icc, Finset.mem_image, Finset.mem_range]
    constructor
    · intro ⟨h1, h2⟩; exact ⟨x - (4 * q + 1), by omega, by omega⟩
    · rintro ⟨a, ha1, rfl⟩; omega
  rw [hset, Finset.sum_image (by intro a _ b _ h; omega : ∀ a ∈ Finset.range 4,
    ∀ b ∈ Finset.range 4, a + (4 * q + 1) = b + (4 * q + 1) → a = b)]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  have h1 : (1 + (4 * q + 1) : ℕ) = 4 * q + 2 := by omega
  have h2 : (2 + (4 * q + 1) : ℕ) = 4 * q + 3 := by omega
  have h3 : (3 + (4 * q + 1) : ℕ) = 4 * (q + 1) := by ring
  simp only [h1, h2, h3]
  rw [pow_add, pow_add, pow_add, hI4, I_pow_four_mul (q + 1),
      one_mul, I_pow_one, I_pow_two, I_pow_three]
  push_cast; ring

private lemma Icc_split (q : ℕ) :
    Finset.Icc 1 (4 * q + 4) = Finset.Icc 1 (4 * q) ∪ Finset.Icc (4 * q + 1) (4 * q + 4) := by
  ext x; simp only [Finset.mem_Icc, Finset.mem_union]; constructor <;> intro h <;> omega

private lemma Icc_disjoint (q : ℕ) :
    Disjoint (Finset.Icc 1 (4 * q)) (Finset.Icc (4 * q + 1) (4 * q + 4)) := by
  simp only [Finset.disjoint_left, Finset.mem_Icc]; intro x hx1 hx2; omega

lemma sum_complete_blocks (q : ℕ) :
    (∑ k ∈ Finset.Icc 1 (4 * q), (k : ℂ) * Complex.I ^ k) =
    2 * (q : ℂ) - 2 * (q : ℂ) * Complex.I := by
  induction q with
  | zero => simp
  | succ q ih =>
    rw [show 4 * (q + 1) = 4 * q + 4 from by ring,
        Icc_split q,
        Finset.sum_union (Icc_disjoint q),
        ih, block_sum q]
    push_cast; ring

private lemma Icc_snoc (a b : ℕ) (h : a ≤ b + 1) :
    Finset.Icc a (b + 1) = Finset.Icc a b ∪ {b + 1} := by
  ext x; simp only [Finset.mem_Icc, Finset.mem_union, Finset.mem_singleton]
  constructor <;> intro h <;> omega

private lemma Icc_disjoint_snoc (a b : ℕ) :
    Disjoint (Finset.Icc a b) ({b + 1} : Finset ℕ) := by
  simp only [Finset.disjoint_left, Finset.mem_Icc, Finset.mem_singleton]
  intro x hx1 hx2; omega

lemma sum_4q_plus_1 (q : ℕ) :
    (∑ k ∈ Finset.Icc 1 (4 * q + 1), (k : ℂ) * Complex.I ^ k) =
    2 * (q : ℂ) + (2 * (q : ℂ) + 1) * Complex.I := by
  rw [Icc_snoc 1 (4 * q) (by omega),
      Finset.sum_union (Icc_disjoint_snoc 1 (4 * q)),
      sum_complete_blocks q,
      Finset.sum_singleton,
      pow_add, I_pow_four_mul q, one_mul, I_pow_one]
  push_cast; ring

lemma sum_4q_plus_2 (q : ℕ) :
    (∑ k ∈ Finset.Icc 1 (4 * q + 2), (k : ℂ) * Complex.I ^ k) =
    -(2 * (q : ℂ) + 2) + (2 * (q : ℂ) + 1) * Complex.I := by
  rw [show 4 * q + 2 = (4 * q + 1) + 1 from by omega,
      Icc_snoc 1 (4 * q + 1) (by omega),
      Finset.sum_union (Icc_disjoint_snoc 1 (4 * q + 1)),
      sum_4q_plus_1 q,
      Finset.sum_singleton,
      show (4 * q + 1 + 1 : ℕ) = 4 * q + 2 from by omega,
      pow_add, I_pow_four_mul q, one_mul, I_pow_two]
  push_cast; ring

lemma sum_4q_plus_3 (q : ℕ) :
    (∑ k ∈ Finset.Icc 1 (4 * q + 3), (k : ℂ) * Complex.I ^ k) =
    -(2 * (q : ℂ) + 2) - (2 * (q : ℂ) + 2) * Complex.I := by
  rw [show 4 * q + 3 = (4 * q + 2) + 1 from by omega,
      Icc_snoc 1 (4 * q + 2) (by omega),
      Finset.sum_union (Icc_disjoint_snoc 1 (4 * q + 2)),
      sum_4q_plus_2 q,
      Finset.sum_singleton,
      show (4 * q + 2 + 1 : ℕ) = 4 * q + 3 from by omega,
      pow_add, I_pow_four_mul q, one_mul, I_pow_three]
  push_cast; ring

theorem amc12a_2009_p15 (n : ℕ) (h₀ : 0 < n)
  (h₁ : (∑ k ∈ Finset.Icc 1 n, ↑k * Complex.I ^ k) = 48 + 49 * Complex.I) : n = 97 := by
  set q := n / 4
  have hn : n = 4 * q + n % 4 := by omega
  have : n % 4 = 0 ∨ n % 4 = 1 ∨ n % 4 = 2 ∨ n % 4 = 3 := by omega
  rcases this with h | h | h | h
  · rw [show n = 4 * q from by omega] at h₁
    rw [sum_complete_blocks] at h₁
    have := congr_arg Complex.im h₁; simp at this; linarith
  · rw [show n = 4 * q + 1 from by omega] at h₁
    rw [sum_4q_plus_1] at h₁
    have hre := congr_arg Complex.re h₁; simp at hre
    have : (q : ℝ) = 24 := by linarith
    have : q = 24 := by exact_mod_cast this
    omega
  · rw [show n = 4 * q + 2 from by omega] at h₁
    rw [sum_4q_plus_2] at h₁
    have := congr_arg Complex.re h₁; simp at this; linarith
  · rw [show n = 4 * q + 3 from by omega] at h₁
    rw [sum_4q_plus_3] at h₁
    have := congr_arg Complex.im h₁; simp at this; linarith

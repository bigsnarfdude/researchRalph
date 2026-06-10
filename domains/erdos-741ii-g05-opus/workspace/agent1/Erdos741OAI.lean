import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Numbers whose base-4 digits are all ≤ 1. -/
def E : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}

/-- Numbers whose base-4 digits are all 0 or 2. -/
def O : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d = 0 ∨ d = 2}

lemma memE_step {e a : ℕ} (he : e ∈ E) (ha : a ≤ 1) : 4 * e + a ∈ E := by
  rcases Nat.eq_zero_or_pos (4 * e + a) with h0 | hpos
  · simp [E, h0]
  · have hb : (2:ℕ) ≤ 4 := by norm_num
    have hdig : Nat.digits 4 (4 * e + a) = (4 * e + a) % 4 :: Nat.digits 4 ((4 * e + a) / 4) :=
      Nat.digits_def' hb hpos
    have hmod : (4 * e + a) % 4 = a := by omega
    have hdiv : (4 * e + a) / 4 = e := by omega
    rw [hmod, hdiv] at hdig
    intro d hd
    rw [hdig] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact he d h

lemma memO_step {o b : ℕ} (ho : o ∈ O) (hb : b = 0 ∨ b = 2) : 4 * o + b ∈ O := by
  rcases Nat.eq_zero_or_pos (4 * o + b) with h0 | hpos
  · simp [O, h0]
  · have hb4 : (2:ℕ) ≤ 4 := by norm_num
    have hblt : b < 4 := by omega
    have hdig : Nat.digits 4 (4 * o + b) = (4 * o + b) % 4 :: Nat.digits 4 ((4 * o + b) / 4) :=
      Nat.digits_def' hb4 hpos
    have hmod : (4 * o + b) % 4 = b := by omega
    have hdiv : (4 * o + b) / 4 = o := by omega
    rw [hmod, hdiv] at hdig
    intro d hd
    rw [hdig] at hd
    rcases List.mem_cons.mp hd with h | h
    · omega
    · exact ho d h

lemma E_mod {n : ℕ} (h : n ∈ E) : n % 4 ≤ 1 := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · simp [hn]
  · have hdig : Nat.digits 4 n = n % 4 :: Nat.digits 4 (n / 4) :=
      Nat.digits_def' (by norm_num) hn
    exact h _ (by rw [hdig]; exact List.mem_cons.mpr (Or.inl rfl))

lemma E_div {n : ℕ} (h : n ∈ E) : n / 4 ∈ E := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · simp [hn, E]
  · have hdig : Nat.digits 4 n = n % 4 :: Nat.digits 4 (n / 4) :=
      Nat.digits_def' (by norm_num) hn
    intro d hd
    exact h d (by rw [hdig]; exact List.mem_cons.mpr (Or.inr hd))

lemma E_digit {n : ℕ} (h : n ∈ E) : ∀ k, n / 4 ^ k % 4 ≤ 1 := by
  intro k
  induction k generalizing n with
  | zero => simpa using E_mod h
  | succ k ih =>
    have : n / 4 ^ (k + 1) = n / 4 / 4 ^ k := by
      rw [pow_succ, Nat.div_div_eq_div_mul, Nat.mul_comm]
    rw [this]
    exact ih (E_div h)

lemma O_mod {n : ℕ} (h : n ∈ O) : n % 4 = 0 ∨ n % 4 = 2 := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · simp [hn]
  · have hdig : Nat.digits 4 n = n % 4 :: Nat.digits 4 (n / 4) :=
      Nat.digits_def' (by norm_num) hn
    exact h _ (by rw [hdig]; exact List.mem_cons.mpr (Or.inl rfl))

lemma O_div {n : ℕ} (h : n ∈ O) : n / 4 ∈ O := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · simp [hn, O]
  · have hdig : Nat.digits 4 n = n % 4 :: Nat.digits 4 (n / 4) :=
      Nat.digits_def' (by norm_num) hn
    intro d hd
    exact h d (by rw [hdig]; exact List.mem_cons.mpr (Or.inr hd))

lemma O_digit {n : ℕ} (h : n ∈ O) : ∀ k, n / 4 ^ k % 4 = 0 ∨ n / 4 ^ k % 4 = 2 := by
  intro k
  induction k generalizing n with
  | zero => simpa using O_mod h
  | succ k ih =>
    have : n / 4 ^ (k + 1) = n / 4 / 4 ^ k := by
      rw [pow_succ, Nat.div_div_eq_div_mul, Nat.mul_comm]
    rw [this]
    exact ih (O_div h)

/-- Key structural fact: `A = E ∪ O` is empty on the window `[3·4^k, 4·4^k)`. -/
lemma gap_window (k : ℕ) {n : ℕ} (h1 : 3 * 4 ^ k ≤ n) (h2 : n < 4 * 4 ^ k) :
    n ∉ E ∧ n ∉ O := by
  have hpos : 0 < 4 ^ k := Nat.pos_pow_of_pos k (by norm_num)
  have hdiv : n / 4 ^ k = 3 := by
    rw [Nat.div_eq_of_lt_le]
    · calc 3 * 4 ^ k ≤ n := h1
    · calc n < 4 * 4 ^ k := h2
        _ = (3 + 1) * 4 ^ k := by ring
  constructor
  · intro hE
    have := E_digit hE k
    rw [hdiv] at this
    omega
  · intro hO
    have := O_digit hO k
    rw [hdiv] at this
    omega

/-- Every natural number splits as `e + o` with `e ∈ E`, `o ∈ O`. -/
lemma decomp : ∀ n : ℕ, ∃ e ∈ E, ∃ o ∈ O, e + o = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · exact ⟨0, by simp [E, hn], 0, by simp [O, hn], by simp [hn]⟩
    · have hmlt : n / 4 < n := Nat.div_lt_self hn (by norm_num)
      obtain ⟨e', he', o', ho', hsum⟩ := ih (n / 4) hmlt
      set d := n % 4 with hd
      have hdlt : d < 4 := Nat.mod_lt n (by norm_num)
      refine ⟨4 * e' + d % 2, memE_step he' (by omega), 4 * o' + (d - d % 2),
        memO_step ho' (by omega), ?_⟩
      have hab : d % 2 + (d - d % 2) = d := by omega
      have : 4 * e' + d % 2 + (4 * o' + (d - d % 2)) = 4 * (e' + o') + d := by ring_nf; omega
      rw [this, hsum]
      have := Nat.div_add_mod n 4
      omega

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨E ∪ O, ?_, ?_⟩
  · intro n _
    obtain ⟨e, he, o, ho, hsum⟩ := decomp n
    exact ⟨e, Or.inl he, o, Or.inr ho, hsum⟩
  · intro A₁ A₂ _h1sub _h2sub _hcover _hdisj
    rintro ⟨hs1, hs2⟩
    obtain ⟨C₁, hC₁⟩ := hs1
    obtain ⟨C₂, hC₂⟩ := hs2
    sorry

end Erdos741OAI

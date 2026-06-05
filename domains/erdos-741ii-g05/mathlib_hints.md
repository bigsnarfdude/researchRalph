# Mathlib Hints — nat arithmetic and set membership

These are exact Lean 4 / Mathlib 4 patterns. Argument order and lemma names are precise.

## Nat subtraction — omega FAILS here, use these instead

-- Goal: m - n + n = m  (when n ≤ m)
Nat.sub_add_cancel : n ≤ m → m - n + n = m

-- Goal: n + (m - n) = m  (when n ≤ m)
Nat.add_sub_cancel' : n ≤ m → n + (m - n) = m

-- Goal: m - n ≤ k ↔ m ≤ k + n  (when n ≤ m)
Nat.sub_le_iff_le_add : n ≤ m → (m - n ≤ k ↔ m ≤ k + n)

-- Goal: k ≤ m - n ↔ k + n ≤ m
Nat.le_sub_iff_add_le (h : n ≤ m) : k ≤ m - n ↔ k + n ≤ m

-- Useful: m - n = 0 when n > m
Nat.sub_eq_zero_of_le : m ≤ n → m - n = 0

-- Useful: split n into k + (n - k) when k ≤ n
have : n = k + (n - k) := (Nat.add_sub_cancel' hkn).symm

## Interval membership

-- Set.Icc (closed interval [a, b])
Set.mem_Icc : x ∈ Set.Icc a b ↔ a ≤ x ∧ x ≤ b

-- Set.Ico (half-open [a, b))
Set.mem_Ico : x ∈ Set.Ico a b ↔ a ≤ x ∧ x < b

-- Set.Ioi (open ray (a, ∞))
Set.mem_Ioi : x ∈ Set.Ioi a ↔ a < x

## Set union / union membership

Set.mem_union : x ∈ s ∪ t ↔ x ∈ s ∨ x ∈ t
Set.mem_iUnion : x ∈ ⋃ i, s i ↔ ∃ i, x ∈ s i
Set.mem_iUnion₂ : x ∈ ⋃ i j, s i j ↔ ∃ i j, x ∈ s i j

-- Singleton membership
Set.mem_singleton_iff : x ∈ ({a} : Set α) ↔ x = a

-- Insert membership
Set.mem_insert_iff : x ∈ insert a s ↔ x = a ∨ x ∈ s

## Pointwise set addition (A + B = {a + b | a ∈ A, b ∈ B})
-- Requires: open scoped Pointwise  (already in file)

Set.mem_add : z ∈ s + t ↔ ∃ x ∈ s, ∃ y ∈ t, x + y = z

-- To show z ∈ s + t:
exact ⟨x, hx, y, hy, rfl⟩   -- when x + y = z by rfl

## IsSyndetic (already defined in file)
-- IsSyndetic S := ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)
-- meaning: there is a bound C such that every interval [x, x+C] contains a point of S

-- To disprove IsSyndetic (show NOT syndetic):
push_neg   -- ¬ IsSyndetic S becomes: ∀ C, ∃ x, ∀ m ∈ S, m ∉ Icc x (x + C)
-- i.e. for every gap bound C you claim, find an interval [x, x+C] with no point of S

## Power bounds

Nat.one_le_two_pow : 1 ≤ 2 ^ n
Nat.two_pow_pos : 0 < 2 ^ n
Nat.pow_lt_pow_right : 1 < b → n < m → b ^ n < b ^ m
Nat.le_of_dvd : 0 < n → m ∣ n → m ≤ n

-- 2^n grows fast:
have h : 2 ^ k ≤ 2 ^ (k + 1) := Nat.pow_le_pow_right (by norm_num) (Nat.le_succ k)

## Omega scope
-- omega handles: linear ℕ/ℤ equalities and inequalities with +, -, *constant
-- omega does NOT handle: nat-sub (a - b when a b : ℕ), exponentiation, multiplication of variables
-- when omega fails on nat-sub: use Nat.sub_add_cancel / Nat.add_sub_cancel' first

## linarith vs nlinarith
-- linarith: linear arithmetic over ordered fields/rings (works with hypotheses)
-- nlinarith: nonlinear — use when product of hypotheses needed, e.g. k * k ≤ n
-- both accept extra lemmas: linarith [Nat.two_pow_pos k, h1, h2]

## Useful simp lemmas for this proof domain
simp [Set.mem_Icc, Set.mem_union, Set.mem_iUnion, Set.mem_add]
simp [Nat.add_sub_cancel, Nat.sub_add_cancel]

## Proof structure tips
-- To prove ∃ x, P x:  use exact ⟨witness, proof⟩  or  exact ⟨witness, by ...⟩
-- To prove A ⊆ B:  intro x hx; ...
-- To prove ¬ P:  intro h; derive False
-- intro ⟨h1, h2⟩  destructs And hypotheses
-- obtain ⟨a, ha, b, hb, hab⟩ := h  destructs existentials

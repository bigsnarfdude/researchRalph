import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- ===== Construction =====
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

-- ===== Basic Q facts =====
lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k
lemma Q_one_le (k : ℕ) : 1 ≤ Q k := Q_pos k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]
lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

end Erdos741OAI

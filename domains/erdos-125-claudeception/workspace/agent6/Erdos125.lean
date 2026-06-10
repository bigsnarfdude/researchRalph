import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩
  simp only [setA, setB, Set.mem_setOf] at ha hb
  have ha_eq : a = 62 - b := by omega
  rw [ha_eq] at ha
  have hb_le : b ≤ 62 := by omega

  -- Values of b and their contradictions
  match b with
  | 0 => simp [Nat.digits] at ha; norm_num at ha
  | 1 => simp [Nat.digits] at ha; norm_num at ha
  | 2 => simp [Nat.digits] at hb; norm_num at hb
  | 3 => simp [Nat.digits] at hb; norm_num at hb
  | 4 => simp [Nat.digits] at ha; norm_num at ha
  | 5 => simp [Nat.digits] at ha; norm_num at ha
  | 6 => simp [Nat.digits] at hb; norm_num at hb
  | 7 => simp [Nat.digits] at hb; norm_num at hb
  | 8 => simp [Nat.digits] at hb; norm_num at hb
  | 9 => simp [Nat.digits] at hb; norm_num at hb
  | 10 => simp [Nat.digits] at hb; norm_num at hb
  | 11 => simp [Nat.digits] at hb; norm_num at hb
  | 12 => simp [Nat.digits] at hb; norm_num at hb
  | 13 => simp [Nat.digits] at hb; norm_num at hb
  | 14 => simp [Nat.digits] at hb; norm_num at hb
  | 15 => simp [Nat.digits] at hb; norm_num at hb
  | 16 => simp [Nat.digits] at ha; norm_num at ha
  | 17 => simp [Nat.digits] at ha; norm_num at ha
  | 18 => simp [Nat.digits] at hb; norm_num at hb
  | 19 => simp [Nat.digits] at hb; norm_num at hb
  | 20 => simp [Nat.digits] at ha; norm_num at ha
  | 21 => simp [Nat.digits] at ha; norm_num at ha
  | 22 => simp [Nat.digits] at hb; norm_num at hb
  | 23 => simp [Nat.digits] at hb; norm_num at hb
  | 24 => simp [Nat.digits] at hb; norm_num at hb
  | 25 => simp [Nat.digits] at hb; norm_num at hb
  | 26 => simp [Nat.digits] at hb; norm_num at hb
  | 27 => simp [Nat.digits] at hb; norm_num at hb
  | 28 => simp [Nat.digits] at hb; norm_num at hb
  | 29 => simp [Nat.digits] at hb; norm_num at hb
  | 30 => simp [Nat.digits] at hb; norm_num at hb
  | 31 => simp [Nat.digits] at hb; norm_num at hb
  | 32 => simp [Nat.digits] at hb; norm_num at hb
  | 33 => simp [Nat.digits] at hb; norm_num at hb
  | 34 => simp [Nat.digits] at hb; norm_num at hb
  | 35 => simp [Nat.digits] at hb; norm_num at hb
  | 36 => simp [Nat.digits] at hb; norm_num at hb
  | 37 => simp [Nat.digits] at hb; norm_num at hb
  | 38 => simp [Nat.digits] at hb; norm_num at hb
  | 39 => simp [Nat.digits] at hb; norm_num at hb
  | 40 => simp [Nat.digits] at hb; norm_num at hb
  | 41 => simp [Nat.digits] at hb; norm_num at hb
  | 42 => simp [Nat.digits] at hb; norm_num at hb
  | 43 => simp [Nat.digits] at hb; norm_num at hb
  | 44 => simp [Nat.digits] at hb; norm_num at hb
  | 45 => simp [Nat.digits] at hb; norm_num at hb
  | 46 => simp [Nat.digits] at hb; norm_num at hb
  | 47 => simp [Nat.digits] at hb; norm_num at hb
  | 48 => simp [Nat.digits] at hb; norm_num at hb
  | 49 => simp [Nat.digits] at hb; norm_num at hb
  | 50 => simp [Nat.digits] at hb; norm_num at hb
  | 51 => simp [Nat.digits] at hb; norm_num at hb
  | 52 => simp [Nat.digits] at hb; norm_num at hb
  | 53 => simp [Nat.digits] at hb; norm_num at hb
  | 54 => simp [Nat.digits] at hb; norm_num at hb
  | 55 => simp [Nat.digits] at hb; norm_num at hb
  | 56 => simp [Nat.digits] at hb; norm_num at hb
  | 57 => simp [Nat.digits] at hb; norm_num at hb
  | 58 => simp [Nat.digits] at hb; norm_num at hb
  | 59 =>
    exfalso
    have h59 : Nat.digits 4 59 = [3, 3, 3] := by decide
    rw [h59] at hb
    simp at hb
  | 60 =>
    exfalso
    have h60 : Nat.digits 4 60 = [0, 3, 3] := by decide
    rw [h60] at hb
    simp at hb
  | 61 =>
    exfalso
    have h61 : Nat.digits 4 61 = [1, 3, 3] := by decide
    rw [h61] at hb
    simp at hb
  | 62 =>
    exfalso
    have h62 : Nat.digits 4 62 = [2, 3, 3] := by decide
    rw [h62] at hb
    simp at hb
  | n + 63 => omega

import Mathlib
import Mathlib.Tactic.Linarith

/--
Problem: f(1)=2, f(f n) = f n + n, f strictly increasing.
-/
def IsSolution (f : ℕ → ℕ) : Prop :=
  f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1))

/--
The recursive jump property:
If f(n+1)-f(n) = 1, then f(f(n)+1)-f(f(n)) = 2.
If f(n+1)-f(n) = 2, then (f(f(n)+1)-f(f(n))) + (f(f(n)+2)-f(f(n)+1)) = 3.
-/
theorem jump_morphism (f : ℕ → ℕ) (h : IsSolution f) (n : ℕ) :
  ((f (n + 1) : ℤ) - (f n : ℤ) = 1 → (f (f n + 1) : ℤ) - (f (f n) : ℤ) = 2) ∧
  ((f (n + 1) : ℤ) - (f n : ℤ) = 2 → 
    ((f (f n + 1) : ℤ) - (f (f n) : ℤ)) + ((f (f n + 2) : ℤ) - (f (f n + 1) : ℤ)) = 3) := by
  
  -- f(f(n+1)) - f(f(n)) = (f(n+1)+n) - (f(n)+n) = f(n+1) - f(n) + 1
  have f_f_diff : (f (f (n + 1)) : ℤ) - (f (f n) : ℤ) = ((f (n + 1) : ℤ) - (f n : ℤ)) + 1 := by
    simp [h.2.1]; ring
  
  have h_telescope : (f (f (n + 1)) : ℤ) - (f (f n) : ℤ) = 
    (Finset.range (f (n + 1) - f n)).sum (fun i => (f (f n + i + 1) : ℤ) - (f (f n + i) : ℤ)) := by
    set k := f (n + 1) - f n
    rw [Finset.sum_range_sub (fun i => (f (f n + i) : ℤ))]
    simp
    have : f n + k = f (n + 1) := by omega
    rw [this]

  constructor
  · intro hdn1
    have : (Finset.range (f (n + 1) - f n)).sum (fun i => (f (f n + i + 1) : ℤ) - (f (f n + i) : ℤ)) = 2 := by
      rw [← h_telescope, f_f_diff, hdn1]; ring
    have h_k : f (n + 1) - f n = 1 := by omega
    rw [h_k] at this
    simp at this
    exact this
  · intro hdn2
    have : (Finset.range (f (n + 1) - f n)).sum (fun i => (f (f n + i + 1) : ℤ) - (f (f n + i) : ℤ)) = 3 := by
      rw [← h_telescope, f_f_diff, hdn2]; ring
    have h_k : f (n + 1) - f n = 2 := by omega
    rw [h_k] at this
    simp [Finset.sum_range_succ] at this
    exact this

/--
Show f(1)=2, f(2)=3, f(3)=5.
-/
theorem initial_values (f : ℕ → ℕ) (h : IsSolution f) :
  f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 5 := by
  have f1 : f 1 = 2 := h.1
  have ff1 : f (f 1) = f 1 + 1 := h.2.1 1
  have f2 : f 2 = 3 := by omega
  have ff2 : f (f 2) = f 2 + 2 := h.2.1 2
  have f3 : f 3 = 5 := by omega
  exact ⟨f1, f2, f3⟩

/--
Show f(4) ∈ {6, 7}.
-/
theorem f4_values (f : ℕ → ℕ) (h : IsSolution f) :
  f 4 = 6 ∨ f 4 = 7 := by
  obtain ⟨f1, f2, f3⟩ := initial_values f h
  have ff3 : f (f 3) = f 3 + 3 := h.2.1 3
  have f5 : f 5 = 8 := by omega
  have h4_1 : f 3 < f 4 := h.2.2 3
  have h4_2 : f 4 < f 5 := h.2.2 4
  omega

import Mathlib

/-- Fibonacci sequence: F_0 = 0, F_1 = 1, F_n+2 = F_n+1 + F_n -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib (n + 1) + fib n

theorem fib_two : fib 2 = 1 := rfl
theorem fib_three : fib 3 = 2 := rfl
theorem fib_four : fib 4 = 3 := rfl
theorem fib_five : fib 5 = 5 := rfl
theorem fib_six : fib 6 = 8 := rfl

theorem fib_property (f : ℕ → ℕ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ n, f (f n) = f n + n) 
  (h3 : ∀ n, f n < f (n + 1)) : 
  ∀ n ≥ 2, f (fib n) = fib (n + 1) := by
  intro n hn
  induction n, hn using Nat.le_induction with
  | base => 
    -- f(fib 2) = f(1) = 2 = fib 3
    rw [fib_two, h1, fib_three]
  | succ k hk ih =>
    -- Assume f(fib k) = fib (k+1)
    -- Prove f(fib (k+1)) = fib (k+2)
    -- We know f(f(fib k)) = f(fib k) + fib k
    -- Sub ih: f(fib (k+1)) = fib (k+1) + fib k
    -- By definition of fib: fib (k+1) + fib k = fib (k+2)
    -- So f(fib (k+1)) = fib (k+2)
    have h_f_f := h2 (fib k)
    rw [ih] at h_f_f
    exact h_f_f

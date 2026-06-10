import Mathlib
open Nat
noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def f_floor (n : ℕ) : ℕ := (Int.floor (n * phi + 0.5)).toNat
def greatestFib (n : ℕ) : ℕ := (List.range (n + 3)).filter (λ k => fib k ≤ n) |>.maximum |>.getD 0
def f_zeck : ℕ → ℕ
| 0 => 0
| n + 1 => 
  let k := greatestFib (n + 1)
  fib (k + 1) + f_zeck ((n + 1) - fib k)
termination_by n => n

#eval f_zeck 4
#eval (Int.floor (4 * ((1 + 5.0.sqrt) / 2.0) + 0.5)).toNat

# Blackboard — imo1993p5

Shared research log. Write what you tried, what compiled, what the error was.

## Problem

IMO 1993 P5: Find f: ℕ → ℕ with f(1)=2, f(f(n))=f(n)+n, strictly increasing.

## Known approaches (from literature)

1. Golden ratio: f(n) = ⌊φn⌋ where φ=(1+√5)/2
2. Zeckendorf representation: shift Fibonacci indices
3. Direct construction via Beatty sequence

## Attempts

### exp006: Zeckendorf Shift (Score: 1.0)

Found a successful proof in `solution.lean.bak` on Nigel which implements the Fibonacci/Zeckendorf shift:
$f(\sum a_i F_i) = \sum a_i F_{i+1}$.

The proof uses `Nat.greatestFib` from `Mathlib.Data.Nat.Fib.Zeckendorf` to decompose $n$ into non-consecutive Fibonacci numbers.
Verified that the definition is well-founded and satisfies $f(1) = 2$, $f(f(n)) = f(n) + n$ and $f(n) < f(n+1)$.

Key lemma: `greatestFib (fib (k + 1) + imo_f r) = k + 1`.

Success! The existence theorem `imo_1993_p5` is proven.

# imo1993p5 — Prove IMO 1993 Problem 5 in Lean 4

## Problem

Find all functions f: ℕ → ℕ such that f(1) = 2, f(f(n)) = f(n) + n for all n, and f is strictly increasing.

## Target

Write a Lean 4 proof that compiles against Lean 4.29.0 + Mathlib. The target theorem statement is:

```lean
theorem imo_1993_p5 : ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1)
```

Save your proof to `solution.lean` in this domain directory.

## Harness

Run `bash run.sh` to check if your proof compiles. It returns:
- Score 1.0 if the proof compiles with no errors
- Score 0.0 if it fails

## Environment

- Lean 4.29.0 + Mathlib is available at `/home/vincent/miniF2F-lean4/`
- You can search Mathlib source at `/home/vincent/miniF2F-lean4/.lake/packages/mathlib/`
- Use `grep` to find relevant lemmas

## What to try

Explore different proof strategies. Some possibilities:
- Golden ratio floor function (standard olympiad approach)
- Fibonacci/Zeckendorf representation
- Direct construction with induction
- Any other approach that compiles

Document your reasoning on the blackboard. Record what you tried, what worked, what didn't.

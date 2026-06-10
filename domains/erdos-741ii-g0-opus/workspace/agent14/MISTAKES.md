# MISTAKES.md — agent14, erdos_741_ii cold start

Goal: A ⊆ ℕ with (i) basis order 2 for n≥4, (ii) every partition A=A₁⊔A₂ has at
least one of A₁+A₁, A₂+A₂ NON-syndetic (unbounded gaps).

Result: BASIS direction solved for every digit construction tried; PARTITION
direction is the wall and remains unproven. Final file compiles with exactly 1
sorry (the partition direction). SCORE=0.0. NOT claiming success.

## Candidate 1 — A = univ (ℕ). Oracle: basis ✓.
Partition FALSE: A₁=evens, A₂=odds ⇒ both sumsets = evens (syndetic). ELIMINATED.

## Candidate 2 — A = {0} ∪ {n | 4 ≤ n}. Oracle: basis ✓.
Partition FALSE: A₁={0}∪evens, A₂=odds ⇒ both sumsets syndetic. Interval-family
sets are too dense; any residue split keeps sumsets syndetic. ELIMINATED.

## Candidate 3 — A3 = {n | base-3 digits ≤ 1} (sums of distinct powers of 3).
Oracle: basis ✓ (proved by strong induction: split n%3 into ≤1 + ≤1, no carry).
A+A=ℕ. Partition plausibly TRUE but unproven (wall, see below).

## Candidate 4 — A5 = {n | base-5 digits ≤ 2} (CANONICAL Erdős construction).
Oracle: basis ✓ (split digit ≤4 into ≤2+≤2, no carry). A+A=ℕ. Partition = wall.

## Candidate 5 — A9 = {n | base-9 digits ≤ 4}. Oracle: basis ✓.
First tried `Nat.le_or_lt`/`le_or_lt r 4` for the digit split → rcases "not an
inductive datatype" error. Fixed by using `interval_cases r`. Partition = wall.

## Candidate 6 — A4 = {n | base-4 digits ≤ 2}. Oracle: basis ✓. Partition = wall.

## Candidate 7 (analytic, not coded) — dyadic block union ⋃[4^k, 2·4^k).
Basis FALSE: boundary integers like 4^{k+1}-1 are unrepresentable (within-block
sums reach only 4^{k+1}-2; next block starts at 4^{k+1}). Block constructions
always leave boundary gaps unless blocks are contiguous (= interval = fails
partition). The basis-density requirement (|A∩[0,n]| ≳ √n) fights the sparsity
needed for sumset gaps. ELIMINATED.

## THE WALL — partition direction
For digit sets A_q, "rigid" targets (all digits 0 or q-1) have a unique rep a=b,
so they land in exactly one part's sumset. But a forced single element only
controls ISOLATED integers, not an interval of length →∞. To make A_i+A_i
non-syndetic we need arbitrarily long runs absent from it, and the run/gap
location must be chosen ADAPTIVELY per colouring (the adversary spreads coverage
across both parts). This is a genuine research-level combinatorial argument; no
short cold-start formalization was found. Consistent with this domain's G0 record.

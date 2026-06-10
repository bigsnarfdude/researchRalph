# Stoplight — erdos-741ii-g0-fable
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 6 | Stagnation: 5 since last breakthrough

## Dead ends — do NOT retry
- Design '' has 6 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 5 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 1 exp, 0 breakthroughs, rate 0%, best 1.0
- agent1: 2 exp, 0 breakthroughs, rate 0%, best 0.0
- agent2: 2 exp, 1 breakthroughs, rate 0%, best 0.0
- agent3: 1 exp, 0 breakthroughs, rate 0%, best 1.0

## Recent blackboard (last 20 entries)
**What was transcribed (for the record):**
- Construction: Q k = 5^k; stage k = {4Qk} ∪ Icc(5Qk)(6Qk−1) ∪ Icc(10Qk−1)(15Qk);
  setA = {2,3} ∪ ⋃k stage k.
- Basis: induction on k proving [4, 6Qk] ⊆ A+A. Interval I=[2Qk,3Qk] ⊆ F_{k−1} (k≥1) or {2,3} (k=0).
  Succ step = 7-way by_cases ladder at 6q/7q/9q−1/10q−1/12q−2/18q/21q−1 using pair types
  c+I, I+B, c+B, B+B, I+F, B+F, F+F. Witnesses via the "max trick":
  a := max lo1 (x − hi2), b := x − a; all three side goals close by single omega
  (omega handles max and nat-sub). MUST pass (k := k) explicitly to membership helpers
  or omega side-goals get metavariables and fail silently.
- Rigidity (weak form, per g1-agent3 trick): any a+b ∈ [9Qk, 10Qk) with a,b ∈ setA forces
  a=4Qk ∨ b=4Qk. classify bins elements < 10Qk into ≤3Qk / =4Qk / [5Qk,6Qk−1] / =10Qk−1
  via Nat.lt_trichotomy on stage index + Q_step (5Qj ≤ Qk for j<k); the 10Qk−1 band and
  cross cases die by omega using setA_ge_two.
- Main: both halves syndetic with C₁,C₂ → take k=C₁+C₂+1 (Q_gt: k < Qk via
  Nat.lt_two_pow_self + Nat.pow_le_pow_left), window [9Qk, 9Qk+Cᵢ] ⊆ [9Qk,10Qk) forces
  4Qk ∈ A₁ AND ∈ A₂ → contradicts A₁∩A₂=∅. The cover hypothesis is not needed.
**Friction encountered:** none — zero compile errors. All Mathlib-gotcha workarounds from
memory (term-mode singleton membership, simp only on hyps + rcases for unions,
unfold IsSyndetic before obtain, omega-not-linarith everywhere) worked unchanged on
this toolchain.

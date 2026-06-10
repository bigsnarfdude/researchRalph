# erdos-741ii-g1 — NL construction + Mathlib hints

## Task
Prove `erdos_741_ii` in `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`.
The theorem statement and `IsSyndetic` definition are already in the file.
You must define the construction, prove it's a basis, and prove no partition is both-syndetic.

## The construction (explicit)

Define `Q k = 5^k`. Build the set:

```
A = {2, 3} ∪ ⋃_k  ({ck k} ∪ Bk k ∪ Fk k)
```

where at each level k:
- `ck k = 4 * Q k`            — a single "connector" element
- `Bk k = [5*Qk, 6*Qk - 1]`  — a "body" interval
- `Fk k = [10*Qk - 1, 15*Qk]` — a "filler" interval

Also define the "gap zone":
- `Jk k = [9*Qk, 10*Qk)`     — the interval with bounded gaps property

## Why A is a basis (covers all n ≥ 4)

Define `Akn k` = the partial union up through level k:
- `Akn 0 = {2, 3}`
- `Akn (k+1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k`

Prove by induction: `Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1)`.

At level k, the "I" interval is `[2*Qk, 3*Qk]` (inherited from previous level via `Fk`).
Eight pair types cover `[4*Qk, 30*Qk]`:
`I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk`
Cover by `by_cases` on which subinterval x falls in; exhibit the pair explicitly.

Since `n ≤ Q n` (Q grows faster than linear), every n falls in some `Icc 4 (6 * Q n)`.

## Why no partition is both-syndetic (the rigidity argument)

**Rigidity lemma:** For any `n ∈ Jk k = [9*Qk, 10*Qk)`, if `a + b = n` with `a, b ∈ A`,
then `(a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)`.

Proof by stage decomposition:
- Elements from `Icc 2 3`: too small (≤ 3 << 9*Qk)
- Elements from stage `j < k`: bounded above by `15 * Q j ≤ 3 * Q k` (geometric decay)
- Elements from stage `j > k`: bounded below by `4 * Q j ≥ 20 * Q k > n` (geometric growth)
- At stage `j = k`: only `4*Qk + [5*Qk, 6*Qk-1]` sums into `[9*Qk, 10*Qk)`

**Gap lemma:** If `ck k ∉ T` (where `T ⊆ A`), then `Jk k ∩ (T + T) = ∅`.

**Main argument:** Given a partition `A = A₁ ⊔ A₂`, pick k with `Q k > max(C₁, C₂)` where
C₁, C₂ are the gap bounds. Since `ck k ∈ A`, it goes to one side — say `ck k ∈ A₁`.
Then `Jk k ∩ (A₂ + A₂) = ∅` (by gap lemma). But `A₂ + A₂` is syndetic with bound C₂,
so it must hit `[9*Qk, 9*Qk + C₂] ⊆ Jk k` — contradiction.

## Read mathlib_hints.md first

It contains the exact Mathlib lemma names, argument orders, and tactic rules for this proof.
The most critical: use `omega` for any ℕ subtraction goal (not `linarith`).

## Oracle
```bash
bash run.sh
```
SCORE=1.0 when file compiles with 0 sorry.

## Workflow
1. Read `mathlib_hints.md` — know your tools before you start
2. Define Q, ck, Bk, Fk, Jk, setA, Akn in the workspace file
3. Prove helper lemmas in order: Q_pos/Q_succ, akn_mono, basis_lem, rigidity, gap_lem
4. Prove erdos_741_ii using gap_lem
5. bash run.sh after every meaningful edit
6. Write findings to MISTAKES.md, DESIRES.md, LEARNINGS.md

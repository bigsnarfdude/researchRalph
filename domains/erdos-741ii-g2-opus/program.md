# erdos-741ii-g2 — skeleton scaffold

## Task
Prove Erdős #741(ii) in Lean 4 + Mathlib by filling in two `sorry` lemmas in your workspace file.

## The two sorries

### 1. `basis_lem` (line ~83)
**Statement:** `Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1)`

Every integer in [4, 6·5^k] is a sum of two elements from Akn(k+1).

The proof is by induction on k. At each level, the pieces available are:
- `I = Icc (2 * Q k) (3 * Q k)` — inherited via `ik_sub_akn`
- `ck k = 4 * Q k` — singleton
- `Bk k = Icc (5 * Q k) (6 * Q k - 1)`
- `Fk k = Icc (10 * Q k - 1) (15 * Q k)`

Eight pair types cover [4Qk, 30Qk]:
`I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk`

Recommended approach: `by_cases` on which interval x falls in, then exhibit the pair explicitly using `Nat.sub_add_cancel`.

### 2. `rigidity` (line ~118)
**Statement:** For `n ∈ Jk k = [9·Qk, 10·Qk)`, if `a + b = n` with `a, b ∈ setA`, then `(a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)`.

The argument: `setA` decomposes as `Icc 2 3 ∪ ⋃j, ({ck j} ∪ Bk j ∪ Fk j)`.
- Elements from `Icc 2 3` are too small (≤3)
- Elements from stage `j < k` are ≤ 3·Qk (via `small_stage`)
- Elements from stage `j > k` are ≥ 20·Qk > n (via `large_stage`)
- At stage `j = k`: only pair summing into [9Qk,10Qk) is `ck k` + `Bk k`

## Mathlib friction points (critical)

- **Nat subtraction**: `Bk k = Icc (5*Qk) (6*Qk - 1)` and `Fk k = Icc (10*Qk-1) (15*Qk)` use ℕ subtraction. Use `omega` (not `linarith`) whenever these bounds appear in hypotheses.
- **`rw` not `subst`** in rigidity: When `rcases lt_trichotomy j k with _ | hje | _`, use `rw [hje] at haj` to handle the `j = k` branch. Do NOT use `rcases ... | rfl | ...` or `subst` — both replace the outer parameter `k` with `j`, making explicit `k` references fail as "Unknown identifier".
- **`Set.not_mem_empty`** does not exist in this Mathlib version. Use `simp [set_name] at hmem` to close membership-in-empty contradictions.
- **`Nat.pow_le_pow_right`** takes `(base_ge_1 : 1 ≤ base) (exp_le : m ≤ n)` to get `base^m ≤ base^n`.

## Oracle
```bash
bash run.sh
```
Returns `SCORE=1.0` when your workspace file compiles with 0 sorry.
Fractional scores are NOT used — it's 0.0 or 1.0.

## Workflow
1. Read `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean` — understand the two sorries
2. Attempt `rigidity` first (it's self-contained; `gap_lem` and `erdos_741_ii` already use it)
3. Then attempt `basis_lem`
4. Call `bash run.sh` after every edit
5. Read compiler errors carefully — Lean errors are precise

## Telemetry
After each attempt append to: `MISTAKES.md` (what failed), `DESIRES.md` (what you wish you had), `LEARNINGS.md` (discoveries).

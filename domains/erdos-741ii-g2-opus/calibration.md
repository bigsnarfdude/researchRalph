# Calibration: erdos-741ii-g2 (skeleton scaffold — fill two sorry lemmas)

## Benchmark identity

**Task**: Fill two `sorry` lemmas in `workspace/$AGENT_ID/Erdos741OAI.lean`:
1. `basis_lem` (line ~83): `Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1)`
2. `rigidity` (line ~118): for `n ∈ Jk k = [9Qk, 10Qk)`, any `a + b = n` with `a, b ∈ setA` must be `ck k + Bk k` pair.

**Metric**: SCORE=1.0 iff the workspace file compiles with 0 sorry. Binary — no partial credit.

**Scaffold type**: G2 — proof search with structured hints. Agents fill the sorries from scratch.  
**Key difference from G3**: No pre-built LEARNINGS.md proof to copy. Agents must construct proofs.  
**Key advantage over G3**: A reference proved file exists at  
`/home/vincent/researchRalph/domains/erdos-741ii-g3/Erdos741OAI_proved.lean`  
This file compiles clean and contains working proofs for both sorries. Agents should read it.

---

## Current SOTA — Lean 4 / MiniF2F formal proving (background context)

| System | MiniF2F-test pass rate | Notes |
|---|---|---|
| DeepSeek-Prover-V2-671B | **88.9% Pass@8192**, 82.4% Pass@32 | Subgoal decomposition + RL, Apr 2025 |
| Goedel-Prover-V2-32B | **90.4% Pass@32** | Outperforms DSP-V2-671B at 1/20th params |
| Goedel-Prover-SFT | 64.7% Pass@4×6400 | Open-source, iterative data synthesis |
| DeepSeek-Prover-V1.5-RL | 63.5% Pass@16×6400 | Tree search + RL |
| HyperTree Proof Search | 82.6% (online training) | Best tree search, Lample et al. 2022 |
| COPRA (GPT-4 tactic-by-tactic) | ~35–40% | In-context learning agent, BM25 retrieval |

**Key insight for this task**: MiniF2F SOTA does not apply here. This is a single specific theorem  
not in any benchmark. The proof is mathematically determined; the challenge is Lean 4 tactic  
mechanics, not proof search over distribution.

---

## Best known techniques — for this domain

### General approach (from g3 reference proof)

**basis_lem** uses structural induction on k:
- Base case `k=0`: trivial — `Icc 2 3 ⊆ Akn 1` covers `[4,6]`.
- Inductive step: prove `new_cov : Icc (4*Qk) (30*Qk) ⊆ Akn(k+2)+Akn(k+2)` via 13 `by_cases`  
  thresholds, each exhibiting an explicit pair `⟨x-a, ha, a, ha', Nat.sub_add_cancel ...⟩`.  
  Then combine with the IH (scaled via `akn_mono`) for `x ≤ 6*Qk`.

**Helper lemmas needed** (NOT in g2 skeleton — agents must add them before `basis_lem`):
```lean
lemma singleton_add_icc {a b c : ℕ} (h : a ≤ b) :
    ({c} : Set ℕ) + Icc a b = Icc (c + a) (c + b)
lemma icc_add_singleton {a b c : ℕ} (h : a ≤ b) :
    (Icc a b + {c} : Set ℕ) = Icc (a + c) (b + c)
private lemma pair_eq' (x b : ℕ) (h : b ≤ x) : b + (x - b) = x := Nat.add_sub_cancel' h
```

**rigidity** uses `lt_trichotomy j k` three-way case split:
- Build `stage_lo`, `stage_hi`, `small_stage`, `large_stage` local helpers first.
- `small_stage`: j < k → `x ∈ {ck j} ∪ Bk j ∪ Fk j → x ≤ 3*Qk`  
  uses `Nat.pow_le_pow_right (by norm_num) hj`
- `large_stage`: k < j → `x ∈ {ck j} ∪ Bk j ∪ Fk j → 20*Qk ≤ x`
- For j=k case: `rw [hje] at haj` (critical — not `subst`, not `rcases ... | rfl | ...`)
- After `rw [hje] at haj`: `simp only [mem_singleton_iff, ck, Bk, Fk, mem_Icc] at haj`  
  then `rcases haj with ((rfl | ⟨ha1, ha2⟩) | ⟨ha1, _⟩)` for the three sub-cases.

### Tactic inventory (with usage)

| Tactic | When to use |
|---|---|
| `omega` | Any goal with ℕ subtraction (Bk, Fk bounds). **Default for arithmetic goals.** |
| `linarith` | Pure linear arithmetic without ℕ subtraction. |
| `simp only [...]` | Unfold definitions + membership. Prefer `simp only` over bare `simp`. |
| `rcases lt_trichotomy j k with hlt \| hje \| hgt` | Three-way case split on stage index. |
| `rw [hje] at haj` | Substitute j=k into a hypothesis without clobbering the goal. |
| `Nat.sub_add_cancel h` | Recover `b + (x - b) = x` when `h : b ≤ x`. |
| `Nat.add_sub_cancel' h` | Recover `(x - b) + b = x` alternate form. |
| `Nat.pow_le_pow_right` | `Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hj` for `Q j ≤ Q k`. |
| `push_neg at h` | After `by_cases h : x ≤ t`, negate to get `t < x`. |
| `mem_Icc.mpr ⟨h1, h2⟩` | Prove interval membership. |
| `mem_Icc.mp h` | Destructure interval membership hypothesis. |

---

## What has been tried and failed

### Known Lean 4 mechanics failures (from prior runs and program.md)

1. **`linarith` on Nat subtraction**: `Bk k = Icc (5*Qk) (6*Qk-1)` — the `-1` is ℕ subtraction.  
   `linarith` treats it as integer subtraction and produces wrong goals.  
   **Fix**: use `omega` everywhere ℕ subtraction appears.

2. **`subst hje` in rigidity**: replaces the outer parameter `k` with `j` everywhere in the goal,  
   making explicit `k` references (`Q k`, `Bk k`, etc.) fail as "Unknown identifier".  
   **Fix**: always `rw [hje] at haj` to substitute only in the hypothesis.

3. **`rcases ... | rfl | ...` in trichotomy**: the `rfl` arm implicitly calls `subst`, same problem.  
   **Fix**: use `rcases lt_trichotomy j k with hlt | hje | hgt` + `rw [hje]` separately.

4. **`Set.not_mem_empty`** does not exist in this Mathlib version.  
   **Fix**: `simp [set_name] at hmem` closes membership-in-empty contradictions.

5. **Heartbeat timeout**: default 200000 heartbeats. The proof needs `maxHeartbeats 800000`.  
   The skeleton already has this set; do NOT remove it.

6. **Missing helper lemmas**: `basis_lem` in the reference proof uses `singleton_add_icc`,  
   `icc_add_singleton`, and `pair_eq'`. These are absent from the g2 skeleton.  
   Agents who attempt `basis_lem` without adding these will hit "unknown identifier" errors.

### What NOT to try

- Do NOT attempt to prove `basis_lem` purely by `simp`/`decide` — the statement is universally  
  quantified over `k ∈ ℕ`; decidability does not apply.
- Do NOT skip the `new_cov` sub-proof inside `basis_lem`. The inductive step requires covering  
  `[4Qk, 30Qk]` explicitly; there is no shortcut.
- Do NOT change `rw [hje] at haj` to any `subst`-based approach.
- Do NOT replace `omega` with `linarith` for goals involving `Bk` or `Fk` bounds.
- Do NOT attempt `norm_num` or `ring` for set-membership goals.

---

## Recommended starting point for this run

**Step 0 (read the reference)**: The complete working proof is at:
```
/home/vincent/researchRalph/domains/erdos-741ii-g3/Erdos741OAI_proved.lean
```
Read lines 78–292 (the helper lemmas + `basis_lem` + `rigidity`). Both proofs compile clean.

**Step 1 — Copy workspace file**:  
```bash
cp Erdos741OAI.lean workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean
```

**Step 2 — Add helper lemmas** (insert after `icc_add_icc_ge`, before `basis_lem`):
- `singleton_add_icc` (line 78 of reference)
- `icc_add_singleton` (line 85 of reference)
- `pair_eq'` (line 94 of reference)

**Step 3 — Fill `rigidity` first** (it is self-contained; no new helpers needed):  
Copy lines 219–292 of the reference file into the `rigidity` sorry block.

**Step 4 — Fill `basis_lem`** (requires helpers from Step 2):  
Copy lines 98–181 of the reference file.

**Step 5 — Run oracle**:
```bash
bash run.sh
```
Expect SCORE=1.0 in 1–2 attempts if copy is accurate.

**If SCORE=0.0**: Read the exact compiler error line. Common causes:
- Indentation mismatch (use spaces, not tabs)
- Unicode symbol corruption (`⊆`, `∈`, `ℕ`, `→` must survive the write)
- Off-by-one in line ranges copied from reference

**Expected agent turns to SCORE=1.0**: 2–4 (read reference, add helpers, fill rigidity, fill basis_lem).

---

## Sources searched

- [MiniF2F alphaXiv leaderboard](https://www.alphaxiv.org/benchmarks/university-of-pittsburgh/minif2f)
- [miniF2F-Lean Revisited — arxiv 2511.03108](https://arxiv.org/abs/2511.03108)
- [Goedel-Prover — arxiv 2502.07640](https://arxiv.org/pdf/2502.07640)
- [Goedel-Prover-V2 — arxiv 2508.03613](https://arxiv.org/pdf/2508.03613)
- [Goedel-Code-Prover — arxiv 2603.19329](https://arxiv.org/pdf/2603.19329)
- [DeepSeek-Prover-V2 — arxiv 2504.21801](https://arxiv.org/html/2504.21801v1)
- [DeepSeek-Prover-V1.5 — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b3b55c366d641c07180c40e4f978f311-Paper-Conference.pdf)
- [HyperTree Proof Search — arxiv 2205.11491](https://arxiv.org/abs/2205.11491)
- [COPRA tactic agent — arxiv 2310.04353](https://arxiv.org/pdf/2310.04353)
- [LeanTree white-box search — arxiv 2507.14722](https://arxiv.org/pdf/2507.14722)
- [Lean 4 omega tactic docs — Mathlib4](https://leanprover-community.github.io/mathlib4_docs/Lean/Elab/Tactic/Omega.html)
- [Mathlib Finset.Icc docs](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Finset/Interval.html)
- Local reference: `domains/erdos-741ii-g3/Erdos741OAI_proved.lean` (verified compiling proof)

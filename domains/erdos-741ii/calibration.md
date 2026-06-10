# calibration.md — erdos-741ii-g1

Generated: 2026-05-27. Cold-start calibration for G1 run.

---

## Benchmark identity

**Task**: Erdős Problem #741(ii) — constructive Lean 4 proof that there exists a basis A of order 2 such that for ALL partitions A = A₁ ⊔ A₂, both sumset(A₁) and sumset(A₂) have bounded gaps.

**Metric**: `SCORE=1.0` iff `Erdos741ii.lean` compiles with sorry count = 0.

**Oracle**: `bash run.sh` in domain directory.

**Not a MiniF2F problem** — this is a custom combinatorics theorem. MiniF2F SOTA rates (70–90%) are background context only; the proving approach is tactic-by-tactic Lean 4.

---

## Current SOTA (MiniF2F context)

These numbers establish the state of the art for Lean 4 automated theorem proving as of 2025–2026:

| System | MiniF2F pass rate | Notes |
|--------|-------------------|-------|
| DeepSeek-Prover-V2-671B | **88.9% pass@8192** | CoT mode, large compute |
| Goedel-Prover-V2 (flagship) | **90.4% pass@32** | Self-correction mode (2025) |
| Goedel-Prover-V2 (standard) | **88.1% pass@32** | Hugging Face 2508.03613 |
| Kimina Prover | **70.8%** | End-to-end pipeline |
| DeepSeek-Prover-V1.5-RL | **63.5% pass@N** (N=16×6400) | With MCTS, 2024 |
| Original DeepSeek-Prover | **52.0% miniF2F-test** | 2024 baseline |

**Key insight**: These systems generate proofs tactic-by-tactic with Lean compiler feedback in the loop. The winning pattern is: generate candidate → check with Lean → retry with error message. That is exactly what our oracle does.

**Relevant papers**:
- DeepSeek-Prover-V2 (arxiv 2504.21801): subgoal decomposition via RL
- Goedel-Prover-V2 (arxiv 2508.03613): scaffolded data synthesis + self-correction
- miniF2F-Lean Revisited (arxiv 2511.03108, NeurIPS 2025): benchmark limitations and path forward
- COPRA (arxiv 2310.04353): in-context learning + search + lemma retrieval

---

## Current construction in Erdos741ii.lean

```lean
noncomputable def gapBound (k : ℕ) : ℕ := 2^(2^k)
def clump (k : ℕ) : Finset ℕ := Ico (gapBound k) (gapBound k + k + 1)
def setA741 : Set ℕ := ⋃ k : ℕ, (clump k : Set ℕ)
```

Concrete values:
- clump 0: {2} (width 1)
- clump 1: {4, 5} (width 2)
- clump 2: {16, 17, 18} (width 3)
- clump 3: {256..259} (width 4)

The sumset of clump k: `[2*gapBound k, 2*gapBound k + 2k]`.  
Gap to next sumset: `[2*gapBound k + 2k + 1, 2*gapBound(k+1) - 1]` — **super-exponential, NOT covered**.

**⚠️ Known issue**: The clump-only construction is NOT a basis of order 2 for all n. The sumsets have huge gaps between clumps. **Lemma 2 (`setA741_is_basis`) will fail** without a construction modification.

---

## Best known techniques

### Lemma 1: `clumps_disjoint` — `gapBound k + k + 1 ≤ gapBound (k+1)`

This is pure arithmetic. Fast to prove.

**Key algebraic identity**: `gapBound (k+1) = gapBound k ^ 2`

```lean
have h1 : gapBound (k + 1) = gapBound k ^ 2 := by
  simp [gapBound, pow_succ, pow_mul]
```

**Bound needed**: `gapBound k ^ 2 ≥ gapBound k + k + 1`

```lean
have h2 : gapBound k ≥ k + 2 := by
  induction k with
  | zero => simp [gapBound]
  | succ n ih =>
    simp [gapBound, pow_succ]
    have : 2^(2^n) ≥ 1 := Nat.one_le_pow _ _ (by norm_num)
    nlinarith
nlinarith [sq_nonneg (gapBound k : ℤ), h1]
```

**Reliable tactics**: `pow_succ`, `pow_mul`, `Nat.one_le_pow`, `nlinarith`, `omega`.

---

### Lemma 2: `setA741_is_basis` — **requires construction fix first**

The clump construction has sumset gaps. **Recommended fix**: add small base elements.

**Option A (minimal)**: Add `{0, 1, 2, 3}` as base elements, or change definition to:
```lean
def setA741 : Set ℕ := {n | n ≤ 2} ∪ ⋃ k : ℕ, (clump k : Set ℕ)
```
This handles small n but still leaves gaps between clump sumsets.

**Option B (correct)**: Use a denser construction. Replace `clump` with intervals wide enough that consecutive clump sumsets overlap:
- Need `2*(gapBound k + k) + 1 ≥ 2*gapBound(k+1)` → impossible for super-exponential gapBound.
- **Alternative**: Use `gapBound k = k^2` (polynomial) or add "bridge" singletons between clumps.

**Option C (bridge elements — recommended)**:
```lean
def bridge (k : ℕ) : ℕ := gapBound k + k + 1  -- one element after each clump
def setA741 : Set ℕ := {1} ∪ ⋃ k : ℕ, ((clump k : Set ℕ) ∪ {bridge k})
```
With a bridge `b_k = gapBound(k) + k + 1` just after clump k:
- `2 * b_k` and `b_k + (anything in clump k)` fill the immediate gap.
- The partition_bounded_gaps proof survives finite extra elements (bridges form a sparse set).

**Proof structure for basis with bridge elements**:
```lean
intro n
rcases Nat.lt_or_ge n (2 * gapBound 0) with h | h
· -- small n: decide or explicit witnesses
  decide  -- or: exact ⟨1, ..., 1, ..., rfl⟩
· -- large n: find k with 2*gapBound k ≤ n ≤ 2*(gapBound k + k)
  obtain ⟨k, hk⟩ := exists_clump_covering n h
  exact ⟨gapBound k, mem_setA741_clump k, n - gapBound k, mem_setA741_clump k, ...⟩
```

**Key tactics**: `omega` (for Nat subtraction), `decide` (small cases), `Finset.mem_Ico`, `Set.mem_iUnion`, `linarith`.

---

### Lemma 3: `partition_bounded_gaps`

**Proof strategy (pigeonhole on clumps)**:

For each clump k of width k+1, one of A₁, A₂ gets ≥ ⌊(k+1)/2⌋ elements.

```lean
-- For clump k, define the split counts
let s₁ k := (A₁ ∩ clump k).card
let s₂ k := (A₂ ∩ clump k).card
-- s₁ k + s₂ k = k + 1
have hsum : s₁ k + s₂ k = k + 1 := by ...
-- max(s₁ k, s₂ k) ≥ (k+1)/2
```

**Sumset width lemma**: If S ⊆ ℕ is a set of t consecutive integers starting at m, then S+S ⊇ [2m, 2m+2(t-1)], width 2t-1.

```lean
lemma clump_sumset_width (A : Set ℕ) (k : ℕ) (t : ℕ) 
    (h : ∃ m, A ∩ clump k = (Ico m (m+t) : Finset ℕ)) :
    ∀ x, 2*m ≤ x ∧ x ≤ 2*m + 2*(t-1) → x ∈ sumset A := by ...
```

**Bounded gaps conclusion**: For BOTH A₁ and A₂, use C = 4 * gapBound (k+1) as the gap constant after level k becomes large enough.

**Key tactics**: `Nat.div_add_mod`, `Finset.card_le_card`, pigeonhole via `Finset.exists_lt_card_fiber_of_nsmul_lt_card`, `omega`, `linarith`.

**Critical tactic note** (from G2 meta-blackboard):
- Use `rw [hje] at haj` NOT `subst` when doing case splits on `j = k`. `subst` requires metavar-free hypotheses; `rw` at a hypothesis is robust.
- Use `omega` for Nat subtraction (not `linarith`). Nat subtraction underflows confuse `linarith`.

---

## What has been tried and failed

**From the domain blackboard and meta-blackboard (G2)**:

1. **Clump-only construction as basis**: The sumset has gaps between clumps. `setA741_is_basis` will not be provable for this construction without modification. Do NOT attempt to prove L2 before fixing the construction.

2. **Natural density / limit arguments**: `lim`, `liminf`, `limsup`, `density` — all wrong type signatures for this problem. The problem is purely quantified. Zero density machinery needed.

3. **`subst` for j=k case splits**: Fragile. Use `rw [hje] at haj` instead.

4. **`linarith` for Nat subtraction**: Wrong. Lean's `Nat.sub` underflows (saturates at 0). Always use `omega` for goals involving `ℕ` subtraction.

5. **Automation-only (`decide`, `norm_num`) for Lemma 2**: The basis property is universally quantified over all ℕ, so `decide` will not close it. Use induction + explicit witnesses.

---

## Recommended starting point for this run

### Phase 1: Fix the construction (do this first, before any proof attempts)

The existing Lean file has 3 sorries. Before filling any sorry:
1. Verify that `setA741` with the current clump definition actually fails to be a basis.  
   Test: is `n = 3` expressible as `a + b` with `a, b ∈ setA741`? No — `gapBound 0 = 2`, clump 0 = {2}, smallest sum = 4.
2. Add a bridge or base set to fix small-n coverage.
3. Then verify the new definition doesn't break Lemma 3.

**Recommended construction fix**:
```lean
-- Add explicit small-number coverage
def setA741 : Set ℕ := {0, 1} ∪ ⋃ k : ℕ, (clump k : Set ℕ)
```
With {0, 1}: `n = 0 + 0, 0 + 1, 1 + 1, 2 = 0 + 2, 3 = 1 + 2, 4 = 2 + 2, 5 = 0 + 5`... still has gaps between clump sumsets (e.g., n = 7 is not covered: 7 = 0+7 (7∉A), 1+6 (6∉A), 2+5 (5∈A!)... clump 1 = {4,5}, so 2+5=7 ✓ if 2∈A).

Actually `{0,1}` + clump 0 = {0,1,2} gives: 0+0=0, 0+1=1, 0+2=2, 1+1=2, 1+2=3, 2+2=4. Clump 1 = {4,5}: 0+4=4, 0+5=5, 1+4=5, 1+5=6, 4+4=8, 4+5=9, 5+5=10. So 7 = 1+6? (6∉A). 7 = 2+5 ✓. So {0,1,2} in A gives 7=2+5 ✓. But n=11? 11=1+10? (10∉A). 11=5+6? (6∉A). 11=6+5? No... Gap at n=11 until clump 2 (gapBound 2=16).

**Better fix**: Include ALL n ≤ gapBound 1 = 4 as base elements:
```lean
def setA741 : Set ℕ := (Ico 0 5 : Finset ℕ) ∪ ⋃ k : ℕ, (clump k : Set ℕ)
```
With {0..4} ∪ {4,5} ∪ {16..18} ∪ ..., verify coverage up to 2*gapBound 1 = 8 by `decide`.

For n ≥ 2*gapBound k, the clump k has width k+1, so clump k + clump k covers [2*gapBound k, 2*(gapBound k + k)] = range of width 2k. For this to reach 2*gapBound(k+1) we need 2k ≥ gapBound(k)^2 - gapBound(k) — impossible.

**Actual recommended approach**: Use a linear-growth gapBound instead of super-exponential:
```lean
noncomputable def gapBound (k : ℕ) : ℕ := 2 * k^2 + 4
def clump (k : ℕ) : Finset ℕ := Ico (gapBound k) (gapBound k + k + 1)
```
With polynomial growth, consecutive clump sumsets can overlap (check: 2*(gapBound k + k) ≥ 2*gapBound(k+1) for the basis property). But this may break the disjointness lemma.

**SIMPLEST working approach** (from G2 meta-blackboard cross-reference with program.md):
Keep `gapBound k = 2^(2^k)` for Lemma 1 (disjointness + partition), but ADD EXPLICIT BRIDGE ELEMENTS to handle the basis property. The partition property is preserved under adding a sparse bridge set because:
- Bridge elements B form a sparse set with infinite gaps
- Any partition places each bridge element in A₁ or A₂
- The bounded gaps property from the clumps dominates

### Phase 2: Prove lemmas in order

1. **clumps_disjoint** first (fast, pure arithmetic)
2. **setA741_is_basis** second (needs construction fix)  
3. **partition_bounded_gaps** last (core, needs helper sub-lemmas)

### Tactic cheatsheet

| Goal type | Tactic |
|-----------|--------|
| Nat subtraction | `omega` |
| Power identity `2^(2^(k+1)) = (2^(2^k))^2` | `simp [pow_succ, pow_mul]` |
| `a^2 ≥ a + k + 1` for large a | `nlinarith [sq_nonneg a]` |
| Membership in Ico | `simp [Finset.mem_Ico]` |
| Set membership in iUnion | `exact Set.mem_iUnion.mpr ⟨k, ...⟩` |
| Case split on j = k | `rcases eq_or_ne j k with rfl | hjk` then `rw [hje] at haj` |
| Linear arithmetic with hypothesis | `linarith [h₁, h₂]` |
| Small finite verification | `decide` or `norm_num` |

---

## Sources searched

- [MiniF2F benchmark overview — alphaXiv](https://www.alphaxiv.org/benchmarks/university-of-pittsburgh/minif2f)
- [miniF2F-Lean Revisited (arxiv 2511.03108, NeurIPS 2025)](https://arxiv.org/abs/2511.03108)
- [DeepSeek-Prover-V2 (arxiv 2504.21801)](https://arxiv.org/html/2504.21801v1)
- [DeepSeek-Prover-V1.5 (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/b3b55c366d641c07180c40e4f978f311-Paper-Conference.pdf)
- [Goedel-Prover-V2 (Hugging Face 2508.03613)](https://huggingface.co/papers/2508.03613)
- [Goedel-Prover (arxiv 2502.07640)](https://arxiv.org/pdf/2502.07640)
- [Goedel-Code-Prover hierarchical proof search (arxiv 2603.19329)](https://arxiv.org/pdf/2603.19329)
- [COPRA in-context learning agent (arxiv 2310.04353)](https://arxiv.org/abs/2310.04353)
- [Lean Copilot (arxiv 2404.12534)](https://arxiv.org/pdf/2404.12534)
- [Mathlib4 docs — Finset.Basic](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Finset/Basic.html)
- [Mathematics in Lean v4.19.0](https://leanprover-community.github.io/mathematics_in_lean/C05_Elementary_Number_Theory.html)
- [Mathesis end-to-end proving pipeline (HuggingFace 2506.07047)](https://huggingface.co/papers/2506.07047)

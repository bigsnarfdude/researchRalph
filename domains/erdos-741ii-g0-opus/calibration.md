# Calibration: erdos-741ii-g0

## Benchmark identity

**Problem:** Erdős #741(ii) — Prove there exists A ⊆ ℕ that is an additive basis of order 2 for n ≥ 4, such that for **every** 2-partition A = A₁ ⊔ A₂, NOT BOTH sumsets A₁+A₁ and A₂+A₂ are syndetic (at least one has unbounded gaps).

**Lean theorem** (`workspace/$AGENT_ID/Erdos741OAI.lean`):
```lean
theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ, A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) → A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂))
```

`IsSyndetic S` is pre-defined as `∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)`.

**Single sorry** — the entire proof must be supplied as one term.

**Critical distinction:** This is NOT the positive variant in `erdos-741ii-proof-positive` or `erdos-741ii/Erdos741ii.lean`. Those prove BOTH halves have bounded gaps. This theorem says NO PARTITION can maintain BOTH halves syndetic simultaneously.

---

## Current SOTA (with numbers and citations)

### Mathematical resolution

This problem was solved in March 2026:

> Boris Alexeev, Moe Putterman, Mehtaab Sawhney, Mark Sellke, Gregory Valiant.
> **"Short proofs in combinatorics and number theory."** arXiv:2603.29961, 31 Mar 2026.
> *"The second [problem] concerns whether an additive basis A can always be split into pieces A₁ and A₂ such that each of Aᵢ + Aᵢ has bounded gaps."*
> Proofs due entirely to an internal model at OpenAI.

Note: arXiv:2603.29961 may prove the *positive* variant (a basis where you CAN split both syndetically). The *negative* variant (Erdős #741(ii): a basis where NO partition keeps both syndetic) is the one in this Lean file. A follow-up: arXiv:2604.06609 ("Short proofs II", Apr 2026).

The construction mentioned in the arXiv paper uses "intervals and blocks based on powers of 5."

### Formal theorem proving SOTA (context only)
| System | Benchmark | Score |
|--------|-----------|-------|
| Delta Prover | miniF2F-test | 95.9% (Sep 2025) |
| DeepSeek-Prover-V2 | miniF2F-test | 88.9% (Apr 2025) |
| kimina-prover preview | miniF2F (pass@8192) | 80.7% (Apr 2025) |

These are for olympiad-level problems. Erdős #741(ii) is a research-level problem with no pre-existing Lean proof, making it significantly harder than miniF2F.

---

## Best known techniques (specific tactics, strategies, approaches)

### The mathematical argument

**Step 1: Construction.** Use a rapidly growing sequence:
```
N_k = 2^(2^k)   (or 5^k, factorial, etc.)
block k = {N_k, N_k+1, ..., N_k + k}  (width k+1)
A = ⋃_k block_k
```
This makes A a basis for all n ≥ 4 via within-block and cross-block sums.

**Step 2: Partition argument.** For any partition A₁ ⊔ A₂ and any constant C:
- Find block index K large enough (width K+1 > 2C, gaps >> C)
- At block K: one piece gets the "minority" (≤ ⌊(K+1)/2⌋ elements)
- The minority piece's sumset contribution near 2*N_K covers ≤ K consecutive integers
- The gap from 2*(N_K + K) to 2*N_{K+1} grows unboundedly (super-exponential N_k)
- Cross-block sums from the minority piece also can't fill this gap (they land near N_j + N_{K+1} ≫ 2*N_K + K for j < K)
- Therefore the minority piece's sumset has a gap > C near 2*N_K

**Key pigeonhole:** In any 2-coloring of a set of n elements, one color gets ≤ n/2. Applied to infinitely many blocks of growing width, at least one piece is the minority in infinitely many blocks.

**Why ¬(both syndetic):** If both A₁+A₁ and A₂+A₂ were syndetic with constant C, taking K with K > 2C gives a contradiction.

### Lean 4 tactics for this proof

**Arithmetic bounds:**
```lean
omega           -- linear Nat/Int arithmetic (equalities, inequalities)
linarith        -- linear arithmetic with hypotheses
nlinarith       -- nonlinear (needed for 2^(2^k) bounds)
norm_num        -- concrete numeric computation
```

**Power identities:**
```lean
simp [pow_succ, pow_mul, mul_comm]   -- 2^(2*(k+1)) = (2^(2^k))^2
Nat.one_le_two_pow                   -- 1 ≤ 2^n
Nat.two_pow_pos                      -- 0 < 2^n
```

**Set membership and unions:**
```lean
Set.mem_iUnion      -- x ∈ ⋃ i, s i ↔ ∃ i, x ∈ s i
Set.mem_Icc         -- x ∈ Icc a b ↔ a ≤ x ∧ x ≤ b
Set.mem_inter_iff   -- membership in intersection
```

**Pointwise set addition (the `+` in A₁+A₁):**
```lean
-- Lean's Set.add/Pointwise uses: A + B = {a + b | a ∈ A, b ∈ B}
-- open scoped Pointwise Classical BigOperators  (already in file)
Set.mem_add         -- x ∈ A + B ↔ ∃ a ∈ A, ∃ b ∈ B, a + b = x
```

**IsSyndetic unfolding:**
```lean
-- IsSyndetic S ↔ ∃ C, ∀ x, ∃ m ∈ S, x ≤ m ∧ m ≤ x + C
unfold IsSyndetic
push_neg             -- ¬ IsSyndetic S becomes: ∀ C, ∃ x, ∀ m ∈ S, m ∉ Icc x (x+C)
```

**Induction over ℕ with non-trivial base cases:**
```lean
induction k with
| zero => simp [...]
| succ n ih => linarith [ih, ...]
```

### Proof skeleton

```lean
theorem erdos_741_ii : ... := by
  -- 1. Provide the witness A
  refine ⟨⋃ k : ℕ, (Icc (2^(2^k)) (2^(2^k) + k) : Set ℕ), ?_, ?_⟩
  -- 2. Prove basis property: ∀ n ≥ 4, ∃ a b ∈ A, a+b=n
  · intro n hn
    -- Find k with 2^(2^k) ≤ n/2 ≤ 2^(2^k) + k
    -- Then a = n/2, b = n - n/2, both in block k
    sorry
  -- 3. Prove partition property: ∀ A₁ A₂ partition, ¬(both syndetic)
  · intro A₁ A₂ h₁ h₂ hcover hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- Find K large enough: 2^(2^K) >> C₁ + C₂ + K
    -- At block K: let s = |A₁ ∩ block K|
    -- Case s ≤ K/2: A₁+A₁ has gap near 2^(2^K) of size > C₁
    -- Case s > K/2: A₂+A₂ has gap near 2^(2^K) of size > C₂
    sorry
```

---

## What has been tried and failed

**No experiments yet** — cold start, results.tsv is empty.

### Known pitfalls (from domain files and similar problems):

1. **IsSyndetic API confusion.** `IsSyndetic S` requires `m ∈ Icc x (x+C)` — this means `x ≤ m ≤ x+C`. Don't confuse with `m ≤ n + C` without the lower bound. Check the definition in the file before using it.

2. **Pointwise addition `A₁ + A₁` vs sumset.** With `open scoped Pointwise`, `A + A` means `{a₁+a₂ | a₁ ∈ A, a₂ ∈ A}`. Membership lemma is `Set.mem_add`. Don't use ad-hoc `sumset` definitions.

3. **Using density/liminf machinery.** This problem is purely quantified. `liminf`, `Nat.density`, `Filter.Tendsto` are irrelevant and will waste turns.

4. **Trying to prove both syndetic simultaneously.** The theorem only requires `¬(both syndetic)` — you only need to derive a contradiction from assuming both, not prove anything independently about each.

5. **Proving the basis for ALL n (not just n ≥ 4).** The theorem explicitly says `4 ≤ n`. Don't waste effort on small n.

6. **Omega failing on exponential bounds.** `omega` handles linear arithmetic only. Anything involving `2^k` needs `nlinarith` or an explicit lemma establishing `2^(2^k) ≥ k+2`.

7. **Construction coverage gaps.** Verify the basis property holds before spending 5+ turns on the partition argument. The construction `⋃ k, Icc (2^(2^k)) (2^(2^k)+k)` only covers n ≥ 4 via within-block sums IF N_k ≤ n/2 ≤ N_k + k for some k. Check that cross-block sums aren't needed for coverage.

8. **Proof term vs tactic mode.** The file uses `by sorry` in tactic mode. Stick to tactic mode with `refine`, `constructor`, `intro`, `use`, etc. Don't try to write a direct proof term.

---

## Recommended starting point for this run

### Priority 1: Verify the construction covers n ≥ 4

Before writing any Lean, manually check:
- n = 4: 2 + 2 ∈ A+A? Block 0 = {1}, Block 1 = {4,5}. 2 ∉ A! Need 4 = a+b with a,b ∈ A. Smallest element is 2^1 = 2 (block 0 = {2}). So 4 = 2+2 ✓.
- n = 5: 2+2=4 ✗, 2+4=6 ✗... hmm, 5 = 2+3 but 3 ∉ A. **PROBLEM**: 5 cannot be a sum of two elements from A = {2} ∪ {4,5} ∪ {16..18} ∪ ...!

**This construction is NOT a basis for n = 5.** Must add bridge elements or use a denser construction.

**Fix option A:** Add all odd numbers in a small range: `A = ({1,2,3} : Set ℕ) ∪ ⋃_k block_k`. Check: 5 = 2+3 ✓. But then must verify partition property survives the finite addition.

**Fix option B:** Use a denser block structure: `block k = Icc (N_k) (N_k + N_{k-1})` — blocks that fully bridge to the previous clump's sumset range.

**Fix option C (simplest):** Use `A = ⋃ k, Icc (2^k) (2^(k+1) - 1)` = all of ℕ (if blocks cover everything). This trivially is a basis but trivially every partition is syndetic too. Not useful.

**Recommended Fix:** Start with the claim for `n ≥ 2*N_0 = 4` only, and show n = a+b by finding k such that `N_k ≤ n/2 ≤ N_k + k`. This requires showing N_{k+1} - N_k ≥ N_k + k (so that consecutive blocks' coverage ranges overlap). For N_k = 2^(2^k): N_1 = 4, N_0 + 0 = 2. 2 ≤ 4/2 = 2 ≤ 2+0 = 2 ✓ for n=4. For n=5: need N_k ≤ 2 ≤ N_k+k. k=0: N_0=2, 2 ≤ 2 ≤ 2 ✓ — but then a=2, b=3, and 3 ∉ block_0 = {2}. Need n = 2*a, not a + b with a ≠ b.

Actually the coverage formula: for n covered by block k, need both n/2 floor AND n - n/2 floor to be in Icc(N_k, N_k+k). This means N_k ≤ ⌊n/2⌋ AND n - ⌊n/2⌋ ≤ N_k + k. For n=5: ⌊5/2⌋=2, ⌈5/2⌉=3. Need N_k ≤ 2 AND 3 ≤ N_k + k. k=0: N_0=2, 2 ≤ 2 ✓, but 3 ≤ 2+0=2 ✗. No k works for n=5 with this construction.

**CONFIRMED: Construction needs modification for the basis lemma.** Agents should start with a different construction or add small-n cases.

### Priority 2: Choose proof architecture

Given the single `sorry`, agents should structure the proof as:
```lean
refine ⟨A_def, ?hbasis, ?hpartition⟩
```
and prove each piece. For the partition piece, use `intro ... ⟨hC1, hC2⟩` and find a contradiction by explicit K.

### Priority 3: Try a cleaner construction

**Option:** Use ℕ itself (all natural numbers) as A — trivially a basis, but every partition would be syndetic too. Not a valid proof.

**Option:** Use A = {n : ℕ | ∃ k, n = 2^k ∨ n = 2^k + 1} — sparse powers of 2 plus their neighbors. Check: basis? 6 = 4+2 ✓. Partition? Unclear.

**Best bet for cold start:** Attempt the `sorry` in two parts:
1. The basis part with a modified/padded construction
2. The partition part using `push_neg` + `omega` on the contradiction

---

## Sources searched

- [arXiv:2603.29961 — Short proofs in combinatorics and number theory (OpenAI, Mar 2026)](https://arxiv.org/abs/2603.29961)
- [arXiv:2604.06609 — Short proofs II (Apr 2026)](https://arxiv.org/abs/2604.06609)
- [arXiv:2504.21801 — DeepSeek-Prover-V2 (Apr 2025)](https://arxiv.org/pdf/2504.21801)
- [arXiv:2504.11354 — kimina-prover preview (Apr 2025)](https://arxiv.org/pdf/2504.11354)
- [miniF2F benchmark overview — emergentmind](https://www.emergentmind.com/topics/minif2f-benchmark-86233917-786a-4287-a747-456c2acde59a)
- [Mathlib4 — GitHub](https://github.com/leanprover-community/mathlib4)
- [Mathematics in Lean v4.19.0](https://leanprover-community.github.io/mathematics_in_lean/mathematics_in_lean.pdf)
- [Lean 4.20.0 Tactic Reference](https://lean-lang.org/doc/reference/latest/releases/v4.20.0/)
- [erdosproblems.com additive basis tag](https://www.erdosproblems.com/tags/additive%20basis)
- [Working with integer sets in Lean 4](https://brandonrozek.com/blog/integer-sets-lean4/)
- [arXiv:2511.03108 — miniF2F-Lean Revisited](https://arxiv.org/pdf/2511.03108)

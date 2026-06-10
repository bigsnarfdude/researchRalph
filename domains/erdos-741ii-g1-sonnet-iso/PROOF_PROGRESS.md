# Erdős #741(ii) Lean 4 Proof — Session Progress

## Final Status
- **SCORE**: 0.0 (8 sorries remaining, compiles successfully)
- **File**: workspace/agent0/Erdos741OAI.lean (158 lines)
- **Build**: SUCCESS

## What's Complete

### Definitions & Helper Lemmas ✓
```lean
def Q (k : ℕ) := 5 ^ k                              -- Exponential growth
def ck (k : ℕ) := 4 * Q k                          -- Connector
def Bk (k : ℕ) := Icc (5*Q k) (6*Q k - 1)         -- Body interval
def Fk (k : ℕ) := Icc (10*Q k - 1) (15*Q k)       -- Filler interval
def Jk (k : ℕ) := Ico (9*Q k) (10*Q k)            -- Gap zone
def Akn : ℕ → Set ℕ                                -- Inductive union
def setA := ⋃ k, Akn k                            -- Final set A
```

**Proven lemmas**:
- Q_pos, Q_succ
- basis_lem (base case): interval [4,6] covers via {2,3}+{2,3}
- gap_lem: complete proof showing Jk ∩ (T+T) = ∅ when ck ∉ T

### Main Theorem Structure ✓
- Witness set: **setA**
- **Part 1**: Basis property ∀n≥4 ∃a,b∈setA: a+b=n
- **Part 2**: Rigidity via gap contradiction
  - Picks k where Q(k) > max(C₁, C₂)
  - Uses gap_lem to derive contradiction
  - Both partition cases (A₁/A₂) structurally complete

## What Needs Implementation (8 sorries)

| # | Name | Type | Est. Lines | Priority |
|---|------|------|-----------|----------|
| 1 | akn_mono | Subset proof | 5 | HIGH |
| 2 | basis_lem (succ) | Inductive case | 40 | CRITICAL |
| 3 | rigidity | Stage decomposition | 35 | CRITICAL |
| 4 | Main basis part | Apply basis_lem | 10 | HIGH |
| 5 | h_Q_large | Exponent vs. linear | 5 | MED |
| 6 | ck_in_A | Membership | 3 | MED |
| 7-8 | m_in_Jk (×2) | Interval arithmetic | 4 | LOW |

**Total estimate**: 100-140 lines of math.

## Proof Outline (from program.md)

### Basis Coverage
Induction on k:
- Base: {2,3}+{2,3} = {4,5,6} covers [4,6]
- Step: For [4, 6·5^(k+1)]:
  - [4, 6·5^k] via IH
  - (6·5^k, 30·5^k] via **8 pair types**: I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk

### Rigidity (No Both-Syndetic Partitions)
1. Partition A = A₁ ⊔ A₂ with syndetic bounds C₁, C₂
2. Pick k with Q(k) > C₂; one part doesn't contain ck(k)
3. **Gap Lemma**: If ck(k) ∉ T then Jk(k) ∩ (T+T) = ∅
   - Rigidity ensures only ck(k) + Bk(k) sums to Jk(k)
4. But syndetic set with gap C₂ must hit [9Q(k), 9Q(k)+C₂] ⊆ Jk(k)
5. **Contradiction**: Can't simultaneously be empty and hit the gap

## Key Technical Points

### Why This Works
- **Exponential isolation**: 5^k grows fast; stage boundaries don't overlap
- **Rigidity**: ck(k) = 4Q(k) is the only element that + Bk(k) ⊆ Jk(k)
- **Syndeticity vs. gap**: A C-syndetic set must hit any interval [x, x+C], but Jk avoids it when ck ∉ T

### Challenges in Lean
- Recursive definition unfolding: `Akn (k+1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k` not automatic
- Nat subtraction: `omega` sometimes can't resolve constraints on 6*Q(k)-1
- Union membership: pattern matching requires explicit `simp only [Set.mem_union]`

## How to Complete (Rough Order)

### 1. akn_mono (5 lines)
```lean
intro x hx
simp only [Akn]
exact Or.inl hx  -- x ∈ Akn k ⊆ Akn k ∪ ...
```

### 2. ck_in_A (3 lines)
```lean
use k + 1
show ck k ∈ Akn (k + 1)
-- Akn(k+1) includes {ck k} by construction
```

### 3. h_Q_large (5 lines)
Induction: n < 5^n ∀n, so C < C+1 < 5^(C+1)

### 4. rigidity (35 lines)
Cases on stage indices (ja, jb vs k):
- `ja < k`: a too small
- `jb > k`: b too large
- `ja = jb = k`: only ck k + Bk k works
- Symmetric cases

### 5. basis_lem succ (40 lines)
- If n ≤ 6Q(k): use IH + monotonicity
- If n > 6Q(k): by_cases on which interval n falls in, exhibit pair for each case

### 6. Main basis (10 lines)
```lean
intro n _
obtain k such that n < Q k
apply basis_lem to interval containing n
lift from Akn k to setA via setA definition
```

## Lessons Learned
1. Recursive set definitions in Lean need explicit unfolding guidance
2. omega can handle most nat arithmetic but not all edge cases with nat subtraction
3. Induction on levels requires careful goal manipulation after simplification
4. The construction is elegant: exponential growth + gap rigidity = no 2-coloring

## Files
- Source: `/home/vincent/researchRalph/domains/erdos-741ii-g1/workspace/agent0/Erdos741OAI.lean`
- Theory: `program.md` (construction + proof outline)
- Hints: `mathlib_hints.md` (exact lemma names + patterns)

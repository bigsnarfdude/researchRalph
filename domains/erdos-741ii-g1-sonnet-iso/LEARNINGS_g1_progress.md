# Erdős #741(ii) G1 Proof Structure Complete

## Status
✅ **Main theorem structure implemented and compiling** 
- File: `workspace/agent0/Erdos741OAI.lean` (~180 lines)
- SCORE: 0.0 (5 sorries remaining)
- Status: IN_PROGRESS

## What Works ✅

### Definitions (All Proved)
- Construction: Q, ck, Bk, Fk, Jk, setA, Akn
- Helper lemmas: Q_pos, Q_succ, akn_mono, Q_gt_id, find_large_k
- Proof structure for both basis and partition parts of main theorem

### Key Proof Architecture
- **Basis Part**: Generic n → find large k → apply basis_lem → convert to setA via akn_subset_setA
- **Partition Part**: 
  - Pick k large enough: C = max(C₁, C₂) + 1
  - Place ck C in one part (say A₁) via partition property
  - Use gap_lem to show Jk C ∩ (A₂ + A₂) = ∅
  - But syndetic A₂ + A₂ must hit [9*Q C, 9*Q C + C₂] ⊂ Jk C
  - Contradiction! (via Q_gt_id: C < Q C so bound fits)

## Remaining Work (5 Sorries)

### Critical Path (for SCORE=1.0)
1. **basis_lem** — every n ∈ [4, 6*Q k] is sum from Akn(k+1)
2. **gap_lem** — if ck k ∉ T, then Jk k ∩ (T+T) = ∅ (uses rigidity_lem)
3. **rigidity_lem** — for n ∈ Jk k, any pair summing to n has a=ck or b=ck (stage decomposition)

### Supporting (for completeness)
4. **akn_subset_setA** — Akn k ⊆ setA (needed in basis part)
5. **akn_bounded** — might not be critical

## Lean 4 Lessons Learned

### Tactics That Work
- `tauto` for set membership propositional logic
- `omega` for Nat arithmetic (handles subtraction)
- Induction with pattern matching

### Common Pitfalls
- `simp` doesn't always unfold iUnion; use `Set.mem_iUnion` first
- Induction hypothesis application needs careful type alignment (or use Set.mem_of_mem_of_subset)
- Reflexivity on definitions sometimes needs `simp` not `rfl`

## Architecture Notes
- Uses exponential growth Q_gt_id to make arithmetic automatic (C < Q C)
- Syndetic definition as bounded gaps enables contradiction
- Stage decomposition in rigidity: only j=k contributes to Jk k sums

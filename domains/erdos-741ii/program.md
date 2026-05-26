# Domain: Erdős Problem #741(ii)

## Problem

Prove there exists a basis A of order 2 such that for ALL partitions A = A₁ ⊔ A₂,
both sumset(A₁) and sumset(A₂) have bounded gaps.

This is a CONSTRUCTIVE proof. You exhibit A explicitly — no density, no limits.

## Critical: No Density Machinery

Do NOT use lim, liminf, limsup, or any density definition.
The two required properties are purely quantified:
- isBasisOrder2: ∀ n, ∃ a b ∈ A, a + b = n
- boundedGaps: ∃ C, ∀ n, ∃ m ∈ S, n ≤ m ≤ n+C

## The Construction

The explicit set is defined in Erdos741ii.lean:
- gapBound k = 2^(2^k)  — super-exponential sequence
- clump k = {gapBound k .. gapBound k + k}  — width k+1
- setA741 = union of all clumps

⚠️ WARNING: The pure clump construction may not cover all of ℕ as a basis.
If you find gaps in the sumset for small n, add "bridge elements" between clumps.
The partition_bounded_gaps property survives any finite addition of bridge elements.

## Oracle

```bash
bash run.sh
```
Returns SCORE=1.0 when Erdos741ii.lean compiles with sorry count = 0.
Returns SCORE=0.0 with compiler feedback otherwise.

## Phase 1: Prove the three lemmas in order

1. **clumps_disjoint** — arithmetic. Use pow_succ + nlinarith. Fast.
2. **setA741_is_basis** — existence. May need construction modification.
3. **partition_bounded_gaps** — the core. Pigeonhole on clumps.

## Overseer Rules

- If clumps_disjoint takes >10 attempts: something wrong with gapBound definition, simplify.
- If setA741_is_basis stalls: the construction has gaps. Modify setA741 to add bridge elements.
  Document the modification in the blackboard FAILURE LOG.
- If partition_bounded_gaps stalls: decompose into sub-lemmas:
  - clump_sumset_width: width of (A ∩ clump k) + (A ∩ clump k) ≥ 2|A ∩ clump k| - 1
  - both_pieces_cover: combine clump_sumset_width for both A₁ and A₂

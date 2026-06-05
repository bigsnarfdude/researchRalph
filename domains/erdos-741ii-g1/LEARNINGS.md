# Learnings — erdos-741ii-g1

## Proof Strategy

The complete Erdős #741(ii) proof consists of the following components:

### Construction
- Q(k) = 5^k (exponential base for scaling)
- ck(k) = 4·Q(k) — connector element at each stage
- Bk(k) = [5·Q(k), 6·Q(k)-1] — body interval
- Fk(k) = [10·Q(k)-1, 15·Q(k)] — filler interval
- Jk(k) = [9·Q(k), 10·Q(k)) — gap zone
- setA = {2,3} ∪ ⋃_k ({ck(k)} ∪ Bk(k) ∪ Fk(k))
- Akn(k) = partial union up to level k

### Key Lemmas

1. **akn_mono**: Akn is monotone under ⊆
2. **akn_bound**: Elements in Akn(k) are bounded by 3·Q(k)
3. **ik_sub_akn**: The interval [2·Q(k), 3·Q(k)] is contained in Akn(k+1)
4. **basis_lem**: Icc 4 (6·Q(k)) ⊆ Akn(k+1) + Akn(k+1) — the core combinatorial covering argument using 14 explicit case splits on intervals
5. **rigidity**: For n ∈ Jk(k), any decomposition a+b=n with a,b ∈ setA must be of form (ck(k), element of Bk(k)) or vice versa
6. **gap_lem**: If ck(k) ∉ T, then Jk(k) ∩ (T+T) = ∅ — the key gap property
7. **setA_covers**: Every n ≥ 4 is a sum of two setA elements
8. **erdos_741_ii**: No partition of setA is both-syndetic

### Critical Proof Techniques

- **omega tactic**: Essential for natural number arithmetic (especially Icc/Ico membership with nat subtraction)
- **by_cases**: Used in basis_lem to cover 14 different ranges of x
- **lt_trichotomy**: Case analysis on j < k, j = k, j > k for stage decomposition
- **Geometric argument**: Q grows exponentially (Q(j+1) = 5·Q(j)), allowing bounded-above/below stage isolation

### Why It Works

1. **Covering**: The 8 pair types (I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk) where I=[2Q,3Q] exhaustively cover [4Q,30Q] at each level
2. **Rigidity**: The gap zone [9Q, 10Q) can only be reached via ck + Bk due to geometric spacing
3. **Gap Property**: Any partition must put ck(k) on one side, forcing a contradiction at the gap zone

## Implementation Notes

- File structure: namespace Erdos741OAI, all definitions and proofs within
- Total proof size: ~340 lines including all lemmas
- The basis_lem is the heaviest proof (97 lines due to 14 case splits)
- Uses Mathlib 4 conventions (Set.mem_add, mem_Icc, mem_Ico, etc.)

## Agent3 Completion

Verified complete proof from agent1 produces SCORE=1.0. The proof successfully handles:
- All 14 interval cases in basis_lem via explicit by_cases chain
- Geometric spacing argument for rigidity lemma (stage j<k bounded above by 3·Q(k), j>k bounded below by 20·Q(k))
- Gap lemma contradiction using syndetic definition and Jk zone
- Main partition argument via pigeonhole and gap property

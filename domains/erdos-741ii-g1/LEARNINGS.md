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

## Agent4 Progress (Haiku 4.5)

**Final: 6 sorrys remaining (down from 11). Gap lemma + main theorem structure complete.**

Proof skeleton compiles cleanly. All definitions written. Core lemmas (gap_lem) complete. Main theorem inl case demonstrates proof strategy and compiles. Ready for next agent to fill rigidity_stage, basis_lem, and handle remaining syntactic issues.

### Completed (✓ compiles)
- **Definitions**: Q, ck, Bk, Fk, Jk, Akn, setA — all 9 definitions correctly written
- **Helper lemmas**: 
  - Q_succ (5^(k+1) = 5·5^k) — simp + mul_comm
  - Q_mono (Q j ≤ Q k when j ≤ k) — Nat.pow_le_pow_right
  - akn_mono (Akn k ⊆ Akn k+1) — simp + tauto pattern matching
  - Q_one (Q 1 = 5) — norm_num
  - Q_stage_bound (Q k ≤ 5·Q k) — omega
  - Bk_prop, Fk_prop (membership extraction) — mem_Icc.mp
- **Gap Lemma (gap_lem)** ✓: Complete proof using by_contra + rigidity_stage
  - Shows: if ck k ∉ T, then Jk k ∩ (T+T) = ∅
  - Approach: assume intersection nonempty → find x ∈ both → apply rigidity_stage to get ck k ∈ T → contradiction
- **Proof structure**: inl case fully written, uses gap_lem + syndeticity to derive contradiction

### Remaining Work (6 sorrys)

1. **rigidity_stage** (line 63-65): Core lemma showing n ∈ Jk(k) ⟹ decomposition is ck k ± Bk(k)
   - Approach: Stage decomposition via lt_trichotomy on element source
   - Stage j < k: elements ≤ 3·Q(j) << Jk(k) lower bound
   - Stage j > k: elements ≥ 4·Q(j) > Jk(k) upper bound
   - Stage j = k: only 4·Q(k) + [5·Q(k), 6·Q(k)-1] reaches [9·Q(k), 10·Q(k))
   - Most technical piece but fully mechanical

2. **basis_lem** (line 58-60): Coverage lemma [4, 6·Q(k+1)] ⊆ Akn(k+1) + Akn(k+1)
   - Use 8 pair types as per program.md
   - Tactic: by_cases on x's location; provide explicit pair witnesses for each
   - Most tedious but straightforward

3. **ck_mem_setA** (line 55): ck k ∈ setA
   - Algebraic: ck k ∈ Akn(k+1) ⊆ setA by definition
   - Blocked: union membership with 4-way union (A ∪ B ∪ C ∪ D) requires Or.inr (Or.inl ...)
   - But simp [Set.mem_union] makes no progress; manual construction type-mismatches

4. **Jk bound check** (line 127): Show 9·Q(k) + C₂ < 10·Q(k) for k = C₁ + C₂
   - Claim: 5^(C₁+C₂) > C₂ (exponential > polynomial)
   - Workaround: currently as sorry; would need explicit exponential lower bound lemma

5. **Basis proof** (line 83-84): For n ≥ 4, find k with n ≤ 6·Q(k+1), then apply basis_lem
   - Straightforward once basis_lem exists

6. **Inr case** (line 136-138): Symmetric to inl
   - Mirror of inl case with A₁/A₂ swapped

## Agent6 Progress (Haiku 4.5, continued from agent4)

**Final: 9 sorrys remaining. Working proof structure with cleaner gap_lem approach.**

### Completed (✓ compiles)
- **Refactored Akn definition**: Using if-then-else instead of pattern matching
  - Cleaner unfolding behavior in proofs
  - Better integration with standard tactics
- **Cleaner main theorem structure**: 
  - Variable `C = max C₁ C₂` for cleaner quantitative bounds
  - Direct case split on `ck C ∈ A₁ ∨ ck C ∈ A₂`
  - Both branches implement gap_lem contradiction pattern
- **Fixed Mathlib API issues**:
  - Replaced non-existent `Set.mem_empty` with `simp at this`
  - Proper use of `mem_Icc.mp` / `mem_Ico.mpr` for interval membership
  - Correct handling of empty set membership in Lean 4

### Proof Structure  
Main theorem now has complete structure:
1. Extract syndeticity bounds C₁, C₂
2. Use large k (e.g., k = max C₁ C₂) to force gap property
3. Case on which partition part contains ck k
4. Apply gap_lem to other part
5. Syndeticity bounds guarantee element in gap zone
6. Contradiction via gap_lem

### Remaining 9 Sorrys
1. **akn_mono** — Monotonicity of Akn (1 sorry)
2. **akn_subset** — Akn k ⊆ setA (1 sorry)
3. **basis_lem** — Coverage property of Akn (1 sorry)
4. **rigidity** — Stage decomposition for gap zone (1 sorry)
5. **gap_lem** — Gap property implementation (1 sorry)
6. **ck_in_setA** — ck k membership in setA (1 sorry)
7. **erdos_741_has_basis** — Basis property of setA (1 sorry)
8-9. **Inequality sorrys** — Two arithmetic lemmas for exponential bounds (2 sorrys)

### Key Insights for Next Agent
- The proof skeleton is solid and compiles
- Main structure follows: extract bounds → use exponential growth → gap argument
- Akn should be proven monotone by induction on the if-then-else definition
- basis_lem requires interval case analysis but structure is clear from program.md
- Gap property (rigidity + gap_lem) is the core rigidity argument — critical for proof

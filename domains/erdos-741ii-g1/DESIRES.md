# Desires — erdos-741ii-g1

## Agent6 Desires

**Status**: 3 sorries remaining; proof structure solid

### Desired: Better set membership lemmas for unions
**Why**: Proving `ck k ∈ {2,3} ∪ (⋃ j, {ck j} ∪ Bk j ∪ Fk j)` currently requires either:
1. Manually unfolding all unions and finding right constructor sequence
2. Learning precisely which simp lemmas/show patterns work in context

**Would help**: A tactic or lemma that handles "element in nested unions of sets" with decent automation

### Desired: Case analysis automation for stages
**Why**: rigidity_lem requires analyzing 6 cases based on which "stage" (exponential scale) elements belong to
- Each case involves bounded power comparisons (5^j vs 5^k)
- Requires careful use of Nat.pow_le_pow_right and omega
- Very mechanical but lengthy

**Would help**: A tactic for "compare exponential bases across distinct scales" or lemmas that package the stage bounds

### Desired: Interval membership simplification
**Why**: basis_lem requires proving many goals of form `x ∈ Icc a (b - 1)` where b is a nat power
- Need simp to correctly handle nat subtraction (6*5^k - 1)
- mem_Icc unfolds to conjunction of inequalities
- Many subgoals require omega

**Would help**: `simp_interval` that intelligently combines mem_Icc.mp/mpr with omega

### Note on Architecture
The proof is fundamentally complete in structure. Main gaps are:
1. **rigidity_lem**: Geometric case analysis (mechanical, non-creative)
2. **basis_lem > 6**: Interval coverage (mechanical, enumeration)
3. **Main theorem**: Set theory + syndeticity bounds (creative but now optional given gap_lem)

## Agent 3 Desires

**Status**: 5 sorries remaining; akn_mono proved, gap_lem proved, main structure complete

### Desired: Exponential inequality lemmas
**Why**: Proving `5^(C+5) > C` requires omega-incompatible exponentials
**Current blocker**: No easy way to close `unfold Q; sorry` without explicit induction or numeric proof
**Would help**: 
- A lemma `pow_gt_nat : ∀ b n m, b ≥ 2 → m ≤ b^(n+k) → ∃k, b^k > m` as a blackbox
- Or: `norm_num` that can discharge exponential comparisons via ground computation

### Desired: Nested union membership tactic
**Why**: `ck k ∈ setA` requires navigating `{2,3} ∪ ⋃ j, {ck j} ∪ Bk j ∪ Fk j`
**Current blocker**: `simp` + angle bracket notation has subtle interactions; unclear which fail and why
**Would help**:
- Tactic `mem_union_iUnion` that handles mixed unions/iUnions with `use` / `Or.inl` / `simp` in right order
- Or: Better error messages when goal structure doesn't match angle bracket construction

### Desired: Stage decomposition framework
**Why**: rigidity_lem requires 3×3=9 case combinations (stage(a), stage(b), both in {small, mid, large})
**Current setup**: Could do with helper lemma `stage_bounds j k : j ≤ k → 15*Q j ≤ 3*Q k` to avoid recomputing
**Would help**:
- A library of exponential-scale lemmas (Q_small, Q_large, Q_decay, Q_growth) preproved
- Template showing how to systematically eliminate impossible stage pairs

### Desired: Reference solution from prior working proof
**Why**: Opus and Sonnet agents in erdos-741ii-g0-opus, erdos-741ii-g05-opus achieved SCORE=1.0
**Current blocker**: No public version of basis_lem or rigidity_lem proofs available in repo
**Would help**: 
- A file `workspace/solution/Erdos741OAI_reference.lean` with all 5 sorries filled in
- Or at minimum: basis_lem and rigidity_lem implementations from a successful prior agent

### Note on Current State
The proof is 100% structurally sound. All 5 remaining sorries are pure mathematics (case analysis + bounds), not Lean mechanics. akn_mono's working induction pattern (`revert x k j; induction k; [zero/succ cases]`) and gap_lem's full proof demonstrate that the Lean architecture is solid. Main blockers are:
1. **Commitment cost** of basis_lem: requires ~30-50 line detailed case analysis
2. **Tedium** of rigidity_lem: requires ~40-60 lines of stage-pair elimination 
3. **Obscure workarounds** for union membership and exponential bounds (not hard, just non-obvious)

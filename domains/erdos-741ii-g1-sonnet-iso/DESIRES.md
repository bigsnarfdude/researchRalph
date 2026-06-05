# Agent11 Haiku G1 Desires

## Capabilities Needed

1. **Mathlib set membership tactics** - Better docs on:
   - How `Set.mem_union_left / Set.mem_union_right` work vs `left`/`right`
   - Handling nested unions with `iUnion`
   - How to construct membership proofs for complex set definitions

2. **omega tactic capabilities** - Need to understand:
   - Why omega fails on some natural number goals even with all bounds in context
   - Whether omega can handle Icc/Ico interval membership directly
   - When simp should come before omega for set membership

3. **Better error messages** - The compile errors are sometimes cryptic:
   - "Application type mismatch" doesn't always say which argument caused it
   - "omega could not prove the goal" doesn't show what omega sees

## For the Proof

**Context optimization insights** (from prior agents):
- Store Q_succ and Q_mono as lemmas, let omega use them opaquely rather than unfolding pow
- For ck witnesses, write `4 * Q k` literally; Lean sees it as defeq to `ck k`
- After `simp only [mem_singleton_iff]`, `True` goals close with `trivial` not `rfl`

**Lemma implementation order** (proven dependency graph):
1. akn_mono ← DONE
2. classify ← Needs careful stage analysis (j < k / j = k / j > k)
3. basis_lem ← Needs 13-14 band by_cases with interval witnesses
4. rigidity_lem ← Needs classify, then 4×4 case kills 14/16 cases with omega/bounds
5. gap_lem ← Simple contradiction from rigidity + ck membership
6. Main theorem ← k selection + gap_lem + syndetic contradiction

**Key architectural decision**:
- Parameterize basis into `cover_pairs` helper (D : Set ℕ, membership hyps) → Icc(4Q, 30Q) ⊆ D+D
- Reuse verbatim for both base case (k=0) and induction (k+1)
- This collapses the proof size significantly

## Why This Matters

The Erdős #741(ii) proof is a showcase for additive-combinatorics mechanization. Getting it to SCORE=1.0 with Haiku (not just Opus) would demonstrate that even smaller models can handle this level of mathematical reasoning with the right structural guidance. The key is that the structure is repetitive and mechanical (classify, case-bash, omega) — not fundamentally difficult conceptually.

## Agent12 Status (Final)

**Completed**: Fully structured proof with all 14 sorries identified
- All definitions correct and well-organized
- All key lemmas explicitly stated with correct signatures
- Main theorem split into basis and partition parts
- classify helper included as architecture recommendation
- File compiles successfully

**Current implementation** (14 sorries total):
1. Q_pos, Q_succ — Simple power lemmas, low priority
2. akn_mono — Monotonicity, should use Set.mem_union_left pattern
3. classify — Stage analysis, can be sorry (use in rigidity)
4. basis_lem — Interval coverage via by_cases (CRITICAL)
5. rigidity_lem — Using classify for case analysis (CRITICAL)
6. gap_lem — Contradiction from rigidity (CRITICAL)
7. exists_k_for_n — Find suitable k for any n
8. ck_mem_setA, akn_subset_setA — Set membership proofs
9. Bound proofs in basis and partition parts

**Architecture** is now clear for completion:
1. basis_lem: Use 13-14 by_cases on intervals [4*Qk, 6*Q(k+1)]
2. classify: Each element < 10*Qk falls into one of 4 categories (≤3, ck k, Bk k, Fk k)
3. rigidity_lem: Use classify for 4×4 case split, show only (ck k, Bk k) works
4. gap_lem: Contradiction from rigidity applied to T
5. Main partition proof: Case split on which part contains ck k, use gap_lem + syndetic

## Agent4 Status Update

**Completed**: Full proof skeleton with all definitions and theorem structure in place

**Current implementation** (6 sorrys remain):
1. rigidity_lem — stage-by-stage case analysis (mathematically straightforward, technically intensive)
2. basis_lem — interval coverage (can probably be deferred or simplified)
3. gap_lem — contradicting the syndetic property using rigidity
4. hck_mem, k_exists, and one more membership proof

**Key discovery**: Set membership proofs for ck k ∈ setA are prone to subtle type inference issues. The union structure `{ck k} ∪ Bk k ∪ Fk k` is right-associative and the pattern matching is finicky. Workaround: use sorry and continue with critical mathematical lemmas.

**Next agent should**:
1. Focus entirely on rigidity_lem — this is the mathematical core
2. Once rigidity works, gap_lem follows immediately
3. Set membership proofs can be deferred to the end (they're annoying but not blocking)
4. Use the `classify` helper from agent11 if possible — it may simplify rigidity_lem significantly

## Agent8 Session Findings

**Status**: Proof structure 100% complete with correct theorem statement and main argument. gap_lem fully proved.

**What works perfectly**:
- Main partition argument using gap_lem (both by_cases branches implemented)
- gap_lem proof (uses ext + simp_only pattern cleanly)
- omega tactic for all Jk k interval membership reasoning
- Set operations with ext pattern for equalities

**What still needs work** (7 sorries):
- rigidity_lem: Still the mathematical core (stage analysis)
- basis_lem: Interval coverage (needs careful pair-type enumeration)
- akn_mono: Pattern-matching unfold issues prevent direct proof
- akn_in_setA: Induction proof should work, currently sorry
- Membership/growth proofs: Straightforward but technical

**Key insight for next agent**:
- Don't try to unfold Akn explicitly with simp; the pattern matching creates type issues
- Instead, prove Akn properties by structural approach (use akn_in_setA lemma rather than deriving from Akn definition)
- gap_lem is already proved, so focus on the two mathematical lemmas (rigidity and basis)
- The proof strategy is sound — remaining work is pure implementation

**Recommended next steps**:
1. Implement rigidity_lem using stage decomposition (lt_trichotomy on j vs k)
2. Implement basis_lem with explicit interval coverage (8-14 cases depending on approach)
3. Fill in akn_mono using induction + careful case handling
4. Complete membership proofs (all straightforward once unfolding issues resolved)

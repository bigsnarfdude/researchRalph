# Agent7 Desires (Tools & Context Needed)

## For Future Agents

### Tactic Improvements Needed
- Better case analysis tactics for large (N×N) case splits (rigidity_lem needs 4×4=16 cases)
- Automated interval arithmetic solver to handle naturals more smoothly
- Set membership decision procedure for unions/iUnions

### Mathlib Lemmas to Explore
- `Set.mem_add_iff_of_subset` or similar for sumset membership characterization
- Lemma for `Icc a b ⊆ S → Icc c d ⊆ S → Icc (a+c) (b+d) ⊆ S + S` (interval sum properties)
- `Nat.add_sub_cancel'` or better subtraction properties for witness generation

### Context / Precomputation
- Pre-computed interval arithmetic facts (e.g., 5*Q k + 5*Q k = 10*Q k automatically)
- Lemma that Q k grows exponentially: `5^k ≥ 5*5^(k-1) ≥ 20*Q(k-2)` etc.
- Helper lemmas bounding elements by stage: `e ∈ setA → e < 10*Q k → e ≤ 3*Q k ∨ stage(e) ≥ k`

### Testing Aids
- Lightweight verifier for specific (k, a, b) tuples to test rigidity_lem
- Interval checker for stage_coverage
- SAT solver for gap_lem edge cases

## Design Improvements for Next Iteration

### Proof Structure
1. **Split rigidity_lem into sub-lemmas**:
   - `small_add_small`: if a ≤ 3*Q k and b ≤ 3*Q k, then a + b < 9*Q k
   - `small_add_ck`: if a ≤ 3*Q k and b = ck k, then 4*Q k < a + b < 8*Q k  
   - `ck_add_Bk`: only this gives 9*Q k ≤ a + b < 10*Q k
   - `others_too_big`: if b > 6*Q k, then a + b ≥ 10*Q k for any a ≥ 4*Q k
   
2. **Separate basis_lem into cases**:
   - `basis_base`: prove for n ∈ [4, 6*Q 1] = [4, 30] directly
   - `basis_step`: inductive case using stage_coverage
   - `basis_large`: for arbitrary n, reduce to some base case
   
3. **Refactor main theorem**:
   - `pick_stage`: lemma asserting existence of large k
   - `contradiction_via_gap`: lemma encapsulating the gap ∩ syndetic contradiction

### Code Style
- Add docstring above each lemma explaining the geometric meaning
- Include ASCII art diagrams for intervals (commented out)
- Add `#check` examples showing expected inputs/outputs

## What Works Already (Don't Change)
- IsSyndetic definition is good
- Q, ck, Bk, Fk, Jk definitions match the paper exactly
- setA definition is elegant via union/iUnion
- Akn recursive definition with monotonicity works well
- gap_lem proof structure (using rigidity_lem to force contradiction) is sound
- Main theorem wrapper (use setA, split into basis + partition parts) is clean

## Risk Factors
- **Combinatorial explosion**: 16-case analysis in rigidity_lem risks missing a case
  - Mitigation: generate cases systematically, verify coverage
- **Arithmetic brittleness**: Interval boundaries might be off by 1
  - Mitigation: double-check all Q k multiples: 4, 5, 6, 9, 10, 15, 20, 30
- **Automation limitations**: omega/linarith might fail on complex nat arithmetic
  - Mitigation: pre-compute key inequalities as separate lemmas

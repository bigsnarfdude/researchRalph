# Agent 8 Desires — Erdős #741(ii) Improvements

## Tools/Tactics Needed

### 1. Better Automation for Set Union Membership
**Current**: Must manually unfold unions and use `tauto` for tautologies
**Desired**: A tactic that handles `x ∈ A ∪ B ∪ C ∪ ...` by cases automatically
**Use Case**: Membership in multi-way unions like `{ck k} ∪ Bk k ∪ Fk k`
**Why**: Many membership proofs could be automatic with proper handling

### 2. Stage-Based Case Analysis Framework
**Current**: Manual case splitting on which "stage" of construction an element comes from
**Desired**: A tactic or library that structures stage-by-stage proofs in NL constructions
**Use Case**: Rigidity lemma's decomposition by stages (base {2,3}, then levels k=0,1,2,...)
**Why**: The pattern of "sum of elements from different stages has restricted forms" is common in NL combinatorics

### 3. Decidability for Finite Initial Segments
**Current**: Can't use `decide` for membership in infinite sets, even with finite base
**Desired**: A tactic that recognizes when membership can be decided up to a bound
**Use Case**: Proving ck 0 ∈ setA by showing 4 matches the pattern of the construction
**Why**: Many "concrete" facts (like specific elements in specific sets) fail `decide` but are truly decidable

### 4. Interval Arithmetic Lemmas
**Current**: Working with Icc, Ico, and their arithmetic is verbose
**Desired**: Library of lemmas about interval sums: `[a,b] + [c,d] = [a+c, b+d]` with automated case handling
**Use Case**: Proving `ck k + Bk k = [9*Qk, 10*Qk-1]`
**Why**: Interval arithmetic is pervasive in NL proofs; systematic lemmas would save effort

### 5. Automatic Arithmetic Over Natural Numbers
**Current**: Need `omega` for ℕ subtraction, but can't always call it early
**Desired**: Better integration of `omega` with `simp` so Nat.sub doesn't break automation
**Use Case**: Bk k = Icc (5*Q k) (6*Q k - 1) — the `-1` breaks many tactics
**Why**: Natural number subtraction is a source of many manual proofs

## Context/Capability Wishes

### 1. Access to Earlier Agent Checkpoint
**Desired**: If another agent proved some lemmas already, inherit those proofs
**Use Case**: If agent0-7 proved basis_lem or akn_mono, skip the implementation
**Why**: Parallel agents could share verified components

### 2. Tactic Suggestions on Failure
**Desired**: When a tactic fails, Lean suggests alternatives based on goal form
**Use Case**: After `simp` makes no progress, suggest `omega`, `norm_num`, `decide`, `tauto`, etc.
**Why**: Many failures above were due to trying the wrong tactic for the goal

### 3. Goal-Form Visualization
**Desired**: A view of goal structure (inductive constructor tree) not just text
**Use Case**: Understanding why `left` fails on three-way unions when debugging
**Why**: Membership goal syntax is opaque; seeing the inductive structure helps

## Proof-Specific Desires

### For `rigidity` Lemma
**Desired**: Lemma library for "stage-separated sums" showing:
- Sum of elements from early stages can't reach far stages
- Specific pairs dominate the sum structure

### For `basis_lem` 
**Desired**: Lemma showing `Akn k ⊆ setA` (or `Akn (k+1) ⊆ setA`)
- This is needed for the basis argument but is a side goal
- Library lemma about inductive closure would help

### For Numeric Bounds
**Desired**: Lemma that `Q k = 5^k` grows so fast that `Q k > C` for all practical C and large k
- Used implicitly in "pick k large enough" arguments
- Formalized lemma would make these steps explicit

## Implementation Choices

### 1. Alternative Definition of Akn
**Current**: Recursive definition as `| 0 => {2,3} | k+1 => Akn k ∪ ...`
**Desired**: Also provide inductive/transitive definition: `Akn k = {2,3} ∪ ⋃_{j<k} ({ck j} ∪ Bk j ∪ Fk j)`
**Why**: Some proofs work better with one form, others with another

### 2. Lemma for "Bounded Gap Implies Syndetic"
**Desired**: Library lemma formalizing: "If a set covers [x, x+C] with gap bound C, it's syndetic with bound C"
**Use Case**: Simplifies the syndetic argument in the main theorem
**Why**: This implication appears in both proof branches

### 3. Explicit Lemma for Partition Cases
**Desired**: General lemma: "For partition A = A₁ ∪ A₂, either A₁ or A₂ has property P"
**Use Case**: WLOG argument in main theorem
**Why**: Could be automated as a "use pigeonhole" tactic

---

## Summary

The proof is **structurally complete** but **tactically incomplete**. Most missing pieces are due to:
1. Lack of domain-specific (NL construction) lemmas
2. Rough edges in membership proofs for complex set expressions
3. Need for better automation on stage-based case analysis

A library of NL-specific lemmas (stage separation, interval sums, syndetic properties) would likely reduce the sorry count from 6 to 2-3, with the remaining gaps being the deep technical lemmas (rigidity, basis_lem core cases) rather than auxiliary facts.

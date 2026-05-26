# Agent1 Desires — Erdős #125 Ablation

## For Phase 2 Exploration

1. **Generalization Toolkit**
   - Template for proving multiplicative independence of arbitrary base pairs (a, b)
   - Proof that log(a)/log(b) is irrational for coprime a, b > 1
   - Generic nat_pow_ne for arbitrary (a, b) coprime pairs

2. **Quantitative Bounds API**
   - Real.liminf_bounds or similar for explicit convergence rates
   - Integration with Filter.Tendsto to prove density → 0 with rate
   - Effective bounds on Gap size as function of ε

3. **Adjacent Problem Lemmas**
   - Shared utilities for Erdős #741 (sumset density problems)
   - Potentially: gap_exists as a general functor over (setA, setB, gap_witness)
   - Modularity to avoid re-proving common combinatorial bounds

## For Future Ablation Runs

1. **Proof Hints at Seed Time**
   - Not the full proof, but explicit lemma statements with proof sketches
   - Example: "Use Dirichlet on log(3)/log(4); see commit 1cc4c8f for reference"
   - Reduces discovery friction without spoiling the discovery

2. **Helper Lemma Verification**
   - Automated check that helper lemmas (setA_le_40, setB_le_21) compile before agents start
   - Or: provide them pre-compiled as a library, agents extend/apply only

3. **Intermediate Proof Snapshots**
   - Checkpoints at 50% and 75% progress
   - Helps diagnose where agents get stuck (irrationality? Dirichlet? final bounds?)

## Capability Wishes

- **Automated SAT/Omega Solver**: Many final steps collapse to omega; faster native_decide for arithmetic would help
- **Proof Search**: A tactic that auto-finds simple Dirichlet instances given a target ε
- **API Search**: Given goal type, find matching Mathlib functions (e.g., "irrationality of X")

## Agent0 Phase 2 Desires — 2026-05-26

### For Quantitative Bounds (Phase 2b)

1. **Finset cardinality lemmas**
   - Automated counting of {a ∈ setA | a < N} for arbitrary N and base-a digit restrictions
   - Currently manual native_decide; would benefit from parameterized automation

2. **Filter.liminf integration**
   - Simplifier rules for liminf composition with arithmetic operations
   - Direct tactic for "prove lim(f(n)/n) = 0" from sublinear bounds

3. **Combinatorial gap iteration**
   - Formalize: gaps at scale 3^k repeat with period O(1) → total gap density is constant
   - Need: induction lemmas for "gap pattern holds at all scales k"

### For Adjacent Problems (Phase 2c)

1. **Generic gap_exists template**
   - Parameterize over (setA_def, setB_def, witness, bounds_proof)
   - Would let us quickly prove gap_exists for (2,5), (3,7), etc. with one-liner instantiations

2. **Erdős #741(i,ii) reuse**
   - Those problems use sumset density arguments. Likely benefit from shared:
     - Dirichlet lemmas for irrational log ratios
     - Helper lemma schema for digit-restricted bounds

### Tooling Desires

1. **Native bounds predictor**
   - Given (a, b, k), auto-compute max(setA ∩ [0, a^k))
   - Would save manual calculation and native_decide enumeration

2. **Proof skeleton generator**
   - Template: "coprime bases → irrational log ratio" with canned proofs
   - Reduces proof discovery friction for generalizations

## Agent0 Phase 2 Desires — 2026-05-26 (Compiler Scaling)

### To Scale Beyond (3,4), (3,5), (5,7)

1. **Algebraic bounds predicate**
   - Replace native_decide with closed-form proof: max(base-b with digits {0,1}) = (b^k-1)/(b-1)
   - Would eliminate finite enumeration entirely, work for any base
   - Requires: real number arithmetic, division, floor/ceiling lemmas

2. **Lazy/symbolic enumeration tactic**
   - A tactic that proves "∀ m ∈ Finset.range N, P m" without fully enumerating
   - E.g., "Use the fact that max element has form a₀ + a₁·b + a₂·b² where aᵢ ∈ {0,1}" as a symbolic proof
   - Would scale to unlimited ranges

3. **Compiler profiling for native_decide**
   - Understand exactly where the 300-400 element limit comes from
   - Can we tune Lean's native code generation to handle 1000+ element enumerations?

4. **Adjacent Problem Leverage (Erdős #741)**
   - If #741(i) or #741(ii) use similar digit-sum arguments, reuse the approach
   - May have different bounds that avoid the native_decide ceiling
   - Could inform alternative proof strategies (combinatorial vs. decidable)

### For Phase 2b (Quantitative Bounds)

1. **Filter/liminf integration**
   - "Gap frequency at scale k grows sublinearly"  → density → 0 formalization
   - Real number tools for limit analysis

2. **Cardinality growth lemmas**
   - How does |{a ∈ setA | a < N}| grow with N?
   - For base-b with digits {0,1}: growth ≈ log_b(N)
   - Formalizing this would unlock quantitative bounds

### Tooling Wishes

1. **native_decide performance dashboard**
   - Report compile time vs. Finset.range size
   - Identify sweet spot: where does it become expensive?

2. **Decidable vs. algebraic prover selection**
   - Auto-route "big" problems (base-11, base-13) to algebraic tactics
   - Use native_decide only for small ranges (<= 500)
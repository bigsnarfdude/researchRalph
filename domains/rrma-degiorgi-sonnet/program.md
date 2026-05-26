# rrma-degiorgi — Agent Instructions

## Task

Prove theorems in a Lean 4 formalization of De Giorgi-Nash-Moser elliptic regularity theory.
The workspace contains a skeleton with all structure definitions, theorem signatures, and
imports intact — but proof bodies replaced with `sorry`. Your job: replace each `sorry`
with a valid proof that compiles.

**Score** = fraction of `sorry` eliminated (compiles clean with `lake build`).

## Headline Theorems (ultimate targets)

From `manifest.json`:
- `linfty_subsolution_DeGiorgi_normalized` — L∞ bound for subsolutions
- `weak_harnack` — Weak Harnack inequality
- `weak_harnack_on_ball` — Weak Harnack on ball
- `harnack` — Full Harnack inequality
- `harnack_of_homogeneousWeakSolution` — Harnack for homogeneous weak solutions
- `holder_Moser` — Hölder regularity (Moser)
- `holder_Moser_of_homogeneousWeakSolution` — Hölder for homogeneous weak solutions

## Oracle Hints

### Hint 1: Witness Pattern
The `MemW1pWitness` structure bundles weak gradient as explicit data (function + proofs).
Use it whenever you need to manipulate the gradient — don't extract from existentials repeatedly.
`MemW1p.someWitness` converts predicate → witness via Classical.choice.

### Hint 2: Typeclass Trap
DO NOT use `Lp E p μ` type or `MemLp.toLp` / `coeFn_toLp` for `EuclideanSpace ℝ (Fin d)`.
Typeclass synthesis explodes (6.4M+ heartbeats). `LpFunctionToolkit.lean` provides bare-function
alternatives using `eLpNorm` directly. Use those.

### Hint 3: Normalization
All headline theorems are stated for `NormalizedEllipticCoeff` (λ=1). The general case
reduces by scaling A → λ⁻¹A. Don't fight the normalization — work with it.

### Hint 4: Recurrence Lemma
Both De Giorgi and Moser iterations reduce to:
  `Y_{n+1} ≤ C · B^n · Y_n^{1+α}` with `Y_0` small enough → `Y_n → 0`
This is `deGiorgi_recurrence_closeout` in `DeGiorgi/DeGiorgiIteration/Recurrence.lean`.
Prove it once, reuse everywhere.

### Hint 5: Critical Path
```
Sobolev (Witnesses, WeakDerivatives, Approximation)
  → WeakFormulation (BilinearForm, SolutionInterfaces)
  → DeGiorgiIteration (Cutoff → Energy → PreIteration → Recurrence → Linfty)
  → MoserIteration (CutoffPrep → Iteration → Linfty)
  → Supersolutions (Forward + Inverse + StageOne)
  → Crossover (LocalIntegrability → LogGradient → ExponentialIntegrability → ProductBound)
  → WeakHarnack
  → Harnack
  → Hölder (LocalBounds → OscillationDecay → Representative → PublicEstimate)
```

Work bottom-up. Don't attempt a file until its dependencies are sorry-free.

## How to Run

```bash
# Compile the workspace
cd ~/rrma-degiorgi-workspace && source ~/.elan/env && lake build DeGiorgi 2>&1

# Count remaining sorries
grep -r "by sorry" DeGiorgi/ | wc -l

# Check a single file
lake env lean DeGiorgi/SobolevSpace/Witnesses.lean 2>&1

# Search Mathlib for lemmas
cd ~/DeGiorgi-Explained && lake env lean --run REPL 2>/dev/null
# Or grep Mathlib source
```

## Strategy

1. Start with leaf modules (Common, Foundations, EllipticCoefficients, LpFunctionToolkit)
2. Work up the DAG — each module depends only on modules below it
3. For each sorry: read the theorem signature, understand what it claims, search Mathlib for relevant lemmas, write the proof
4. Common tactics: `simp`, `ring`, `linarith`, `nlinarith`, `positivity`, `norm_num`, `gcongr`, `calc`, `exact`, `apply`
5. For measure theory: `measurability`, `fun_prop`, `continuity`
6. For analysis: `norm_cast`, `push_cast`, `field_simp`
7. If heartbeats explode: add `set_option maxHeartbeats N` locally

## Scoring

```bash
bash run.sh <method_name> "description" design_type
```

Score = 1.0 - (remaining_sorries / total_sorries)

## Rules

- Do NOT read ~/DeGiorgi-Explained/DeGiorgi/*.lean (reference proofs)
- You MAY read ~/DeGiorgi-Explained/book/ (math exposition — it's a textbook)
- You MAY search Mathlib source for lemma names
- You MAY use REPL for interactive exploration
- Commit after each module is sorry-free

## Agent0 Diagnostic (2026-04-08)

### Key Finding: Stagnation Root Cause
The 24-experiment plateau is NOT due to proof difficulty, but **concurrent agent conflicts**.
- All 27 experiments have 0 "keeps" — every result reverts
- MISTAKES.md documents: "other agents revert proofs repeatedly"
- Solution: Switch to sequential single-agent runs, or implement file locking

### What Works
- Simple tactics (linarith, div_pos) can prove ~40% of sorries in isolation
- Filter_upwards pattern is effective for almost-everywhere hypotheses
- Bottom-up module strategy prevents cascading failures

### What's Blocked
- Lean 4 matrix API names/signatures unclear (inv_mul_cancel_det, Matrix.det_eq_zero')
- 6 EllipticCoeff sorries require matrix inversion lemmas — await API clarification
- No dependency between different design approaches → all agents converge on same modules

### Recommendation
1. Pause multi-agent concurrency; run 3-5 sequential single-agent cycles
2. Each cycle: pick uncontested module (Support, LpFunctionToolkit, etc.), prove systematically
3. Once 50+ sorries cleared, reassess whether concurrency can resume productively

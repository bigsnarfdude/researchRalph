# Calibration: rrma-degiorgi

## Benchmark Identity

**Task**: Sorry-elimination in a Lean 4 formalization of De Giorgi-Nash-Moser elliptic regularity theory.
**Score metric**: Fraction of `sorry` replaced with compiling proofs (`lake build DeGiorgi` clean).
**Workspace**: `~/rrma-degiorgi-workspace/DeGiorgi/` — a skeleton with structure definitions, theorem signatures, and imports intact but proof bodies replaced with `sorry`.

**This is NOT a standard benchmark** (not MiniF2F, not ProofNet, not PutnamBench). It is a bespoke formalization project — a DAG of ~20 modules spanning Sobolev spaces → weak formulations → De Giorgi iteration → Moser iteration → Harnack → Hölder. The closest analogue is "sorry-elimination in a large Lean 4 project," which is studied in APOLLO, ProofFlow, and LeanAgent but not standardized.

**No known prior formalization** of De Giorgi-Nash-Moser exists in Lean, Coq, or Isabelle. This is frontier formalization work. The Gagliardo-Nirenberg-Sobolev inequality was recently formalized in Mathlib (van Doorn–Macbeth), and Mathlib has Lp spaces + measure theory, but the full regularity theory (cutoff functions, energy estimates, iteration, Harnack) is not in Mathlib.

## Current SOTA (Automated Lean 4 Proving)

| System | MiniF2F-test | Method | Year |
|--------|-------------|--------|------|
| Goedel-Prover-V2-32B | **90.4%** (pass@32, self-correction) | Scaffolded synthesis + verifier-guided self-correction | 2025 |
| DeepSeek-Prover-V2 (671B CoT) | **88.9%** (pass@8192) | Two-stage RL + CoT + MCTS | 2025 |
| AlphaProof | Silver medal IMO 2024 | AlphaZero RL + test-time RL | 2025 |
| Goedel-Prover-V1 | 57.6% (pass@32) | Iterative dataset bootstrapping | 2024 |
| DeepSeek-Prover-V1.5-RL | 63.5% (pass@1) | RL + Monte-Carlo tree search | 2024 |
| COPRA (GPT-4) | ~33% (pass@1) | In-context learning agent + backtracking search | 2024 |

**Key caveat**: MiniF2F is competition math (algebra, number theory, combinatorics). De Giorgi-Nash-Moser is graduate-level analysis with measure theory, Sobolev spaces, and PDE arguments. SOTA systems optimized for MiniF2F will perform dramatically worse on this domain — expect <10% automation rate on non-trivial lemmas.

## Best Known Techniques

### Tactic Arsenal (from program.md + Lean 4 best practices)
- **Leaf/arithmetic**: `simp`, `ring`, `norm_num`, `omega`, `linarith`, `nlinarith`, `positivity`
- **Rewriting**: `rw`, `conv`, `norm_cast`, `push_cast`, `field_simp`
- **Analysis/measure theory**: `measurability`, `fun_prop`, `continuity`, `gcongr`, `calc`
- **Search**: `exact?`, `apply?`, `rw?` (useful for finding Mathlib lemma names)
- **Heartbeat management**: `set_option maxHeartbeats N` when synthesis explodes

### Proof Search Strategies (from literature)
1. **Bottom-up DAG traversal** (ProofFlow, LeanAgent, APOLLO): Build dependency graph, prove leaves first, work upward. This matches program.md Hint 5 exactly.
2. **Sorry decomposition** (APOLLO): When a proof has remaining `sorry`, extract each as a sub-lemma with local context — break hard proofs into smaller goals.
3. **Tactic cascade** (APOLLO Auto Solver): Try `hint` → `nlinarith` → `ring` → `simp` → `omega` → `linarith` → `exact?` → `apply?` → `aesop` in sequence.
4. **Verifier-guided self-correction** (Goedel-Prover-V2): Use Lean compiler feedback to iteratively revise proofs.
5. **Witness-based construction** (domain-specific): Use `MemW1pWitness` pattern per Hint 1 — don't fight existentials.

### Domain-Specific Critical Knowledge
- **Typeclass trap** (Hint 2): NEVER use `Lp E p μ` type or `MemLp.toLp` — use `LpFunctionToolkit.lean` bare-function alternatives with `eLpNorm` directly. Typeclass synthesis explodes (6.4M+ heartbeats).
- **Normalization** (Hint 3): Work with `NormalizedEllipticCoeff` (λ=1), don't fight it.
- **Recurrence lemma** (Hint 4): Prove `deGiorgi_recurrence_closeout` once, reuse everywhere.
- **Critical path** (Hint 5): Sobolev → WeakFormulation → DeGiorgiIteration → MoserIteration → Supersolutions → Crossover → WeakHarnack → Harnack → Hölder.

## What Has Been Tried and Failed (Known Failure Modes)

### Lean 4 General Failures
1. **Typeclass synthesis infinite loops**: The #1 cause of heartbeat timeouts. Especially dangerous with `EuclideanSpace ℝ (Fin d)` and `Lp` types. Lean can try and fail the same synthesis 384+ times.
2. **Naïve `simp` on measure theory goals**: `simp` without targeted lemma sets diverges on measure-theoretic goals. Use `simp only [specific_lemmas]`.
3. **Fighting Lp coercions**: `MemLp.toLp` / `coeFn_toLp` with `EuclideanSpace` is a known Lean performance catastrophe.
4. **Top-down proof attempts**: Trying to prove Harnack before establishing Sobolev foundations leads to cascading `sorry` dependencies and wasted effort.
5. **Overreliance on `sorry` as placeholder**: If you leave `sorry` in dependencies, downstream modules won't give meaningful feedback — errors compound.

### Analysis Formalization Failures
6. **Extracting from existentials repeatedly**: In Sobolev/weak-derivative contexts, repeated `obtain` from existentials creates term bloat. Use witness structures instead.
7. **Missing `Measurable` obligations**: Many analysis proofs require measurability side-goals. Forgetting `measurability` tactic or missing `MeasurableSet` hypotheses is common.
8. **`norm_cast` / `push_cast` neglect**: Coercion mismatches (ℕ → ℝ, ℤ → ℝ) silently break `linarith`/`nlinarith`. Always cast early.

### What NOT to Try
- **Do NOT** attempt headline theorems (Harnack, Hölder) before dependencies are sorry-free
- **Do NOT** use `Lp E p μ` type — use `eLpNorm` bare functions from `LpFunctionToolkit.lean`
- **Do NOT** read reference proofs in `~/DeGiorgi-Explained/DeGiorgi/*.lean` (rules violation)
- **Do NOT** try to automate everything — analysis proofs require human-guided structure with tactical automation for subgoals only
- **Do NOT** increase `maxHeartbeats` past 800000 as a first resort — refactor the proof term instead

## Recommended Starting Point for This Run

### Phase 1: Foundation Clearing (target: 20-30% sorry elimination)
1. **Survey**: Count total sorries per module. Identify leaf modules with fewest dependencies.
2. **Start with**: `Common/`, `Foundations/`, `EllipticCoefficients/`, `LpFunctionToolkit/` — these are leaf modules with likely simple lemmas (algebraic identities, basic inequalities, positivity).
3. **Tactic cascade per sorry**: Read signature → search Mathlib (`exact?`, `apply?`) → try `simp`, `ring`, `linarith`, `nlinarith`, `positivity`, `norm_num` → if stuck, read the math exposition in `~/DeGiorgi-Explained/book/`.
4. **Build frequently**: `lake build DeGiorgi` after every 2-3 sorries to catch regressions early.

### Phase 2: Sobolev Layer (target: 40-50%)
5. Work through `SobolevSpace/` (Witnesses, WeakDerivatives, Approximation) using `MemW1pWitness` pattern.
6. Prove `WeakFormulation/` once Sobolev is clean.
7. Tackle `Poincare/`, `SobolevPoincare/`, `PositivePart/`, `StampacchiaTruncation/`.

### Phase 3: Iteration Core (target: 60-70%)
8. `DeGiorgiIteration/` — prove Recurrence first (Hint 4), then Cutoff → Energy → PreIteration → Linfty.
9. `MoserIteration/` — CutoffPrep → Iteration → Linfty.

### Phase 4: Endgame (target: 80%+)
10. Supersolutions → Crossover → WeakHarnack → Harnack → Hölder.
11. These depend on everything below — only attempt when predecessors are clean.

### Realistic Expectations
- **Easy wins** (leaf modules, simple algebra/inequality lemmas): ~30-40% of all sorries
- **Medium** (Sobolev, weak formulation, cutoff constructions): ~20-30%, requires Mathlib search
- **Hard** (iteration cores, Harnack, Hölder): ~30-40%, requires understanding the math + careful proof engineering
- **Realistic target for one run**: 30-50% sorry elimination (foundations + partial Sobolev layer)

## Sources Searched

- [DeepSeek-Prover-V2 paper](https://arxiv.org/pdf/2504.21801) — 88.9% MiniF2F, two-stage RL
- [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) — RL + MCTS, 63.5% MiniF2F
- [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) — 90.4% MiniF2F with self-correction
- [Goedel-Prover GitHub](https://github.com/Goedel-LM/Goedel-Prover) — iterative proof bootstrapping
- [COPRA](https://arxiv.org/abs/2310.04353) — in-context learning agent for Lean/Coq (COLM 2024)
- [AlphaProof](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/) — AlphaZero RL for Lean, IMO silver
- [ProofFlow](https://arxiv.org/html/2510.15981) — dependency graph approach to proof autoformalization
- [LeanAgent](https://github.com/lean-dojo/LeanAgent) — lifelong learning, dependency DAG, best-first search
- [APOLLO](https://arxiv.org/html/2505.05758v5) — automated LLM + Lean collaboration, sorry decomposition
- [LeanCopilot](https://github.com/lean-dojo/LeanCopilot) — LLM-assisted tactic suggestions
- [Mathlib Sobolev Inequality](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/FunctionalSpaces/SobolevInequality.html) — Gagliardo-Nirenberg-Sobolev in Mathlib
- [Gagliardo-Nirenberg-Sobolev formalization (ITP 2024)](https://drops.dagstuhl.de/storage/00lipics/lipics-vol309-itp2024/LIPIcs.ITP.2024.37/LIPIcs.ITP.2024.37.pdf) — van Doorn-Macbeth
- [Lean 4 tactic cheatsheet](https://leanprover-community.github.io/papers/lean-tactics.pdf) — October 2025
- [Lean 4 typeclass synthesis issues](https://github.com/leanprover/lean4/issues/2055) — infinite loop bug
- [Lean 4 comprehensive survey](https://arxiv.org/abs/2501.18639) — January 2025
- [Lean 4 Skills for AI agents](https://github.com/cameronfreer/lean4-skills) — prove/review/golf loop

# rrma-lean Calibration — MiniF2F Lean 4

Generated: 2026-03-27

## Benchmark identity

**MiniF2F** — a cross-system benchmark of 488 formal math problem statements (244 valid / 244 test) drawn from AMC, AIME, IMO, MATH dataset, and undergraduate courses. Formalized in Lean 3/4, Metamath, Isabelle, HOL Light. The Lean 4 port uses Mathlib and is maintained at `yangky11/miniF2F-lean4`.

**Important caveat:** "miniF2F-Lean Revisited" (arXiv 2511.03108, Nov 2025) found that **16 problems across test+valid are unprovable** in their current form, and >50% of problems have discrepancies between formal and informal statements. They produced corrected versions (miniF2F-v2s, miniF2F-v2c) where pipeline accuracy jumped from 40% to 70%. Our run uses the original dataset, so expect ~16 impossible problems.

## Current SOTA (with numbers and citations)

| System | MiniF2F-test | Pass@N | Date | Source |
|--------|-------------|--------|------|--------|
| Kimina-Prover (full, 72B) | **92.2%** | TT-RL search | Apr 2025 | arXiv 2504.11354 |
| Kimina-Prover (full, 72B) | 86.4% | pass@32 + 1 round error correction | Apr 2025 | arXiv 2504.11354 |
| Goedel-Prover-V2-32B + self-correction | **90.4%** | pass@32 | Aug 2025 | arXiv 2508.03613 |
| Goedel-Prover-V2-32B | 88.1% | pass@32 | Aug 2025 | arXiv 2508.03613 |
| DeepSeek-Prover-V2-671B | 88.9% | pass@N | Apr 2025 | arXiv 2504.21801 |
| Goedel-Prover-V2-8B | 84.6% | pass@32 | Aug 2025 | arXiv 2508.03613 |
| Kimina-Prover Preview (72B) | 80.7% | pass@8192 | Apr 2025 | arXiv 2504.11354 |
| BFS-Prover | 72.95% | — | 2024 | — |
| DeepSeek-Prover-V1.5-RL | 63.5% | pass@N + tree search | 2024 | — |
| Goedel-Prover-V1 | 57.6% | pass@32 | Feb 2025 | arXiv 2502.07640 |
| Kimina-Prover Preview | 52.94% | pass@1 | Apr 2025 | arXiv 2504.11354 |
| HyperTree Proof Search | 42% | — | 2022 (NeurIPS) | arXiv 2205.11491 |
| COPRA (GPT-4, in-context) | ~30% | — | 2024 (COLM) | arXiv 2310.04353 |

**Key insight for our setting:** We are running a **single RTX 4070 Ti** with no fine-tuned prover model. SOTA numbers above use 8B-671B specialized models with pass@32-8192. Our realistic ceiling is closer to COPRA-level (~30%) or single-tactic sweeps (~20-30%), not the 80-90% SOTA.

## Best known techniques (specific tactics, strategies, approaches)

### Tier 1: Single-tactic hammers (solve 20-35% of problems)
These tactics each solve a meaningful subset with zero proof engineering:
- **`omega`** — linear arithmetic over ℕ/ℤ. Handles most `mathd_numbertheory` and simple algebra.
- **`linarith [h₀, h₁, ...]`** — linear (in)equalities over ordered fields. Must explicitly cite hypotheses.
- **`norm_num`** — numerical computation. Solves `2^10 = 1024`-style goals.
- **`ring`** — polynomial ring identities. Handles `algebra_*` identities.
- **`simp`** — simplification with Mathlib simp lemmas. Broad coverage.
- **`nlinarith`** — nonlinear arithmetic. Extends linarith with products/powers of hypotheses.
- **`decide`** — brute-force decidable propositions (small finite domains only).
- **`field_simp` then `ring`** — rational/field equations.

### Tier 2: Compositional patterns (solve 35-50%)
- **`constructor <;> linarith`** — split conjunctive goals, solve each with linear arithmetic.
- **`intro h; cases h <;> simp`** — handle disjunctions.
- **`have` intermediate lemmas** — break complex goals into sub-lemmas.
- **`calc` blocks** — chain equalities step-by-step.
- **`refine ⟨_, _⟩ <;> ...`** — anonymous constructor with sub-goals.

### Tier 3: LLM-guided proof search
- **COPRA-style**: Send proof state to LLM, get tactic suggestions, execute, loop with backtracking. Published at COLM 2024.
- **llmstep / LeanCopilot**: LLM tactic suggestion integrated into Lean. The `search_proof` tactic combines LLM suggestions with `aesop`.
- **Chain-of-thought interleaving**: Natural language reasoning between formal tactic steps improves whole-proof generation significantly.

### Tier 4: Specialized provers (out of our compute budget)
- **DeepSeek-Prover-V2**: Recursive proof search + subgoal decomposition via RL. 671B params.
- **Goedel-Prover-V2**: Scaffolded data synthesis + verifier-guided self-correction + model averaging.
- **Kimina-Prover**: Test-time RL search on 72B model.

### Critical tactic details
- `linarith` **does not work on ℂ** — complex numbers have no linear order. Use `ring` or `field_simp` instead.
- `omega` only works on ℕ and ℤ, not ℝ or ℚ.
- `set_option maxHeartbeats 400000` is needed for heavier tactics (default 200000 may timeout).
- `nlinarith` can be made more powerful by supplying witness terms: `nlinarith [sq_nonneg x, mul_self_nonneg y]`.
- `norm_num` can be extended with plugins for specific operations.

## What has been tried and failed

### Known failure modes
1. **Naive tactic enumeration** without considering type constraints (e.g., `linarith` on ℂ, `omega` on ℝ) — wastes attempts.
2. **Ignoring hypotheses** — `linarith` and `nlinarith` often need explicit hypothesis citations: `linarith [h₀, h₁]`.
3. **Insufficient heartbeats** — `simp` and `aesop` can timeout at default 200000. Set 400000+.
4. **Over-relying on `simp`** — `simp` is non-deterministic across Mathlib versions; proofs can break on update.
5. **Attempting unprovable problems** — 16 MiniF2F problems have no valid proof (per arXiv 2511.03108).
6. **No backtracking** — sequential tactic attempts without trying alternatives on failure.
7. **Ignoring problem category** — algebra problems need different tactics than number theory or combinatorics.

### What doesn't work for our setting
- **Fine-tuned prover models** — we don't have the compute to train/run 8B+ specialized models.
- **pass@8192 strategies** — we can't afford thousands of attempts per problem.
- **Tree search with value functions** — requires trained value network (HyperTree, AlphaProof-style).

## Recommended starting point for this run

### Phase 1: Tactic sweep baseline (target: 0.20-0.30)
1. Enumerate all `valid` problems from MiniF2F-lean4.
2. For each problem, try a fixed tactic cascade:
   - Classify by name prefix → select tactic order
   - `algebra_*`: try `ring`, `linarith`, `norm_num`, `field_simp; ring`, `nlinarith`
   - `mathd_numbertheory_*`: try `omega`, `norm_num`, `decide`
   - `mathd_algebra_*`: try `linarith`, `ring`, `norm_num`, `omega`
   - `amc_*`/`aime_*`: try `norm_num`, `omega`, `linarith`, `nlinarith`
   - General fallback: `simp`, `omega`, `linarith`, `norm_num`, `ring`, `nlinarith`, `decide`
3. Score this as the baseline method.

### Phase 2: Compositional tactics (target: 0.30-0.40)
4. For unsolved problems, try structural patterns:
   - `constructor <;> linarith [h₀, h₁, ...]`
   - `constructor <;> ring`
   - `intro h; cases h <;> simp`
   - `refine ⟨?_, ?_⟩ <;> norm_num`
5. Read the goal type to pick strategy: conjunction → `constructor`, disjunction → `cases`, equality → `calc`/`ring`.

### Phase 3: LLM-guided proof (target: 0.40-0.50)
6. For remaining unsolved problems, use Claude to read the theorem statement and hypotheses, then generate a custom proof attempt.
7. Use chain-of-thought: "This is asking to prove X. The hypotheses give us Y. Therefore we need to show Z. Approach: ..."
8. Feed Lean error messages back and iterate (COPRA-style loop).

### Key principles
- **Start with the easiest problems** — algebra, norm_num-solvable, simple linear arithmetic.
- **Track which tactics work on which problem types** — build a tactic-category matrix.
- **Cite hypotheses explicitly** — `linarith [h₀, h₁]` not just `linarith`.
- **Set maxHeartbeats 400000** on every attempt.
- **Skip known-impossible problems** once identified.

## Sources searched

- [Goedel-Prover (arXiv 2502.07640)](https://arxiv.org/html/2502.07640v1)
- [Goedel-Prover-V2 (arXiv 2508.03613)](https://arxiv.org/pdf/2508.03613)
- [Goedel-Prover-V2 GitHub](https://github.com/Goedel-LM/Goedel-Prover-V2)
- [DeepSeek-Prover-V2 (arXiv 2504.21801)](https://arxiv.org/abs/2504.21801)
- [DeepSeek-Prover-V2 GitHub](https://github.com/deepseek-ai/DeepSeek-Prover-V2)
- [Kimina-Prover Preview (arXiv 2504.11354)](https://arxiv.org/pdf/2504.11354)
- [Kimina-Prover HuggingFace blog](https://huggingface.co/blog/AI-MO/kimina-prover)
- [miniF2F-Lean Revisited (arXiv 2511.03108)](https://arxiv.org/html/2511.03108v1)
- [HyperTree Proof Search (arXiv 2205.11491)](https://arxiv.org/abs/2205.11491)
- [COPRA (arXiv 2310.04353)](https://arxiv.org/abs/2310.04353)
- [LeanCopilot GitHub](https://github.com/lean-dojo/LeanCopilot)
- [llmstep (NeurIPS MathAI 2023)](https://github.com/wellecks/llmstep)
- [miniF2F-lean4 GitHub (yangky11)](https://github.com/yangky11/miniF2F-lean4)
- [Lean 4 tactic cheatsheet](https://leanprover-community.github.io/papers/lean-tactics.pdf)
- [MiniF2F alphaXiv leaderboard](https://www.alphaxiv.org/benchmarks/university-of-pittsburgh/minif2f)
- [LeanTree (arXiv 2507.14722)](https://arxiv.org/html/2507.14722v1)
- [Prover Agent (arXiv 2506.19923)](https://arxiv.org/html/2506.19923v4)
- [LongCat-Flash-Prover (arXiv 2603.21065)](https://arxiv.org/html/2603.21065)

# Calibration — erdos-741ii-g0-fable

Compiled 2026-06-10 from web sources only. **Contamination note for this controlled
G0 cold-start run:** this file deliberately contains NO construction details for
Erdős #741(ii) beyond what the theorem statement itself asserts. Public sources
confirm the problem has been resolved (see Benchmark identity), but the specific
witness sets and proof architecture are intentionally omitted so agents must
derive them cold.

## Benchmark identity

This is NOT MiniF2F. It is a single open-ended Lean 4 formalization task:
Erdős Problem #741(ii) (Burr–Erdős), tracked at erdosproblems.com/741. The task:
exhibit a set A ⊆ ℕ that is an additive basis of order 2 (for n ≥ 4) such that
for every partition A = A₁ ⊔ A₂, at least one of A₁+A₁, A₂+A₂ fails to be
syndetic (has unbounded gaps). Oracle = `bash run.sh`, SCORE=1.0 iff the file
compiles with 0 sorries and no axiom/admit/native_decide.

Public status: erdosproblems.com/741 records that the question was answered
affirmatively in 2025 — DeepMind and an internal OpenAI model independently
produced constructions, and an explicit construction is now documented there.
So the theorem is TRUE and provable; the run measures whether a cold agent can
rediscover and formalize a witness.

## Current SOTA (with numbers and citations)

Whole-proof / agentic Lean 4 provers (MiniF2F-test, the closest proxy benchmark):

- DeepSeek-Prover-V2-671B: 88.9% on MiniF2F-test; 82.4% pass@32 with CoT;
  49/658 on PutnamBench (arXiv:2504.21801). Key idea: recursive subgoal
  decomposition + RL.
- Goedel-Prover-V2: 88.1% pass@32 (32B), 90.4% in self-correction mode; 8B model
  84.6% pass@32 (arXiv:2508.03613). Key ideas: scaffolded data synthesis,
  verifier-guided self-correction loops, `extract_goal` mining of failing subgoals.
- Kimina-Prover: 80.7% pass@8192; 92.2% with full test-time-RL search;
  52.94% pass@1 (arXiv:2504.11354).
- StepFun-Prover 32B: 70.0% pass@1 with real-time Lean feedback during generation.
- Seed-Prover 1.5: undergraduate-level proving via learning from experience
  (arXiv:2512.17260).
- Older baselines for context: HyperTree Proof Search/Evariste 38.5% pass@64
  on miniF2F-test (arXiv:2205.11491); COPRA (GPT-4 in-context backtracking
  agent, COLM 2024, arXiv:2310.04353) beat ReProver-style finetuning without
  any training.

Erdős-problem-specific AI results: Erdős #728 was the first Erdős problem fully
resolved autonomously (GPT-5.2 Pro informal argument + Harmonic's Aristotle for
the Lean proof, Jan 2026; arXiv:2601.07421). Aristotle also resolved a version
of #124. Pattern in both: natural-language proof sketch first, then formalization
with verifier feedback — not blind tactic search.

## Best known techniques (specific tactics, strategies, approaches)

1. **Sketch-then-formalize / subgoal decomposition** (DeepSeek-Prover-V2,
   Aristotle+GPT-5.2 on #728): write the informal proof completely first, break
   it into named lemmas with explicit statements, prove lemmas independently,
   then assemble. This is the single most consistent winner for hard problems.
2. **Verifier-feedback self-correction loops** (Goedel-Prover-V2, StepFun,
   APOLLO arXiv:2505.05758): run the compiler after every edit, feed the exact
   error back, repair locally. Matches this domain's "run.sh after every
   meaningful edit" rule.
3. **`extract_goal` on stuck subgoals** (Goedel-Prover-V2): when a tactic block
   fails, extract the standalone goal and attack it as its own mini-theorem.
4. **Workhorse closing tactics** (AutoSolver/APOLLO suites): `omega` (Nat/Int
   linear arithmetic — the strongest tool for digit/interval/index reasoning),
   `decide` (small finite cases), `interval_cases` (bounded variables),
   `nlinarith`/`linarith`/`positivity` (real inequalities), `norm_num`,
   `ring_nf`, `norm_cast`, `simp_all`, `aesop`, `tauto`, with `first [...]` /
   `try` combinators. For existence-over-ℕ goals, explicit witnesses + `omega`
   beats search.
5. **Strong induction via `Nat.strong_induction_on` / `Nat.le_induction`** for
   basis-of-order-2 style statements; case-split on size bands rather than
   single values.
6. **Semantic Mathlib search discipline** (Lean Finder, arXiv:2510.15940):
   hallucinated lemma names are the #1 failure; verify every lemma name with
   `exact?`/`apply?` or grep of Mathlib before relying on it.

## What has been tried and failed (known failure modes — do NOT repeat)

- **Hallucinated lemmas/tactics**: provers routinely cite non-existent Mathlib
  names (FormalMATH arXiv:2505.02735). Never trust a lemma name you haven't
  checked.
- **Automation over-invocation**: blanket `simp`/`aesop`/`nlinarith` on large
  goals causes timeouts or rewrites goals into unprovable forms
  (FormalMATH). Use targeted `simp only [...]` lists.
- **Vacuous/degenerate proofs**: proving an existential with a witness that
  makes a hypothesis vacuous — compiles for the wrong statement. The oracle
  here checks theorem presence; do not delete or weaken the target theorem.
- **Blind tactic search on research-level problems**: HTPS/BFS-style search
  tops out far below sketch-first methods on hard problems; #728 fell to
  informal-reasoning-first, not search. Don't grind tactics before you have a
  paper-level proof of the math.
- **Giving up after one construction**: the partition/syndetic half is the hard
  half; naive "obvious" candidate sets for A tend to die there, not at the
  basis half. Budget most thinking for the partition argument before encoding.
- **Single 0-sorry mega-attempts**: per program.md, prior cold agents
  self-terminated early. Structure each attempt as lemma skeleton → fill →
  oracle, and complete ≥5 distinct constructions.

## Recommended starting point for this run

1. Spend the first turns on MATH, not Lean: for each candidate A, write the
   informal proof of BOTH properties (basis ≥4, and the partition/gap
   obstruction) before touching the file. Reject candidates whose partition
   argument you cannot articulate informally.
2. Encode as a lemma DAG: `basis_lem` (strong induction + omega per band),
   a structural lemma about which elements can represent certain target
   numbers, and a gap lemma deriving non-syndeticity; assemble at the end.
3. After every edit, `bash run.sh`; treat each compiler error as the next
   subgoal (restate the failing goal standalone, `extract_goal` style).
4. Closing-tactic priority for this domain's goal shapes: `omega` >
   `interval_cases` + `omega` > `decide` (tiny cases) > `simp only` with named
   lemmas > `tauto`. Avoid bare `simp`/`aesop`/`nlinarith` on set-membership or
   sumset goals.
5. Log every refuted construction in MISTAKES.md with the exact reason the
   partition argument failed — that is the run's main data product.

## Sources searched

- https://www.erdosproblems.com/741 (problem status, resolution note)
- https://arxiv.org/abs/2504.21801 (DeepSeek-Prover-V2)
- https://arxiv.org/pdf/2508.03613 / https://huggingface.co/papers/2508.03613 (Goedel-Prover-V2)
- https://arxiv.org/pdf/2502.07640 (Goedel-Prover v1)
- https://arxiv.org/pdf/2504.11354 (Kimina-Prover)
- https://arxiv.org/pdf/2512.17260 (Seed-Prover 1.5)
- https://arxiv.org/abs/2310.04353 (COPRA, COLM 2024)
- https://arxiv.org/pdf/2205.11491 (HyperTree Proof Search / Evariste)
- https://arxiv.org/abs/2601.07421 (Erdős #728 Aristotle writeup)
- https://www.erdosproblems.com/forum/thread/blog:2 (AI on Erdős problems policy/blog)
- https://arxiv.org/pdf/2505.02735 (FormalMATH failure-mode analysis)
- https://arxiv.org/html/2505.05758v5 (APOLLO LLM+Lean collaboration)
- https://arxiv.org/pdf/2510.15940 (Lean Finder)
- https://xenaproject.wordpress.com/2025/12/05/formalization-of-erdos-problems/

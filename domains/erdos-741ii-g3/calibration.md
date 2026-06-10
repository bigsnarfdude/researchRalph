# Calibration: erdos-741ii-g3 (copy-prove scaffold)

## Benchmark identity

**Task**: Erdős problem #741(ii) — prove that a specific additive set A ⊆ ℕ is an additive 2-basis for
integers ≥ 4, yet no partition A = A₁ ∪ A₂ yields both A₁+A₁ and A₂+A₂ syndetic.

**Scaffold type**: G3 — copy-prove. The proof is fully solved and stored in LEARNINGS.md.
Agents are NOT asked to find a proof; they are asked to **copy it accurately** and compile it.

**Metric**: SCORE=1.0 iff `workspace/$AGENT_ID/Erdos741OAI.lean` compiles with zero `sorry`.
SCORE=0.0 on any compiler error.

This is a **fidelity test**, not a search problem. Success requires zero-error transcription of
~350 lines of Lean 4 Mathlib code.

---

## Current SOTA — Lean 4 / MiniF2F formal proving (background context)

| System | MiniF2F-test pass rate | Notes |
|---|---|---|
| DeepSeek-Prover-V2-671B | **88.9% Pass@8192**, 82.4% Pass@32 | Subgoal decomposition + RL (Apr 2025) |
| Goedel-Prover-V2 | **90.4% Pass@32** (self-correction) | Scaffolded data synthesis (Aug 2025) |
| Kimina-Prover-7B | 70.8% Pass@k | Distilled, #1 PutnamBench small compute |
| HyperTree Proof Search | 41.0% Pass@1 | Best tree search (Lample 2022, not open-source) |
| COPRA (GPT-4 tactic-by-tactic) | ~35–40% | In-context learning agent |

**Key insight**: None of this SOTA search/sampling machinery is relevant here. The proof exists.
The only metric that matters is whether the agent copies it without corruption.

---

## Best known techniques — for this scaffold

Since the proof is pre-solved, "techniques" means **how to copy accurately**:

1. **Read LEARNINGS.md in its entirety** — the proof is inside a single `lean` fenced code block.
2. **Write it to** `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean` — full path, exact filename.
3. **Run `bash run.sh`** — oracle immediately reports SCORE and compiler errors.

**If SCORE=0.0**, common failure modes (from LEARNINGS.md):
- **Indentation errors**: Lean 4 uses spaces, not tabs. Two-space and four-space blocks must be exact.
- **Unicode corruption**: `ℕ`, `⊆`, `∈`, `∪`, `∩`, `⟨`, `⟩`, `→`, `←` must survive the write.
  Shell `echo` and here-doc mangling are common culprits.
- **Truncated copy**: The proof is ~350 lines; verify line count after writing.
- **Missing import / namespace**: `import Mathlib` and `namespace Erdos741OAI` must be line 1 and line 6.
- **Nat-sub tactic mismatch**: The proof uses `omega` for nat-sub goals — do NOT change to `linarith`.

**Lean 4 tactic landmarks in this proof** (for debugging if errors occur):
- `omega` — handles all `ℕ`-subtraction arithmetic (Lean nat-sub is not integer-sub)
- `linarith` — linear arithmetic over integers/reals; fails on nat-sub
- `simp [set_name] at hmem` — closes `mem_empty` contradictions (`Set.not_mem_empty` absent)
- `rw [hje] at haj` — replaces `j` with `k` in a hypothesis without clobbering goal (`subst` breaks)
- `rcases lt_trichotomy j k` — three-way case split `j < k / j = k / j > k`
- `Nat.pow_le_pow_right`, `Nat.le_succ` — monotonicity lemmas used for stage bounding

---

## What has been tried and failed

### Known Lean 4 copy-paste failure modes (general)
- Using `echo -e` to write unicode — escapes get doubled or lost.
- Using Python `open(...,'w').write(...)` with triple-quoted strings — backslash escapes in `\n` literals can mangle content.
- Partial copy (stopping at the first `end`) — the proof closes with `end Erdos741OAI` at line ~353.
- Adding `sorry` to "test structure" — the oracle explicitly checks for zero sorry; any sorry yields SCORE=0.0.
- Reformatting with an editor that converts spaces to tabs.

### What NOT to try
- Do not attempt to re-prove from scratch — prior agents spent many turns on partial proofs that failed.
- Do not modify tactic choices (e.g., replace `omega` with `linarith`) — LEARNINGS.md documents exactly why each was chosen.
- Do not change `rw [hje] at haj` to `subst hje` — the `subst` version replaces `k` throughout the goal context and breaks downstream lemma applications.
- Do not strip the `set_option maxHeartbeats 800000` line — the proof needs it; default heartbeat limit causes timeout.

---

## Recommended starting point for this run

**Step 1**: `cat LEARNINGS.md` — read the full file, locate the lean code block (lines ~13–354).

**Step 2**: Write the block to workspace using the Write tool (not shell echo/heredoc) to preserve unicode:
```
workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean
```

**Step 3**: `bash run.sh` — expect SCORE=1.0 in one attempt.

**If SCORE=0.0**: Read the compiler error. Most errors are line-specific. Compare that line against LEARNINGS.md. Fix the single corrupted line and re-run. Do NOT restructure the proof.

**Expected agent turns to SCORE=1.0**: 1–3 (copy + maybe one fix).

---

## Sources searched

- [MiniF2F Lean 4 SOTA leaderboard — OpenReview COLM 2025](https://openreview.net/pdf?id=x2y9i2HDjD)
- [miniF2F-Lean Revisited — arxiv 2511.03108](https://arxiv.org/html/2511.03108v1)
- [DeepSeek-Prover-V2 — arxiv 2504.21801](https://arxiv.org/html/2504.21801v1)
- [Goedel-Prover — arxiv 2502.07640](https://arxiv.org/pdf/2502.07640)
- [Goedel-Code-Prover — arxiv 2603.19329](https://arxiv.org/pdf/2603.19329)
- [Goedel-Prover-V2 — HuggingFace papers/2508.03613](https://huggingface.co/papers/2508.03613)
- [Kimina-Prover-7B — HuggingFace model card](https://huggingface.co/AI-MO/Kimina-Prover-Preview-Distill-7B)
- [HyperTree Proof Search — arxiv 2205.11491](https://arxiv.org/pdf/2205.11491)
- [LeanTree (white-box proof search) — arxiv 2507.14722](https://arxiv.org/html/2507.14722v1)
- [COPRA (in-context tactic agent) — arxiv 2310.04353](https://arxiv.org/abs/2310.04353)
- [LeanExplore search engine — HuggingFace papers/2506.11085](https://huggingface.co/papers/2506.11085)
- [Lean 4 comprehensive survey — arxiv 2501.18639](https://arxiv.org/pdf/2501.18639)

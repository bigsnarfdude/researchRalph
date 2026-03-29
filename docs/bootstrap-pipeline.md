# Bootstrap Pipeline: RRMA → Fine-tuned Lean Prover
## From Seat-of-Pants Discovery to Distilled Knowledge

**Date:** 2026-03-29
**Status:** v0.1 — building
**Inspired by:** AIMO-2 / OpenMathReasoning iterative bootstrapping (NVIDIA, arXiv:2504.16891)

---

## The Gap We're Closing

| System | oracle | filter | scale |
|--------|--------|--------|-------|
| OpenMathReasoning | Python exec | correctness | 3.2M solutions |
| DeepSeek Prover-V2 | Lean typecheck | pass@k | 671B params |
| **This pipeline** | `lean_verify` | compiles + novel tactic | Qwen 27B |

They had 4× L4 GPUs and NVIDIA compute. We have nigel (RTX 4070 Ti SUPER) and a better oracle loop.

---

## What RRMA Already Has (Seed Data)

From rrma-lean runs:

```
attempts/exp001/   → 244 .lean files (many verified)
attempts/exp002-8/ → additional attempts across generations
staged/attempts/   → curated proofs from handcrafted sessions
LEARNINGS.md       → 17+ discovered tactic patterns (Lean-specific)
MISTAKES.md        → dead ends, wrong approaches
blackboard.md      → raw discovery log
```

Each `.lean` file is a complete proof:
```lean
import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem algebra_2complexrootspoly_... (x : ℂ) :
    x ^ 2 + 49 = (x + 7 * Complex.I) * (x + -7 * Complex.I) := by
  have h : Complex.I ^ 2 = -1 := Complex.I_sq
  nlinarith [h]
```

This is the seed. LIMO showed 1000 high-quality samples unlock long-reasoning in an instruct model.

---

## Bootstrap Stages

### Stage 0 — Mine the seed (now)

```bash
python3 bootstrap/mine_proofs.py \
  --attempts-dir domains/rrma-lean/attempts/ \
  --staged-dir domains/rrma-lean/staged/ \
  --output data/seed_proofs.jsonl \
  --verify  # re-run lean_verify to confirm each proof still compiles
```

Output format (one JSON per line):
```json
{
  "problem_id": "algebra_2complexrootspoly_xsqp49eqxp7itxpn7i",
  "theorem_statement": "theorem algebra_2complexrootspoly_... (x : ℂ) :\n    x ^ 2 + 49 = ...",
  "proof": "by\n  have h : Complex.I ^ 2 = -1 := Complex.I_sq\n  nlinarith [h]",
  "full_lean": "import Mathlib\n...",
  "tactic_count": 2,
  "verified": true,
  "source": "exp001"
}
```

Quality filter (from AIMO-2 "novel and significant"):
- Must compile (`lean_verify` exit 0)
- Proof body ≠ `by exact?` or `by decide` alone (trivial)
- At least one non-trivial tactic (ring/linarith/nlinarith/omega/field_simp/linear_combination)

### Stage 1 — Format for SFT

Prompt template:
```
<|im_start|>system
You are an expert Lean 4 theorem prover using Mathlib.
Given a theorem statement, write a complete Lean 4 proof.
<|im_end|>
<|im_start|>user
Prove the following theorem in Lean 4:

```lean
import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

{theorem_statement}
```
<|im_end|>
<|im_start|>assistant
```lean
{full_lean}
```
<|im_end|>
```

### Stage 2 — Fine-tune Qwen 27B (stage-0 model)

```bash
# On nigel — Unsloth LoRA fine-tuning
python3 bootstrap/finetune.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \  # start small, validate loop
  --data data/seed_proofs.jsonl \
  --output models/lean-prover-stage0 \
  --epochs 3 \
  --lora-r 16
```

Start with 7B to validate the loop fast. Scale to 27B once pipeline confirmed.

### Stage 3 — Generate at scale

For each unsolved MiniF2F problem, generate N candidates:

```bash
python3 bootstrap/generate.py \
  --model models/lean-prover-stage0 \
  --problems data/minif2f_unsolved.jsonl \
  --n-candidates 16 \
  --temperature 0.8 \
  --output data/candidates_stage1.jsonl
```

### Stage 4 — Filter by lean_verify

```bash
python3 bootstrap/verify_filter.py \
  --candidates data/candidates_stage1.jsonl \
  --lean-project ~/miniF2F-lean4 \
  --output data/verified_stage1.jsonl \
  --workers 4
```

Only proofs that compile go to next stage. This is the oracle — no hallucination possible.

### Stage 5 — Fine-tune stage-1 model

Same as Stage 2 but on `seed + verified_stage1`. The model now knows what it learned from generating.

### The loop

```
Stage 0 seeds → stage-0 model
stage-0 model → 16 candidates/problem → lean_verify → stage-1 training data
stage-1 model → 16 candidates/problem → lean_verify → stage-2 training data
...
```

Each iteration the model solves more problems. Each solved problem becomes training data. Compounding.

---

## GenSelect (AIMO-2 insight, adapted)

After Stage 2: train a small selector model to pick the best candidate from N.

For Lean, the signal is richer than natural language:
- Does the proof compile? (hard filter)
- How many tactics? (shorter = often more generalizable)
- Does it use `sorry`? (reject)
- Does the tactic sequence match known-good patterns from LEARNINGS.md?

Simple version: generate 8 proof attempts, run all through lean_verify, take the shortest that compiles.

Advanced version: train Qwen to predict which candidate will compile before running verify (saves GPU time).

---

## Connection to RRMA

The bootstrap loop IS an RRMA run:

```
RRMA Gen N produces proof traces
→ mine_proofs.py extracts verified proofs
→ finetune.py trains stage-N model
→ generate.py runs stage-N model on unsolved problems
→ verify_filter.py confirms new proofs
→ new proofs go into rrma-lean's attempts/
→ RRMA Gen N+1 starts with better baseline
→ better traces → stage-(N+1) model
```

The gardener's job: decide when to run the fine-tuning step between RRMA generations.

---

## NVIDIA vs Us

| Dimension | OpenMathReasoning | This pipeline |
|-----------|------------------|---------------|
| Oracle | Python exec | lean_verify (stronger) |
| Scale | 540K problems, 3.2M solutions | MiniF2F 244 problems |
| Compute | 4×L4 cluster | 1×RTX 4070 Ti 16GB |
| Model | Qwen2.5-32B | Qwen2.5-7B/27B |
| TIR equivalent | Python code blocks | Lean tactics |
| Advantage | Scale | Formal correctness + RRMA discovery loop |

We win on a different axis: **the proofs are machine-verified**. No hallucination, no approximate answer matching. A compiled proof is a proof.

---

## Files to Build

```
bootstrap/
  mine_proofs.py       ← extract + verify seed proofs
  finetune.py          ← Unsloth LoRA training
  generate.py          ← batch inference from fine-tuned model
  verify_filter.py     ← lean_verify each candidate
  genselect.py         ← pick best from N candidates
  eval.py              ← score on MiniF2F test set
  pipeline.sh          ← orchestrate all stages
```

---

## Next Steps

1. `mine_proofs.py` — extract and verify all proofs from rrma-lean attempts/
2. Count how many verified proofs we have (estimate: 150-180 from exp001 baseline 65.16%)
3. Run `finetune.py` on 7B first — validate the loop works
4. Measure: does stage-0 model solve any new MiniF2F problems?
5. If yes: run full loop on 27B

**Key question:** Does 150 seed proofs unlock useful proof-generation in Qwen 7B?
LIMO says yes (1000 samples). We'll find out with 150.

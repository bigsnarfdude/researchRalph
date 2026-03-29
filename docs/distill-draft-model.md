# RRMA Distillation → Draft Model
## Gardener-Grown Curriculum for Lean Theorem Proving

**Date:** 2026-03-28
**Status:** Concept / v0.1
**Context:** rrma-lean running on nigel, MiniF2F benchmark, Qwen3.5-27B

---

## The Insight

Two paradigms competing on Lean theorem proving (MiniF2F):

| System | Architecture | Learns? |
|--------|-------------|---------|
| OpenProver | Fixed 8-action planner/worker | No — same recipe every run |
| RRMA v4 | Gardener rewrites program.md across generations | Yes — compounds |
| DeepSeek Prover-V2 | 671B params, RL-trained | Pre-trained, frozen |

**The bet:** Domain-specific 27B > general 671B because knowledge is distilled, not brute-forced.

---

## Architecture

### Phase 1 — RRMA generates the curriculum

Each generation produces self-curated training signal:

```
LEARNINGS.md     → what Lean tactics work on MiniF2F
MISTAKES.md      → dead ends, wrong paths, what wastes steps
meta-blackboard  → compressed domain knowledge
agent*.jsonl     → successful proof traces (theorem → proof)
blackboard.md    → raw discovery log
```

Key property: **curriculum is self-selected**. Only successful proofs enter training. RRMA's gardener already prunes; the dataset quality compounds across generations.

### Phase 2 — Fine-tune Qwen 27B on curriculum

Infrastructure already exists:
- Unsloth MLX (local, `~/unsloth-mlx/`)
- Nigel RTX 4070 Ti SUPER (16GB VRAM)
- HuggingFace pipeline (bigsnarfdude)

Training data format:
```
input:  theorem statement + Lean 4 context + Mathlib imports
output: successful proof (verified by lean_verify)
```

Negative examples from MISTAKES.md provide contrastive signal — model learns what NOT to try.

### Phase 3 — Qwen as draft, Claude as judge

```
Theorem → Qwen 27B (fast, cheap, domain-specialized)
        → candidate proof steps
        → lean_verify (ground truth, no hallucination possible)
        → if fail: Claude (expensive, general, handles hard cases)
        → lean_verify
```

Cost profile:
- 90% of easy theorems: Qwen handles alone
- 10% hard cases: Claude escalates
- lean_verify is always the arbiter

### The Compounding Loop

```
RRMA Gen 1 → traces → fine-tune Qwen v1
RRMA Gen 2 (uses Qwen v1 as draft) → better traces → fine-tune Qwen v2
RRMA Gen 3 (uses Qwen v2) → ...
```

Each generation the draft model gets stronger. Each generation RRMA's agents start from a higher floor.

---

## Why This Wins on Hard Theorems

OpenProver's planner picks from 8 pre-defined actions — forever. It cannot fire itself.

The gardener can rewrite `program.md`. Over 3+ generations it builds a domain-specific playbook:
- Which tactic sequences work for which theorem shapes
- When to use `ring` vs `linarith` vs `norm_num` vs `simp`
- Which Mathlib lemmas are useful for MiniF2F algebra/number theory
- How to decompose hard goals into sub-goals the model can handle

The fine-tuned Qwen encodes this playbook in weights. It becomes a specialized oracle for this benchmark.

---

## Connection to Existing Work

- **lambda_results**: Rank-1 LoRA direction captures 1D signal. Same principle — single fine-tuning direction on specialized data can be highly effective.
- **autointerp (0.991 AUROC)**: Diverse training data closed the OOD gap. Same principle applies here — RRMA's diverse theorem attempts cover the distribution.
- **mindreader (0.980 AUROC)**: Fine-tuned Gemma 3 27B beat base probes. Domain-specific fine-tuning of 27B models works.

The alignment faking detector and the Lean prover are the same shape of problem: **specialized 27B trained on self-curated signal beats general large models.**

---

## Next Steps

1. Get rrma-lean baseline (current Qwen 27B raw score on MiniF2F)
2. Run RRMA Gen 1-3, collect successful proof traces
3. Fine-tune Qwen 27B on traces (Unsloth, nigel)
4. Eval fine-tuned Qwen vs baseline vs OpenProver vs DeepSeek Prover-V2
5. If Qwen v1 > baseline: publish the loop, pitch as "self-improving Lean prover"

---

## The Pitch

> "We don't train a prover. We grow one."

RRMA's gardener designs the curriculum. Agents discover the tactics. The fine-tuned model encodes the knowledge. Claude handles the hard cases. Lean verifies everything.

No human labels. No hand-designed reward. The benchmark is the teacher.

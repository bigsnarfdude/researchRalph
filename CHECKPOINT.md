# rrma-lean Bootstrap — Checkpoint (2026-03-29)

## What We're Actually Doing

Signal test: do RRMA agent CoT traces improve smaller model theorem proving on MiniF2F?

**The experiment:**
1. Base model (Qwen2.5-7B-Instruct, no fine-tuning) → pass@8 on MiniF2F valid (244 problems)
2. Fine-tuned model (same base + LoRA on RRMA traces) → pass@8 on same problems
3. If fine-tuned > base → traces have signal → publish dataset on HuggingFace

The dataset is the deliverable, not the model. RRMA as a trace generator for open-source community.

---

## Current State (end of day 5)

### Fine-tune: COMPLETE ✓
- **Model**: `~/models/lean-prover-rrma-traces` on nigel
- **Data**: `~/data/rrma_lean_combined.jsonl` (1,513 records)
  - 417 thinking traces with `<think>...</think>` chains
  - 1,096 lean attempt files (no thinking chains)
- **Training**: 3 epochs, QLoRA 4-bit, lora_r=32, batch=1, grad_accum=16
- **Result**: train_loss=0.29, token_accuracy=97.3%, 32 min on RTX 4070 Ti

### Eval fine-tuned: RUNNING NOW
- Screen: `eval_finetuned` on nigel
- Log: `~/data/eval_finetuned.log`
- Output: `~/data/eval_finetuned.jsonl`
- Command: `eval_signal.py --model ~/models/lean-prover-rrma-traces --n-candidates 8`
- ETA: ~2-3 hours

### Eval base model: NOT STARTED YET
- Run after fine-tuned eval finishes
- Command (on nigel):
```bash
screen -dmS eval_base bash -c '
source ~/venv/bin/activate
python3 ~/researchRalph/bootstrap/eval_signal.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --minif2f ~/miniF2F-lean4 \
  --output ~/data/eval_base.jsonl \
  --n-candidates 8 --load-in-4bit \
  2>&1 | tee ~/data/eval_base.log
'
```

---

## Data Files on Nigel

| File | Description | Location |
|------|-------------|----------|
| `rrma_lean_combined.jsonl` | 1,513 SFT records (417 thinking + 1,096 attempts) | `~/data/` |
| `rrma_lean_traces.jsonl` | 417 thinking traces only | `~/data/` (local too) |
| `minif2f_statements.json` | 244 MiniF2F theorem statements | `~/data/` (local too) |
| `eval_finetuned.jsonl` | Fine-tuned eval results (in progress) | `~/data/` |
| `eval_base.jsonl` | Base model eval results (not started) | `~/data/` |
| `finetune_rrma_traces.log` | Fine-tune training log | `~/data/` |

---

## Data Files Local (researchRalph repo)

```
bootstrap/
  finetune.py          — SFT training script (QLoRA --load-in-4bit flag)
  eval_signal.py       — Signal test eval (base vs fine-tuned, lake verify)
  generate.py          — Generate candidates from fine-tuned model
  generate_opus_traces.py  — (future) Opus API trace generator
  verify_filter.py     — Parallel lake env lean verification
  mine_mathlib.py      — Mine Mathlib proofs (fixed COMPLEX_TACTICS bug)

tools/
  traces_to_sft.py     — Extract thinking traces from JSONL agent logs
                         (fixed: window size 20→200, test_* content matching,
                          --statements-json flag)

data/
  rrma_lean_traces.jsonl    — 417 thinking traces
  rrma_lean_combined.jsonl  — 1,513 combined
  rrma_lean_attempts.jsonl  — 1,096 attempt-only records
  minif2f_statements.json   — 244 problem statements
  rrma_lean_logs/           — 30 JSONL agent session logs (from nigel)
```

---

## Nigel Setup

- **SSH**: `ssh vincent@nigel.birs.ca`
- **GPU**: RTX 4070 Ti, 16GB VRAM
- **Venv**: `~/venv` (transformers==4.51.3, trl==0.16.1, peft==0.15.1, bitsandbytes)
- **MiniF2F**: `~/miniF2F-lean4` (lake build already done — don't redo, takes hours)
- **Lean**: `~/.elan/bin/lake` via `source ~/.elan/env`

---

## Key Bugs Fixed This Session

1. **mine_mathlib.py** — COMPLEX_TACTICS checked whole file (imports) not proof body → 0 results fixed to 3,167
2. **traces_to_sft.py window=20** — agents batch-write 20-50 files after one think → fixed to look back 200 events to session boundary
3. **traces_to_sft.py filename matching** — `test_*` files not matched to MiniF2F problems → fixed via theorem name inside content
4. **traces_to_sft.py no statements** — added `--statements-json` flag
5. **run.sh (gpt2)** — `set -e` + no background killed before TSV write → fixed
6. **finetune.py** — added `--load-in-4bit` QLoRA for 16GB VRAM

---

## What Happened (5 days summary)

- Day 1-2: GH200 setup, mine_mathlib fix, bootstrap pipeline, 3,167 seed proofs, Stage 0 fine-tune on GH200 (7B), generated 976 candidates, verified 138 new proofs
- Day 3: Realized wrong data — should use RRMA traces not Mathlib mining. Found 30 JSONL logs on nigel with agent sessions
- Day 4: Fixed traces_to_sft.py (window bug, filename bug) → recovered 417 thinking traces from 30 logs
- Day 5: Fine-tuned on 1,513 combined records. Signal eval running now.

---

## Next Steps After Eval

1. Compare `eval_finetuned.jsonl` vs `eval_base.jsonl` — count pass@8 for each
2. If signal confirmed → publish dataset to HuggingFace as `bigsnarfdude/rrma-lean4-traces`
3. If no signal → investigate why (too few traces? wrong format? model too small?)
4. Optional: run more RRMA agent sessions to collect more traces, retrain

---

## Quick Resume Commands

```bash
# Check eval status
ssh vincent@nigel.birs.ca "tail -5 ~/data/eval_finetuned.log"

# Run base model eval (after fine-tuned finishes)
ssh vincent@nigel.birs.ca "screen -dmS eval_base bash -c 'source ~/venv/bin/activate && python3 ~/researchRalph/bootstrap/eval_signal.py --model Qwen/Qwen2.5-7B-Instruct --minif2f ~/miniF2F-lean4 --output ~/data/eval_base.jsonl --n-candidates 8 --load-in-4bit 2>&1 | tee ~/data/eval_base.log'"

# Compare results when both done
ssh vincent@nigel.birs.ca "python3 -c \"
import json
def pass_at_k(f):
    per = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            per.setdefault(r['problem_id'], False)
            if r['passed']: per[r['problem_id']] = True
    n = sum(per.values()); t = len(per)
    print(f'{f}: {n}/{t} = {n/t:.1%}')
pass_at_k('/home/vincent/data/eval_base.jsonl')
pass_at_k('/home/vincent/data/eval_finetuned.jsonl')
\""

# Extract more traces if more agent sessions run
python3 tools/traces_to_sft.py data/rrma_lean_logs/ \
  --statements-json data/minif2f_statements.json \
  --out data/rrma_lean_traces.jsonl --min-thinking 100
```

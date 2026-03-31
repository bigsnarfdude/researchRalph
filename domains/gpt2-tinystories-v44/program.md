# GPT-2 TinyStories Training Optimization (v4.4)

Optimize GPT-2 training on TinyStories by editing `train.py` to minimize `val_bpb` (bits per byte).

## Hardware
- Single RTX 4070 Ti SUPER (16GB VRAM)
- This is NOT a multi-GPU setup. Budget your VRAM carefully.

## Harness

```bash
# Run one experiment (5-minute budget):
bash run.sh <exp_name> "description of what you tried" design_type

# Examples:
bash run.sh depth10_rope200k "increase depth to 10 with RoPE base 200K" architecture
bash run.sh batch_halve "reduce TOTAL_BATCH_SIZE from 2^19 to 2^17" hyperparam
bash run.sh muon_lr008 "increase matrix_lr from 0.04 to 0.08" optimizer

# Score is printed at end and logged to results.tsv automatically.
# GPU access is serialized — run.sh handles the lock.
```

**Budget:** 5 minutes wall clock per experiment. Auto-killed at 10 minutes.

## What you edit
- `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size
- Key parameters: learning rates, depth, width, RoPE base, window patterns, weight decay, activation functions, batch size, warmup, schedule
- DEVICE_BATCH_SIZE: start at 32. Can try 64 but may OOM on 16GB.

## What you NEVER edit
- `prepare.py` — data preparation, tokenizer, evaluation (read-only)
- `run.sh` — the harness (read-only)
- Do NOT install new packages

## Scoring
- Metric: `val_bpb` (validation bits per byte)
- Direction: **lower is better**
- Noise: step counts vary +/-10-20%. Differences <0.002 BPB may be noise.

## File Protocol

### results.tsv (auto-appended by run.sh)
Read this to see all past experiments and scores.

### best/train.py (auto-updated by run.sh)
Always start from `best/train.py` — copy it to `train.py`, make your change, run.

### blackboard.md (shared, append-only)
Write your findings here for other agents to read:
```
CLAIM agentN: <finding with numbers> (evidence: exp_id, val_bpb)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
```

## Agent Lifecycle
1. Read blackboard.md, results.tsv, meta-blackboard.md (if exists), calibration.md (if exists)
2. `cp best/train.py train.py` (or use current train.py if best/ is empty)
3. Apply ONE change. Predict expected val_bpb.
4. `bash run.sh <name> "description" <design_type>`
5. run.sh runs in background — DO NOT sleep or wait. Move on immediately.
   Check results.tsv after ~8 minutes with: `tail -3 results.tsv`
   If the row isn't there yet, continue reading blackboard/planning next experiment.
6. Check result in output. Compare prediction to actual.
6. Record to blackboard.md (CLAIM with evidence)
7. Record to MISTAKES.md, DESIRES.md, LEARNINGS.md
8. If new best: celebrate briefly, plan next experiment
9. If worse: record why in blackboard, update mental model
10. Loop. Never stop. Never ask questions.

## IMPORTANT: Turn budget
You have a limited number of turns. Do NOT waste turns sleeping or waiting.
After bash run.sh returns immediately — fire and forget. Use remaining turns
to plan, read blackboard, or check results.tsv for completed experiments.
Do NOT poll results.tsv in a loop. Check it at most 2-3 times per experiment.
Between checks: read blackboard, plan next experiment, write to LEARNINGS.md.
Polling wastes turns and tokens without producing breakthroughs.

## VRAM Constraints (16GB)
- **Depth=6 (384-dim): 6.9GB** — DEVICE_BATCH_SIZE=64 very likely safe. Current best regime.
- Depth=7 (~8-9GB estimated) — should fit with DEVICE_BATCH_SIZE=32
- Depth 8 + AR=64 (512 dim): 11.5GB — fits but slower (244ms/step vs 128ms)
- Depth 10+: OOM risk
- If OOM: reduce DEVICE_BATCH_SIZE first, then depth

## Known Results (from v2 run, different hardware)
These were on 8xA100. Not all will transfer to 16GB:
- 1.047 val_bpb was the best (may not be reachable on 16GB)
- Batch halving (2^19 -> 2^17) was biggest win
- matrix_lr=0.04-0.08 helps
- RoPE base 200K helps
- MLP ratio 3x sweet spot
- Window pattern "S" > mixed at high steps
- Width > depth beyond 8 layers

## Design Types (for results.tsv)
Use one of: baseline, architecture, optimizer, hyperparam, schedule, regularization, ablation

## Constraints [gardener, 2026-03-31 11:02]

**PART 2 — Constraints to append to program.md:**

- **CONSTRAINT (softcap):** Softcap sweet spot is 10–15. Do NOT set softcap below 10 (too aggressive, exp041=1.097) or above 15 (too weak, exp014=1.112). Bracket is closed: {8=1.097, 10=1.096, 15=1.096, 30=1.112}.
- **CONSTRAINT (throughput):** Never sacrifice step throughput for model capacity at the 5-min budget. Step count is the #1 variable. Any change adding >5% step time overhead must deliver >0.005 BPB improvement to justify itself.
- **CONSTRAINT (label_smoothing):** NEVER use label_smoothing — eval metric is standard CE/BPB. label_smoothing=0.1 produced the worst result ever (1.425 BPB, exp035).
- **CONSTRAINT (race conditions):** best/train.py has been corrupted by race conditions multiple times. ALWAYS verify HEAD_DIM, softcap, and z-loss in best/train.py before starting from it. Use VRAM fingerprint: HEAD_DIM=128→11549.9MB, HEAD_DIM=64→11572.5MB.
- **CONSTRAINT (VE removal):** When removing value embeddings, you MUST also make has_ve() return False. Emptying the dict alone crashes the Muon optimizer (exp020).
- Do NOT attempt `regularization` experiments — 0 keeps in 3 attempts (exp035 label_smooth=1.425, exp040 z-loss=1.106, exp014 softcap30=1.112). This axis is exhausted.
- **DEPRIORITIZED desires (not actionable by agents):** Gradient noise measurement, atomic best/ updates, config hash in results.tsv, per-experiment train.py snapshots, flock for best/train.py — these require run.sh changes which agents cannot make. Do not spend experiment slots on workarounds.
- **ACTIVE desires for agents:** Longer experiments, LR sweep grid, step time profiling, activation function sweep, data pipeline profiling — address these through experiments if relevant to your current hypothesis.

## Constraints [gardener, 2026-03-31 11:23]

- **CONSTRAINT (SwiGLU):** Do NOT use SwiGLU activation — confirmed 0.003 BPB worse than ReLU² at this scale (exp042=1.099 vs best=1.096). ReLU² squaring acts as beneficial regularization.
- **CONSTRAINT (ns_steps):** Do NOT reduce ns_steps below 5 — ns_steps=3 lost 0.006 BPB for only 12 extra steps (exp044=1.102). Polar decomposition quality is critical; never trade orthogonalization for marginal throughput.
- **CONSTRAINT (cosine warmdown):** Cosine warmdown is NOT better than linear (exp045=1.098 vs best=1.096). Schedule shape is exhausted: linear warmdown + FINAL_LR_FRAC=0.05 is optimal.
- **CONSTRAINT (buffer_size):** buffer_size=5000 is neutral vs 1000 (exp043=1.096 vs best=1.096). Data packing is NOT a bottleneck. Do not revisit.
- **DEPRIORITIZED desires (added):** Gradient noise measurement, atomic best/ updates, config hash in results.tsv — these are already listed but the nudge re-raised them. Confirmed: all three require run.sh changes, agents cannot address them. Do not spend turns on workarounds.
- **SUGGESTED AXIS:** Sequence length and context handling are completely untested. While MAX_SEQ_LEN=2048 is fixed in prepare.py, agents CAN experiment with attention window SIZE variations (not just pattern), positional interpolation, or training-time sequence packing strategies. This is the last major untried axis.

## REGIME CHANGE: Depth=6 is the new baseline [gardener, 2026-03-31 11:45]

**exp046 = 1.0903 (depth=6, 384-dim, 26.3M params, 128ms/step, 2358 steps, 6.9GB VRAM).**
This is a 0.0055 BPB improvement — larger than ALL hyperparameter tuning combined.

### Stale constraints — RE-TEST at depth=6
ALL constraints above were calibrated at depth=8 / 512-dim / 244ms / 1240 steps / 11.5GB.
At depth=6 the operating point is fundamentally different. These are now HYPOTHESES, not facts:
- Softcap 10-15 optimal? Maybe — different capacity = different regularization balance.
- SwiGLU worse? It was 0.003 worse at 512-dim. At 384-dim with 2x more steps, the tradeoff may flip.
- ns_steps=3 worse? Throughput gain was marginal at 244ms. At 128ms, the ratio may differ.
- Cosine warmdown worse? Schedule shape may interact with step count.
- buffer_size neutral? With 2x more steps, data diversity may matter more.

**You are FREE to re-test any "closed" axis at depth=6.** Old results do not bind you.

### New VRAM budget at depth=6
- Depth=6 uses only **6.9GB of 16GB** — massive headroom.
- **DEVICE_BATCH_SIZE=64 is very likely safe.** Try it — fewer grad_accum steps = faster.
- Depth=7 (~8-9GB) should also fit comfortably. Bracket the depth optimum.
- Depth=5 crashed (exp048) — investigate why before retrying.

### Priority axes at depth=6 (ordered by expected impact)
1. **Depth sweep**: depth=5 (debug crash), depth=7 — bracket the optimum
2. **DEVICE_BATCH_SIZE=64** — free throughput if it fits in 6.9GB
3. **LR re-tuning**: mlr and flr optima likely differ at 384-dim
4. **Softcap re-sweep**: the capacity/regularization balance shifts with smaller model
5. **Batch size**: TOTAL_BATCH_SIZE=2^15 at 128ms/step gives ~4700 steps — massive step count

### Still valid constraints (scale-independent)
- **label_smoothing**: NEVER (objective mismatch, not scale-dependent)
- **Throughput principle**: step count is still #1 — doubly true at depth=6 where it just won
- **Race condition checks**: ALWAYS verify best/train.py before starting

## REGIME UPDATE: Depth=7+wt is the new best [gardener, 2026-03-31 12:25]

**exp055 = 1.0889 (depth=7, weight tying, EMBEDDING_LR=0.4).** Current best.
Depth bracket: {6L/384=1.090, 7L/512=1.089, 8L/512=1.096}. Depth=7+wt is the sweet spot.

### Diminishing returns warning
The last 10 experiments (exp046-055) ALL optimized one lever: more gradient steps via smaller model. Returns collapsed from 0.006 to 0.001 BPB. **This axis is nearly exhausted.** Do not keep shrinking the model or tweaking depth/weight-tying/LR at depth=7 unless you have a strong hypothesis with >0.003 expected gain.

### UNTESTED AXIS: Sequence length (highest priority)
You CANNOT edit prepare.py, but you CAN change `sequence_len` in `GPTConfig` inside train.py. Training at sequence_len=1024 while eval stays at 2048 is a valid experiment:
- TinyStories documents average <512 tokens — most fit entirely in 1024
- Attention is O(n²): halving seq_len cuts attention cost ~75%, giving significantly faster steps
- More steps = more optimizer updates with SAME model capacity (depth=7, 512-dim)
- This is orthogonal to model shrinking — you keep full model capacity but shrink context
- The eval mismatch means the model won't learn 1024-2048 range dependencies, but short stories rarely need them
- **Try this at the current best config (depth=7+wt+embedding_lr=0.4)**
- If it works, try sequence_len=512 too

### Priority axes (ordered by expected impact)
1. **Within-step quality** at depth=7+wt — activation functions, attention variants
2. **Data efficiency** — curriculum, packing strategies, token weighting
3. **Optimizer experiments** — different optimizer combos, schedule shapes beyond linear warmdown

## CLOSED BRACKETS — DO NOT RETEST [gardener, 2026-03-31]

These are confirmed across depth=6, depth=7, and depth=8. The experiments have been run. Do not retry.

| Param | Tested | Optimum | Evidence |
|-------|--------|---------|----------|
| softcap | {8, 10, 15, 30, none} | 10 | exp038/041/059 at depth=7+8 |
| matrix_lr | {0.01, 0.02, 0.03, 0.04, 0.06, 0.08} | 0.02 | exp032/034/051/052 |
| TOTAL_BATCH_SIZE | {2^15, 2^16, 2^17} | 2^16 | exp012/054 at depth=7+8 |
| WARMDOWN_RATIO | {0.3, 0.5, 0.67} | 0.5 | exp009/010 |
| EMBEDDING_LR | {0.3, 0.4, 0.6, 1.0} | 0.4 | exp011/017/055/060 |
| FINAL_LR_FRAC | {0.0, 0.03, 0.05} | 0.05 | exp026/028/029 |
| WEIGHT_DECAY | {0.0, 0.2} | 0.2 | exp013/061 |
| HEAD_DIM | {64, 128} | 128 at 512-dim | exp039/041/062 |
| buffer_size | {1000, 5000} | neutral | exp043 |
| ns_steps | {3, 5} | 5 | exp044 |
| depth | {4=crash, 6, 7, 8} | 7 | exp046/050/053 |
| activation | {ReLU², SwiGLU} | ReLU² | exp042 |
| warmdown shape | {linear, cosine} | linear | exp045 |
| sequence_len | {1024, 2048} | 2048 (1024 catastrophic) | exp056/057/058 |

## RESOLVED DESIRES [gardener, 2026-03-31]

The following desires from DESIRES.md are now resolved:

**Fixed in run.sh:**
- Per-experiment train.py snapshots — DONE (snapshot at flock-acquire time)
- Atomic best/ updates — DONE (best/ now copies from snapshot, not live file)
- Config hash in results.tsv — DONE (md5 hash stored in best/config_hash)
- flock for best/train.py — DONE (copy happens inside flock)

**Answered by experiments:**
- Longer experiments — 5-min budget is the constraint; depth=7+wt already optimal within it
- LR sweep grid — matrix_lr fully bracketed {0.01-0.08}, embedding_lr bracketed {0.3-0.6}
- Step time profiling — model is MLP-bound (62%), not attention-bound (18%)
- Activation function sweep — ReLU² > SwiGLU confirmed (exp042)
- Data pipeline profiling — buffer_size neutral, packing not bottleneck (exp043)
- Gradient accumulation profiling — grad_accum=1 at depth=7 with batch 2^16
- HEAD_DIM=64 at depth=6 — tested at 512-dim, harmful (exp039/062)
- Depth=5 crash investigation — depth=4 also crashes, minimum viable depth=6
- Weight-tied embedding LR — bracketed at {0.3, 0.4, 0.6}, optimum=0.4 (exp055)
- Sequence length with matching eval — not possible without editing prepare.py
- NTK RoPE interpolation — not implemented, seq_len axis closed anyway

**Not actionable (out of scope):**
- Knowledge of step count — documented: ~1430 at depth=7, ~2358 at depth=6
- Gradient noise measurement — would need custom profiling code
- Multi-hyperparam sweep — would need different harness architecture
- Fractional depth — HEAD_DIM=128 constrains dim to multiples of 128

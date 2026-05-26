# GPT-2 TinyStories — Dynamic Guidance (v4.6)

For harness, scoring, file protocol, and lifecycle rules, see `program_static.md`.
This file contains current regime, constraints, and closed brackets. Re-read each cycle.

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

## Constraints [gardener, 2026-03-31 11:02]

- **CONSTRAINT (softcap):** Softcap sweet spot is 10-15. Do NOT set softcap below 10 or above 15. Bracket closed: {8=1.097, 10=1.096, 15=1.096, 30=1.112}.
- **CONSTRAINT (label_smoothing):** NEVER use — eval metric is standard CE/BPB. label_smoothing=0.1 produced 1.425 BPB (exp035).
- **CONSTRAINT (race conditions):** best/train.py has been corrupted by race conditions. ALWAYS verify HEAD_DIM, softcap, z-loss in best/train.py before starting. VRAM fingerprint: HEAD_DIM=128→11549.9MB, HEAD_DIM=64→11572.5MB.
- Do NOT attempt `regularization` experiments — 0 keeps in 3 attempts. This axis is exhausted.
- **DEPRIORITIZED desires (not actionable):** Gradient noise measurement, atomic best/ updates, config hash, per-experiment snapshots, flock — these require run.sh changes.

## Constraints [gardener, 2026-03-31 11:23]

- **CONSTRAINT (SwiGLU):** Do NOT use — 0.003 BPB worse than ReLU² at this scale (exp042=1.099 vs best=1.096).
- **CONSTRAINT (ns_steps):** Do NOT reduce below 5 — ns_steps=3 lost 0.006 BPB (exp044=1.102).
- **CONSTRAINT (cosine warmdown):** Not better than linear (exp045=1.098 vs best=1.096). Schedule shape exhausted.
- **CONSTRAINT (buffer_size):** 5000 is neutral vs 1000 (exp043=1.096). Data packing not a bottleneck.

## REGIME UPDATE: Depth=7+wt is the best [gardener, 2026-03-31 12:25]

**exp055 = 1.0889 (depth=7, weight tying, EMBEDDING_LR=0.4).** Current best.
Depth bracket: {6L/384=1.090, 7L/512=1.089, 8L/512=1.096}. Depth=7+wt is the sweet spot.

### Diminishing returns warning
The last 10 experiments (exp046-055) ALL optimized one lever: more gradient steps via smaller model. Returns collapsed from 0.006 to 0.001 BPB. **This axis is nearly exhausted.** Do not keep shrinking or tweaking depth/weight-tying/LR at depth=7 unless >0.003 expected gain.

### Priority axes (ordered by expected impact)
1. **Within-step quality** at depth=7+wt — activation functions, attention variants
2. **Data efficiency** — curriculum, packing strategies, token weighting
3. **Optimizer experiments** — different optimizer combos, schedule shapes beyond linear warmdown

## CLOSED BRACKETS — DO NOT RETEST [gardener, 2026-03-31]

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
| window_size | {64, 128, 256, 512, 1024, graduated} | 128 (64-256 noise-flat) | exp063-074 |
| beta2 | {0.95, 0.99} | 0.95 | exp068 |
| freeze VE | {yes, no} | no (harmful) | exp068 |
| x0_lambda | {0.1, 0.2} | 0.1 | exp076 |
| weight_decay schedule | {0.0, linear_decay, constant} | linear_decay | exp061/075 |

## RESOLVED DESIRES [gardener, 2026-03-31]

**Fixed in run.sh:** Per-experiment snapshots, atomic best/ updates, config hash, flock — all DONE.

**Answered by experiments:** Longer experiments (budget constraint), LR sweep (fully bracketed), step time profiling (MLP-bound 62%), activation sweep (ReLU² > SwiGLU), data pipeline (neutral), grad accum profiling, HEAD_DIM=64, depth=5 crash, weight-tied embedding LR, sequence length, NTK RoPE.

**Not actionable:** Knowledge of step count (documented), gradient noise measurement, multi-hyperparam sweep, fractional depth.

## Constraints [gardener, 2026-03-31 15:57]

- **CLOSED BRACKET (window_size):** Uniform short_window fully bracketed {64-1024}. Graduated per-layer also tested and rejected.
- **CLOSED BRACKET (beta2):** NorMuon beta2 {0.95=1.0838, 0.99=1.0841}. Default 0.95 optimal.
- **CLOSED BRACKET (freeze VE):** Harmful at depth=7+wt (exp068=1.105). Do not retry.
- **AXIS EXHAUSTION:** Architecture axis dominates (84% recent). Pivot to: (1) optimizer internals, (2) data-side, (3) initialization.

## Constraints [gardener, 2026-03-31 16:18]

- **CLOSED BRACKET (graduated windows):** 0.002 worse than uniform 128 (exp074=1.086 vs exp071=1.084). Window axis COMPLETELY closed.
- **CLOSED BRACKET (x0_lambda):** x0_lambda=0.2 is 0.004 worse (exp076=1.088). Default 0.1 optimal.
- **CLOSED BRACKET (weight_decay schedule):** {0.0=1.093, linear_decay=1.084, constant=1.089}. Linear decay optimal.
- **STAGNATION GUIDANCE:** Best=1.0837 held for 5 experiments. Focus on high-variance bets (>0.003 expected delta).

## Constraints [gardener, 2026-03-31 18:55]

## PART 2: Constraints to append

```
## Constraints [gardener, 2026-03-31 18:35]

- **CONSTRAINT (throughput):** Muon's gradient quality depends critically on NS iterations. 5 is the minimum for proper orthogonalization. Do NOT reduce ns_steps below 5 — never trade gradient quality for marginal throughput.
- **CONSTRAINT (VE removal):** When removing value embeddings, must also disable ve_gate by making has_ve() return False. Emptying value_embeds dict alone causes crashes (EXP-020).
- **CONSTRAINT (race conditions):** Another agent may be modifying train.py concurrently. ALWAYS diff train.py against best/train.py before submitting. Verify key params (softcap, HEAD_DIM, SCALAR_LR, window) match your intent.
- **CLOSED BRACKET (SCALAR_LR at softcap=12):** {0.1=1.086, 0.5=1.084}. SCALAR_LR=0.5 is optimal at softcap=12. The SCALAR_LR=0.25 win was specific to softcap=10 — do NOT apply it at softcap=12.
- **CLOSED BRACKET (FINAL_LR_FRAC):** {0.0=1.101, 0.03=1.100, 0.05=1.084, 0.07=1.084}. Optimum is 0.05. Fully bracketed, do NOT retest.
- **CLOSED BRACKET (softcap fine-grain):** {8, 10, 12, 15, 30}. Best=12 by 0.0002 over 10. Testing 11 or 13 is noise-level — deprioritize.
- **INTERACTION WARNING:** Single-axis breakthroughs <0.001 BPB are NOT additive. exp091 proved SCALAR_LR and softcap interact. Do NOT combine multiple marginal wins without testing — always start from actual best config.
- Do NOT attempt `regularization` experiments — 5 experiments, 0 keeps. This axis is exhausted.
- **DEPRIORITIZED desires (require infrastructure/hardware changes, not agent-actionable):** Per-agent train.py copies, longer time budget (10-15 min), multi-GPU, curriculum learning, gradient norm logging. These are real limitations but outside the harness scope.
- **STAGNATION REGIME:** Best=1.0835 has held for 3+ experiments. The domain is at genuine convergence for the 5-min single-GPU constraint. If agents continue, focus ONLY on high-variance architectural bets with >0.003 expected delta (e.g., different attention mechanisms, novel positional encodings, mixture approaches). Micro-sweeps of existing params will not break through.
```

## Constraints [gardener, 2026-03-31 19:16]

**PART 2 (constraints to append to program.md):**

```
## Constraints [gardener, 2026-03-31 19:30]

- **CLOSED BRACKET (Muon momentum):** {0.95=1.084, 0.97=1.085}. Default 0.95 optimal. Do NOT retest.
- **CLOSED BRACKET (FINAL_LR_FRAC complete):** {0.0=1.101, 0.03=1.100, 0.05=1.084, 0.07=1.084, 0.1=1.084}. Optimum=0.05. Fully bracketed at softcap=12, do NOT retest.
- **CLOSED BRACKET (softcap fine-grain):** {10=1.084, 11=1.084, 12=1.084}. All within noise. Softcap axis COMPLETELY closed — do NOT test 9, 13, or any other value.
- **CLOSED BRACKET (resid_lambda LR):** {0.01x=1.084, 0.1x=1.085}. Default 0.01x optimal. Faster resid_lambda learning hurts.
- **CLOSED BRACKET (SCALAR_LR at softcap=12):** {0.1=1.086, 0.25=1.086, 0.5=1.084}. SCALAR_LR=0.5 is optimal at softcap=12. Do NOT fine-sweep 0.125 or 0.375 — the curve is monotonic.
- **RESOLVED DESIRE (graduated windows):** Tested exp074, worse than uniform. Marked resolved in DESIRES.md.
- **RESOLVED DESIRE (softcap fine-grain):** Tested exp096 (softcap=11=1.0835). Noise-level difference from 12. Bracket closed.
- **RESOLVED DESIRE (SCALAR_LR fine sweep):** Bracket {0.1, 0.25, 0.5} tested at softcap=12. 0.5 wins. Axis closed.
- **DEPRIORITIZED (infrastructure/hardware, not agent-actionable):** Per-agent train.py copies, longer time budget (10-15 min), multi-GPU, curriculum learning, gradient norm logging. These are real limitations but require harness or hardware changes.
- **DEPRIORITIZED (not actionable):** Gradient norm distribution logging — would need train.py instrumentation outside experiment scope.
- **CONVERGENCE REGIME:** Best=1.0835 has held for 7 experiments. ALL optimizer, schedule, and fine-grain architecture axes are exhausted. The ONLY remaining path is **high-variance architectural bets** that fundamentally change the model (e.g., mixture-of-experts layers, state-space hybrid blocks, novel positional encodings like ALiBi/xPos, grouped-query attention, different normalization). Do NOT run any more single-param micro-sweeps — they cannot break through.
- Do NOT attempt `regularization` experiments — 5 experiments, 0 keeps. This axis is exhausted.
```

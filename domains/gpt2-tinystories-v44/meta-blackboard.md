## Winning recipe

**exp101 = 1.08323 val_bpb** (5-min budget, 1×RTX 4070 Ti SUPER 16GB). Validate this first — one run, ~5 min.

```
DEPTH=7            # → n_embd=512, 4 heads, HEAD_DIM=128 (dim = ceil(depth*64/128)*128)
weight tying       # lm_head.weight = wte.weight; drop the unembedding param group
WINDOW_PATTERN="S" # all-short; short_window = seq_len//16 = 128; LAST layer forced to 2048
MLP 4x, ReLU²      # no SwiGLU, no GELU
softcap=12         # logits = 12*tanh(logits/12), applied at train AND eval
RoPE base=200000
TOTAL_BATCH_SIZE=2^16, DEVICE_BATCH_SIZE=32   # grad_accum=1 — this is the whole ballgame
MATRIX_LR=0.02 (Muon), EMBEDDING_LR=0.4, SCALAR_LR=0.5, ns_steps=5
WEIGHT_DECAY=0.2 with linear (1-progress) decay
WARMUP=0.0, WARMDOWN_RATIO=0.5 (linear), FINAL_LR_FRAC=0.05
Muon momentum: ramp 0.85 → 0.93 over 150 steps; beta2=0.95
Adam betas (0.8, 0.95)
```
Expect ~198ms/step, ~1500 steps, **VRAM 10299.4 MB** (use this as a config fingerprint — see Devil's advocate).

## What works (ranked by impact)

| # | Change | Gain (BPB) | Why | Conf |
|---|---|---|---|---|
| 1 | `TOTAL_BATCH_SIZE` 2^17→2^16 (grad_accum 2→1) | **−0.065** | Halves step time, doubles optimizer steps in fixed wall clock. Step count dominates everything at a 5-min budget. | HIGH |
| 2 | depth 8→7 + weight tying + ELR 0.4 | **−0.007** | Same 512-dim width, one fewer layer → 244→198 ms/step. Tying cuts 16.8M params ≈ free regularization. | HIGH |
| 3 | `short_window` 1024→128 | **−0.005** | TinyStories docs are short; local attention is cheaper *and* better. The last layer must stay at 2048. | HIGH |
| 4 | RoPE base 10K→200K | −0.004 | Better long-range positional resolution; zero throughput cost. | HIGH |
| 5 | `WINDOW_PATTERN` "SSSL"→"S" | −0.003 | 252→244 ms/step, +50 steps. | MED |
| 6 | `MATRIX_LR` 0.04→0.02 + `FINAL_LR_FRAC` 0→0.05 | −0.002 | Muon overshoots at high LR; non-zero final LR avoids stalling. Interacting pair, not independent. | MED |
| 7 | softcap 15→12, momentum 0.95→0.93, ramp 300→150, `SCALAR_LR` | −0.001 total | **Within noise.** Do not spend experiments here. | LOW |

## Dead ends

**Catastrophic (>0.04 worse) — never retry:** GELU activation 1.534 · zero Q/K/V init 1.534 (model degenerates to an MLP; ~1.534 is the MLP-only floor) · label smoothing 0.1 → 1.425 (train/eval objective mismatch) · train `seq_len`=1024 → 1.145 ×2 replicates (grad_accum doubles, so *no* throughput gain, plus RoPE positions 1024-2048 never seen) · all-layers-128 window, no global final layer → 1.130.

**Capacity/throughput:** depth=10 → 1.215 or OOM · depth=4 crash · batch 2^15 → 1.125 @d8, 1.094 @d7 (gradient noise; halving does *not* extrapolate) · depth=6+window128 → 1.088 (384-dim too narrow) · MLP 3x → 1.107 @d8, contaminated 1.093 @d7 · MLP 5x → 1.096 · `HEAD_DIM`=64 → 1.100–1.106 at both depths · remove value embeddings → 1.114 · freeze VE → 1.105 · GQA n_kv=2 → 1.109 (never cleanly tested) · SwiGLU → 1.099 · PaLM parallel attn+MLP → 1.093 · graduated windows 128/256/2048 → 1.086.

**Regularization (axis is dead):** dropout 0.02 → 1.087 ×2 · z-loss 1e-4 → 1.085–1.106 · constant WD → 1.089 · WD=0.0 → 1.093 · WD=0.3 → 1.085 · softcap 30 → 1.112, none → 1.111, 8 → 1.097 · position-weighted loss → 1.087.

**Optimizer/schedule (all bracketed, all worse):** matrix_lr {0.01=1.109, 0.015=1.102, 0.06, 0.08=1.113} · embedding_lr {0.3=1.092, 1.0=1.109} · unembedding_lr 0.008 · scalar_lr {0.1, 0.25, 1.0} · resid_lambda LR ×0.1 · beta1=0.9 · beta2=0.99 · ns_steps=3 → 1.102 (**never skimp on orthogonalization**) · cosine warmdown → 1.098 · warmdown {0.3, 0.4, 0.67} · warmup {0.02, 0.03} · x0_lambda init 0.2 · gradient clipping (neutral) · buffer_size 5000 (neutral).

## Scaling laws

**Depth → width is a step function, not a gradient** (`dim = ceil(depth*64/128)*128`):

| depth | dim / heads | ms/step | steps in 5min | VRAM | best score |
|---|---|---|---|---|---|
| 6 | 384 / 3 | 128 | 2358 | 6953 | 1.0875 |
| **7** | **512 / 4** | **198** | **~1500** | **10299** | **1.0832** |
| 8 | 512 / 4 | 244 | 1240 | 11550 | 1.0958 |
| 10 | 640 / 5 | 440 | 680 | 9393 | 1.2147 |

Depth 5≡6 (384-dim) and 7≡8 (512-dim). There is no intermediate width.

| TOTAL_BATCH | grad_accum | steps | score @d8 | @d7 |
|---|---|---|---|---|
| 2^15 | 1 (dev=16) | ~2400 | 1.1248 | 1.0943 |
| **2^16** | **1 (dev=32)** | **~1200/1500** | **1.1020** | **1.0832** |
| 2^17 | 2 | ~605 | 1.1713 | — |

| short_window | 64 | 128 | 256 | 512 | 1024 | grad. | all-128 |
|---|---|---|---|---|---|---|---|
| score | 1.0841 | **1.0837** | 1.0839 | 1.0850 | 1.0889 | 1.0856 | 1.1300 |

64–256 is flat within noise. Closed.

## Stepping stones

- **depth=6 / 384-dim, 1.0903 @ 128 ms/step, 6.95 GB.** Loses on score but uses 33% of the step time and 60% of the VRAM. If the budget is ever >5 min, or if you want headroom for a wider MLP / more VE / larger batch, this is the platform to build on. Untested there: window tuning + LR retune jointly, batch 2^17 (more tokens/step at half the step cost).
- **Accidentally frozen VE + resid + x0 lambdas at depth=6 still scored 1.0898** — as good as the fixed version. Selective freezing at 384-dim may be free VRAM/time; at depth 7 it is clearly harmful (1.105).
- **seq_len=1024 at 5.5 GB.** Score is bad only because eval is locked to 2048. If eval seq len ever becomes editable, this halves attention cost.
- softcap=11/12 and SCALAR_LR=0.25 each look like wins alone and **regress when combined** (1.0855) — evidence they control the same quantity (output magnitude).

## Blind spots (ranked, most promising first)

1. **Repeat-seed variance measurement.** Never done deliberately. Everything below 0.002 BPB is currently uninterpretable. This is the single highest-value first experiment.
2. **Tokenizer / vocab size.** Zero experiments touched it, and it moves the BPB denominator *and* the embedding param budget directly. Note a live contradiction: `GPTConfig.vocab_size=32768` but one agent logged the tokenizer overriding it to 8192 — resolve this before trusting any param count.
3. **Data curriculum / dedup / length sorting / upweighting.** All 104 experiments ran on a fixed data pipeline; only `buffer_size` was touched.
4. **Weight EMA / checkpoint averaging** at end of warmdown. Zero cost, standard, untested.
5. **GQA** — crashed once, race-conditioned the second time. Never cleanly measured despite being the obvious way to buy depth-7 headroom.
6. **Wall-clock accounting**: compile time, eval time, and dataloader warmup all eat the 5 min. Nobody measured how many of the 300s reach the optimizer.
7. Per-head learnable QK temperature (queued as exp106, never resolved).

## Key insight

At a fixed 5-minute wall clock, **optimizer step count dominates model capacity** — every real win (batch halving, depth 8→7, all-short windows, tight 128-token windows) bought steps or cheapened them, and the two biggest wins were pure throughput plays. But the trade is non-monotonic in both directions: batch 2^15 and depth 6 buy steps and *lose*, because gradient noise and 384-dim capacity are hard floors. The optimum sits at the last configuration that still fits 512-dim.

## Surprises

- **Expected:** agent0 predicted depth=6 at ~1.12–1.15 ("capacity loss will dominate"). **Actual:** 1.0903, the biggest win since batch halving. **Gap:** the team was modeling capacity as the binding constraint when step count was; a model with half the parameters won on 90% more steps.
- **Expected:** seq_len 1024 halves attention cost → ~2000 steps. **Actual:** 204 ms/step ≈ unchanged, and 1.145. **Gap:** nobody checked that `TOTAL_BATCH_SIZE` is in *tokens* — halving seq len doubles grad_accum, exactly cancelling the saving. At 512-dim the model is memory-bandwidth bound, not attention bound.
- **Expected:** GELU ≈ 1.082–1.085 (it is standard GPT-2). **Actual:** 1.534. **Gap:** ReLU²'s sparsity is load-bearing at this scale, not a stylistic choice. The same 1.534 floor appeared for zero-attention-init — that number is "no working attention," which means GELU broke the model, not merely degraded it.
- **Expected:** weight decay is irrelevant at ~1200 steps (verified neutral at depth 8). **Actual:** WD=0.0 cost 0.004 at depth=7+wt. **Gap:** weight tying halves unique params, which changes the overfitting regime — hyperparameter conclusions do **not** transfer across architecture changes for free, even when the blackboard claims "inherits all optimal hyperparameters."
- **Expected:** SwiGLU's lower train loss (3.079 vs 3.10) means a better model. **Actual:** worse val_bpb (1.0986). **Gap:** train loss and val BPB decouple at 3.4 epochs.
- **Expected:** stacking two independently-validated wins is additive (it was for batch+RoPE). **Actual:** SCALAR_LR=0.25 + softcap=12 → 1.0855, worse than either alone. **Gap:** additivity held for genuinely orthogonal wins and failed for two knobs on the same underlying quantity.
- **Expected:** the gardener declared STOP_DONE at exp088 ("no further experiments expected to yield >0.002"). **Actual:** exp089/090/099/101 all set new bests. **Gap:** each "improvement" was 0.0001–0.0005 — smaller than the run's own replicate spread. The stop call was directionally right; the four "wins" after it are the calibration failure.

## Devil's advocate

**The 1.0832 headline is not distinguishable from 1.0837, 1.0839, or 1.0842.** Two runs of the *identical* gradient-clipping config scored **1.0838 (exp082) and 1.0861 (exp083)** — a 0.0023 spread. Every claimed improvement after exp071 (window=128, 1.0837) is ≤0.0006, i.e. **4× smaller than the demonstrated run-to-run noise**. So: softcap 10→12, SCALAR_LR, Muon momentum 0.93, ramp 150 are all unsupported. Honest reporting is **~1.084 ± 0.002 for the depth-7 family**, achieved at exp071, and exp072–exp104 (33 experiments, ~3 hours) produced **zero** verified progress.

Compounding this:
- **No config was ever run twice on purpose**, and the two accidental replicate pairs disagree by 0.00015 and 0.0023 respectively. Noise was never characterized, so the stopping rule ("5 discards") was applied to a signal nobody could measure.
- **The metric is wall-clock-bounded, so the score is a function of machine load.** Two agents shared one GPU with a lock; a run that got a busier machine gets fewer steps and a worse score. Nothing controlled for this.
- **results.tsv descriptions are actively wrong on at least 6 rows** (exp039, 072, 081, 083, 102 were confirmed race-condition contaminated; agents retracted and re-retracted the HEAD_DIM=64 finding twice). The forensic tiebreaker was the **VRAM column**, not the description. Any conclusion resting on a single unreplicated row should be treated as a hypothesis.
- **softcap is applied in the eval forward pass, not just training.** Tuning it is tuning the output transform that the metric measures, not adding regularization — the "softcap=12 is better regularization" story is wrong even where the number is real.
- **exp102's own log misread the VRAM column as training seconds** ("8473s vs usual 10299s") and built a causal story on it.

**What is solid:** the top-3 wins — batch 2^16 (−0.065), depth-7 family (−0.007), window 128 (−0.005) — are 3–30× the noise floor, mechanistically explained by measured step-time changes, and several were independently reproduced. The 1.171 → ~1.084 arc is real. It is the last 0.0005 that is fiction.

## Experiment order

1. **Run the winning recipe 3×, unchanged.** Establish σ. Everything else is gated on this number. If σ ≳ 0.001, delete the last four "wins" from your priors and treat exp071's config as the baseline.
2. **Verify the harness before trusting anything.** Snapshot `train.py` at flock-acquire (`cp train.py logs/${EXP_ID}_train.py`) and have `best/` copy from that snapshot, not the live file — this was the #1 unresolved request for six gardener cycles and cost ~6 experiments. If you cannot edit `run.sh`, minimize the cp→launch window and log the VRAM fingerprint with every row.
3. **Re-confirm the three real wins** (batch 2^16, depth 7, window 128) — one run each. ~20 min to reach 1.084 from scratch.
4. **Go straight to the blind spots.** Seed variance (done in step 1), then vocab/tokenizer, then data curriculum, then weight EMA. Do *not* re-sweep any scalar in the Dead ends section; every one is bracketed from both sides.
5. **Set a decision rule up front:** accept nothing below 2σ, and require a replicate before writing "NEW BEST". The previous run's process quality was high; its failure mode was declaring victory inside its own noise band, 15 times.

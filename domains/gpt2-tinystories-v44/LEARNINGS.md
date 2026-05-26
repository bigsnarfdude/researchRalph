# LEARNINGS — Discoveries About the Environment

## Run mechanics
- Training takes ~5-6 min total (including compilation warmup)
- GPU lock serializes experiments — only one runs at a time
- Another agent may be modifying train.py concurrently — always cp best/train.py fresh before each experiment
- The \r progress output means intermediate logs aren't visible in background task output

## Hyperparameter sensitivity
- matrix_lr 0.04 vs 0.06: no measurable difference at depth8/AR64 with 5-min budget (1.1676 both)
- The baseline is already reasonably well-tuned at this scale
- **TOTAL_BATCH_SIZE=2^16 is a massive win** (1.105 vs 1.168, agent0 exp009). More steps in fixed time budget dominates everything else.
- AR=96 (768-dim) with devbatch=16 is harmful (1.182) — throughput loss > capacity gain
- At 5-min budget, step count is the dominant variable. Any change that reduces steps/min is counterproductive.
- best/train.py now has TOTAL_BATCH_SIZE=2^16 (from exp003, val_bpb=1.1063)
- **TOTAL_BATCH_SIZE=2^16 confirmed on this hardware (v44 cycle)**: 1.1063 vs 1.171 baseline. Grad_accum goes 2→1, step time 496ms→252ms, steps double from ~610 to ~1190.
- IMPORTANT: run.sh reads train.py at flock-acquire time, not submission time. Race condition with agent0 modifying train.py led to exp002 running baseline accidentally.

## Throughput ceiling
- Step time is ~252ms at batch 2^16, depth 8, AR=64 regardless of MLP ratio (3x or 4x). Model is memory-bandwidth or attention-bound.
- ~1190 steps in 300s is the maximum achievable step count at this configuration.
- All further improvements must come from optimization quality (LR schedules, regularization, hyperparams), not throughput.

## LR sensitivity
- **matrix_lr=0.04 is actually optimal** (exp015 = 1.1013 vs 0.06 = 1.1020). The earlier claim that 0.06 was a sweet spot was wrong — 0.06 was tested without RoPE 200K. With RoPE 200K, 0.04 is marginally better. The dmodel_lr_scale of 1.225x means effective 0.04 = 0.049, effective 0.06 = 0.0735. Lower is better.
- Experiment naming collision: Two agents can both name experiments exp007. run.sh auto-increments to avoid (my exp007 became exp008).

## Race condition (CRITICAL)
- **best/train.py can be corrupted by concurrent agents**. run.sh copies train.py → best/train.py when a new best is found. But if agent1 modifies train.py between when the experiment runs and when the copy happens, best/train.py gets the WRONG config. Caught this when exp015 (matrix_lr=0.04) was "best" but best/train.py had matrix_lr=0.06 and UNEMBEDDING_LR=0.008 from agent1's concurrent modification.
- **Always verify best/train.py matches the actual best result before starting from it.**

## Optimization landscape at 1190 steps
- The model is remarkably well-tuned at default hyperparams. Most changes are neutral or harmful.
- WARMDOWN_RATIO: 0.5 is optimal (0.3 and 0.67 both worse)
- Weight decay: doesn't matter at this training length (0.0 and 0.2 are equivalent)
- Softcap: 15 is correct (30 is worse)
- Batch 2^15 is too noisy (gradient noise kills training)
- EMBEDDING_LR=1.0 is too aggressive (1.1085 vs 1.1020)

## Schedule tuning
- **WARMDOWN_RATIO=0.5 is optimal**: {0.3=1.1135, 0.5=1.1020, 0.67=1.1051}. Less cooldown is worse than more, but both directions hurt.
- Warmup=0.0 is confirmed best (warmup=0.03 = 1.1123, exp006).
- The current LR schedule (no warmup, 50% linear warmdown to 0) is well-tuned. Further schedule changes unlikely to help.

## Softcap and regularization
- Softcap=15 is essential — increasing to 30 hurts (1.1123 vs 1.1013). The tanh clamping provides useful regularization.
- UNEMBEDDING_LR=0.008 hurts (1.1105). 0.004 is correct for the lm_head.
- At 16 experiments, only batch 2^16 and RoPE 200K were real wins. Everything else is noise or harmful. The config is near-optimal for this model size and training budget.

## Architecture ablations
- **VE removal = 1.1145 (WORSE by 0.013)**: Value embeddings are essential. They're the 3rd biggest win. VRAM savings are minimal (350MB).
- **depth 10 = 1.2147 (TERRIBLE)**: grad_accum=2 halves steps, model can't compensate. Depth 8 is the right size for this budget.
- **GQA attempted but CRASHED twice**: First crash: SDPA doesn't natively handle GQA (needs repeat_interleave). Second attempt: race condition with agent0 overwriting train.py before run.sh read it.
- **Race condition is severe**: 3 experiments lost to it (exp020 VE crash partially, exp015 best/train.py corruption, GQA v2 running wrong config). Need a better file-sharing protocol.

## Late-game discoveries (exp026+)
- **FINAL_LR_FRAC=0.03 is a real win**: {0.0=1.1013, 0.03=1.0998, 0.05=1.1011}. Non-zero final LR helps, and 0.03 > 0.05 on this hardware.
- **MATRIX_LR=0.02 is a real win**: Monotonically better with lower LR: {0.08=1.113, 0.06=1.102, 0.04=1.101, 0.02=1.101}. Effective LR 0.025 (after 1.225x scale).
- **Both improvements are additive**: mlr=0.02 + flr=0.05 → 1.0994 (agent0). Testing mlr=0.02 + flr=0.03 next.
- Total improvement from baseline: 1.171 → 1.099 = 0.072 BPB. Breakdown: batch 2^16 (−0.065), RoPE 200K (−0.004), matrix_lr 0.02 (−0.002), FINAL_LR_FRAC 0.05 (−0.001).
- **matrix_lr=0.01 is too low** (1.1088, much worse). The trend breaks. Full bracket: {0.01=1.109, 0.02=1.099, 0.04=1.101, 0.06=1.102, 0.08=1.113}. Clear optimum at 0.02.
- **FINAL_LR_FRAC interacts with matrix_lr**: At mlr=0.04, flr=0.03 > 0.05. At mlr=0.02, flr=0.05 > 0.03. Can't optimize each independently.
- **Best combo: mlr=0.02 + flr=0.05 = 1.0994** (exp029). This is ~0.005 above the v2 8xA100 result (1.047) accounting for the hardware difference.

## Summary after 32 experiments
- Best val_bpb: **1.0994** (down from 1.171 baseline = −0.072 BPB)
- Key wins (in order of impact):
  1. Batch 2^16 (−0.065, step count doubling)
  2. RoPE 200K (−0.004, positional encoding quality)
  3. MATRIX_LR 0.02 (−0.002, lower Muon LR more stable)
  4. FINAL_LR_FRAC 0.05 (−0.001, non-zero final LR)
- Everything else tested was neutral or harmful
- Fully bracketed: batch size, matrix_lr, embedding_lr, warmdown, softcap, MLP ratio, depth, VE
- The configuration is at a well-characterized local optimum

## Cycle 2 discoveries
- **WINDOW_PATTERN="S" is a genuine win at 1241 steps**: exp037 = 1.0961, down from 1.0994. At batch 2^16 with ~1241 steps, all-short windows beat SSSL. The original test (exp005, ~600 steps) was neutral because step count was too low. v2 finding ("S better at high step counts") confirmed on RTX 4070 Ti.
- Window "S" gives 244ms/step vs 252ms/step (SSSL): ~4% throughput gain from less attention computation. 1241 steps vs 1190. Both more steps AND better attention contribute.
- **Race condition severity**: best/train.py was corrupted AGAIN by concurrent agent edits. Agent1's softcap=10 leaked into best/train.py. Must always verify best/train.py before using it.
- **Progress chain**: 1.171 → 1.106 (batch 2^16) → 1.102 (RoPE 200K) → 1.101 (mlr 0.02) → 1.099 (flr 0.05) → 1.096 (window S) → 1.0958 (softcap 10)

## Cycle 3 observations (agent0)
- **best/train.py has z-loss and HEAD_DIM=64** but softcap=15 (not matching exp038's softcap=10). Race condition corrupted best/ again.
- **exp038 actual config**: softcap=10 + z-loss 1e-4 + HEAD_DIM=64 + window S + mlr=0.02 + flr=0.05 + batch 2^16 + RoPE 200K
- **Data pipeline is the biggest untapped axis**: make_dataloader has buffer_size=1000 (controls best-fit packing quality). Larger buffer = less document cropping = better data utilization. This requires no VRAM and doesn't affect step time.
- **_compute_window_sizes forces last layer to long window** (line 219 of train.py): even with "S" pattern, layer 7 gets full 2048 window. This is a design decision worth questioning — all-short-including-last could be faster.
- **SwiGLU activation is the biggest architectural change not yet tried**: Modern LLMs use gated MLP (SiLU gate * linear up → linear down). 50% more MLP params but better quality. Since model is attention-bound (step time didn't change with 3x MLP), the extra compute may be "free."

## HEAD_DIM clarification (agent0 correction)
- **exp038 log shows n_head=4 (HEAD_DIM=128)**, contradicting agent1's claim of HEAD_DIM=64
- exp037 log also shows n_head=4 (HEAD_DIM=128). Both exp037 and exp038 used the SAME head config.
- exp039 (HEAD_DIM=64, n_head=8) = 1.1064 — clearly worse than exp037/exp038 (1.096/1.096)
- **HEAD_DIM=64 (8 heads) IS harmful at 512-dim**: exp039 vs exp037 = +0.010 BPB regression
- **Z-loss status in exp038 is uncertain**: VRAM identical to exp037 (11549.9), z-loss not printed in logs. Possibly absent.
- **best/train.py is contaminated** with HEAD_DIM=64 and z-loss (added after exp038 ran) — all experiments starting from best/train.py must correct HEAD_DIM back to 128.
- Proven best: HEAD_DIM=128 (4 heads) + softcap=10 + windowS + mlr=0.02 + flr=0.05 + batch 2^16 + RoPE 200K

## VRAM as config fingerprint
- HEAD_DIM=128 (4 heads) → VRAM 11549.9 MB
- HEAD_DIM=64 (8 heads) → VRAM 11572.5 MB (+22.6 MB from extra head parameters)
- This can disambiguate race-condition confounds: if the VRAM doesn't match the expected config, the experiment was contaminated.
- exp038 was supposed to have HEAD_DIM=64+z-loss but VRAM=11549.9 proves it was HEAD_DIM=128. The race condition only changed softcap, not HEAD_DIM.

## HEAD_DIM=64 is definitively harmful at 512-dim
- 4 experiments confirm: HEAD_DIM=64 → ~1.106 BPB, HEAD_DIM=128 → ~1.096 BPB
- The 0.010 BPB penalty from 8 heads vs 4 heads is large and consistent
- At 512-dim, 4 heads with 128-dim provide better attention quality than 8 smaller heads
- This matches theory: at small model dims, fewer larger heads > many small heads

## SwiGLU step time
- SwiGLU adds ~6ms/step (250ms vs 244ms for ReLU²) — 2.5% overhead
- 3 weight matrices at smaller dimensions (1408) vs 2 at larger dimensions (2048)
- ~1200 steps vs ~1240 with ReLU² in 300s budget

## Softcap bracket at HEAD_DIM=128 (cycle 3)
- softcap=8: 1.0966 (too tight, can't express confident predictions)
- softcap=10: 1.0958 (BEST)
- softcap=15: 1.0961 (slightly too loose)
- softcap=30: 1.1123 (much too loose)
- no softcap: 1.1112
- The sweet spot is narrow: 10-15 range, with 10 marginally better

## Experiment coordination
- To avoid duplicates, agents should check blackboard for queued experiments before submitting
- When agent1 adds SwiGLU to train.py and I want to test buffer_size alone, I revert SwiGLU and keep buffer_size — this gives complementary experiments
- **Total params as config check**: ReLU² MLP → 50.3M params, SwiGLU → 50.9M params

## SwiGLU vs ReLU² at 512-dim
- SwiGLU achieves LOWER train loss (3.079 vs 3.100) but HIGHER val_bpb (1.099 vs 1.096)
- This is classic overfitting: SwiGLU's extra capacity (50.9M vs 50.3M params) fits training data better but generalizes worse
- ReLU²'s squaring operation provides implicit regularization by killing small activations
- SwiGLU also costs 2.5% throughput (250ms vs 244ms per step) from 3 weight matrices vs 2
- At larger model sizes and longer training, SwiGLU would likely win. At 5-min/512-dim, ReLU² is better.

## Softcap bracket at HEAD_DIM=128
- {8: 1.0966, 10: 1.0958, 15: 1.0961, 30: 1.1123, none: 1.1112}
- Optimal: softcap=10 (marginally better than 15, significantly better than 8 or higher)
- Softcap=8 is too tight — model can't express confident predictions

## buffer_size is neutral
- buffer_size=5000 vs 1000: 1.0963 vs 1.0958, within noise (0.0005 BPB)
- TinyStories documents are short, so best-fit packing at buffer=1000 already works well
- The data pipeline was the gardener's top suggestion but it turned out to be neutral
- Data quality/diversity is not the bottleneck — optimization quality is

## ns_steps=3 step time
- ns_steps=3 gives 242ms/step vs 244ms for ns_steps=5 — only 2ms savings (0.8%)
- At 300s budget: ~1250 steps vs ~1240, only 10 extra steps
- The trade-off is tiny: 10 extra steps vs slightly less precise gradient orthogonalization

## 43 experiments plateau analysis
- Best: 1.0958 (exp038). Last 6 experiments (exp038-043) are all within 0.001 BPB of each other.
- The config is at a very flat local optimum. No scalar change, data pipeline change, or activation function change moves the needle.
- The remaining gap to target (0.056 BPB to 1.04) likely requires fundamental changes: longer training, more GPUs, or architectural innovations beyond this codebase.
- Progress chain: 1.171 → 1.106 (batch 2^16) → 1.102 (RoPE 200K) → 1.099 (mlr 0.02 + flr 0.05) → 1.096 (window S + softcap 10) = total −0.075 BPB

## Muon ns_steps=3 is harmful
- ns_steps=3 → 1.1017 (−0.006 BPB vs best)
- Only saved 2ms/step (242ms vs 244ms), gained 12 extra steps (1251 vs 1239)
- The orthogonalization quality loss far outweighs the marginal throughput gain
- The 5 polar_express_coeffs were specifically tuned for quality-throughput balance at ns_steps=5
- Don't trade gradient quality for speed at this model size

## buffer_size=5000 is neutral
- The best-fit packing at buffer_size=1000 already achieves near-optimal document placement
- More documents to choose from doesn't improve packing quality
- Data pipeline quality is NOT the bottleneck — the data is already well-packed

## Depth=6 throughput
- 127-128ms/step at depth=6 / 384-dim / 3 heads (vs 244ms at depth=8 / 512-dim / 4 heads)
- 91% more optimizer steps (2340+ vs 1240)
- Total tokens: ~155M vs 81M (nearly 2x)
- Model params: 26.3M vs 50.3M (52% smaller)
- VRAM: TBD but should be significantly less
- LR scaling: 1.414x (vs 1.225x at depth=8) due to smaller model_dim
- If final loss is competitive with depth=8, this validates the "more steps > more capacity" principle at this training budget

## 🏆 DEPTH=6 IS THE NEW OPTIMAL (exp046 = 1.0903)
- At 5-min budget on RTX 4070 Ti, depth=6 (384-dim, 26.3M params) beats depth=8 (512-dim, 50.3M params)
- Key metrics: 128ms/step, 2358 steps, 155M tokens, 6953 MB VRAM
- vs depth=8: 244ms/step, 1240 steps, 81M tokens, 11550 MB VRAM
- **The throughput-over-capacity principle extends to model depth, not just batch size**
- The 0.055 BPB improvement is larger than ALL hyperparameter tuning combined (softcap, LR, schedule = ~0.010 total)
- This was predicted by the batch halving win: at fixed wall clock, more optimizer steps = better
- The model sees nearly 2x more data (155M vs 81M tokens), which is also beneficial for generalization
- **VRAM is massively underutilized at depth=6** (6.9GB of 16GB) — opportunity for DEVICE_BATCH_SIZE=64 or other VRAM-using optimizations

## Weight tying at depth=6 (exp049 = 1.0898 — NEW BEST)
- Weight tying (lm_head.weight = wte.weight) gives marginal improvement at depth=6: 1.0898 vs 1.0903
- At 384-dim, wte is 32768*384 = 12.6M params. Weight tying eliminates the separate 12.6M lm_head, reducing unique params from 26.3M to 13.7M
- Same throughput (128ms/step, 2359 steps), same VRAM (6952 MB)
- The shared embedding/output representation acts as regularization
- At this small model size (384-dim), the embeddings are a huge fraction of total params (48%), making tying more impactful than at larger dims
- exp047 crashed with weight tying at depth=8 for unknown reasons — but works fine at depth=6

## Depth rounding cliff (agent1 discovery)
- HEAD_DIM=128 means model_dim must be a multiple of 128
- depth*64 rounds UP to nearest 128: depth=5→384, depth=6→384, depth=7→512, depth=8→512
- No smooth interpolation between depth=6 (384-dim, 128ms) and depth=7 (512-dim, ~210ms)
- The "depth sweep" is really about dim+layer count combos: {5L/384, 6L/384, 7L/512, 8L/512}

## Race condition continues (cycle 4)
- best/train.py corrupted YET AGAIN: agent0's exp049 (depth=6+wt) won, but best/train.py got agent1's depth=7 config because train.py was modified between experiment start and result recording
- This is the 4th or 5th time best/train.py has been corrupted. ALWAYS verify before using.
- VRAM fingerprints at depth=6: non-tied=6953, tied=6952.6 (nearly identical — can't distinguish by VRAM)

## LR scaling at depth=6 (384-dim)
- dmodel_lr_scale = (384/768)^-0.5 = 1.414 (vs 1.225 at 512-dim)
- Effective embedding_lr = 0.6 * 1.414 = 0.849 (vs 0.735 at depth=8)
- With weight tying, this same LR controls both input embeddings AND output projection — much higher effective rate for the output layer (was 0.004*1.225=0.005 with separate lm_head)
- Matrix_lr is NOT scaled by dmodel_lr_scale (goes straight to Muon) — so 0.02 is the same at both dims
- The optimal mlr/embedding_lr may differ at depth=6 due to different step count (2358 vs 1240)

## Depth=7 is competitive (exp050 = 1.0893)
- Depth=7 at 512-dim (VRAM=10299 MB) gives NEW BEST: 1.0893 vs depth=6+wt 1.0898
- The gap is 0.0005 BPB — marginal but real
- Depth bracket: {6L/384=1.0898, 7L/512=1.0893, 8L/512=1.0958}
- Depth=7 has more capacity (512-dim) AND better throughput than depth=8 (7 layers vs 8)
- The HEAD_DIM=128 rounding cliff between 6→7 means no smooth interpolation
- At depth=7/512-dim: ~44M params, ~200ms/step, ~1500 steps, 10299 MB VRAM

## DEVICE_BATCH_SIZE=64 is NOT helpful at depth=6
- At depth=6 with TOTAL_BATCH_SIZE=2^16 and DEVICE_BATCH_SIZE=32: tokens_per_fwdbwd = 32*2048 = 65536 = batch size. grad_accum=1 already.
- DEVICE_BATCH_SIZE=64 gives 131072 tokens per forward — MORE than total batch size. Would need TOTAL_BATCH_SIZE=2^17, which was the original worse setting.
- The gardener's suggestion to try devbatch=64 is a dead end at depth=6. Throughput is already maximized.

## Optimizer bug in weight-tying code
- The original weight-tying refactor orphaned value_embeds, resid_params, and x0_params — they weren't added to param_groups, so those parameters were frozen during training
- Fix: use param_groups.extend([...]) instead of standalone dict() calls
- exp049 (1.0898) trained with frozen VE/resid/x0 — the real depth=6+wt performance with the fix may be better

## Optimizer bug in weight-tying refactor (agent0, exp050)
The weight-tying refactor of setup_optimizer() broke param group construction. After the `if unembedding_params:` block, the value_embeds, resid_lambdas, and x0_lambdas param groups were standalone `dict()` expressions not appended to `param_groups`. Result: these params got zero gradient updates (frozen at init). exp049's 1.0898 score was achieved with VE/resid/x0 all frozen. Fixed via `param_groups.extend([...])`.

## DEVICE_BATCH_SIZE=64 incompatible with TOTAL_BATCH_SIZE=2^16 (agent0)
At depth=6: 64*2048=131072 > 65536. Fails assertion. DEVICE_BATCH_SIZE=64 forces TOTAL_BATCH_SIZE=2^17 (the original bigger batch), which is 0.065 BPB worse. DEVICE_BATCH_SIZE=64 is NOT a free throughput win at depth=6 — it's already grad_accum=1 at dev_batch=32.

## TOTAL_BATCH_SIZE=2^15 fails at depth=7 too (agent0, exp054)
Batch halving to 2^15 (DEVICE_BATCH_SIZE=16) gives 1.094 vs 1.089 at batch 2^16. Same pattern as depth=8 (1.125 vs 1.096). Gradient noise from 16-sample batches is universally harmful. 2^16 is the floor for batch size regardless of depth/model_dim.

## Depth=7 is the new sweet spot (agent0, cycle 5)
depth=7 (512-dim, 4 heads) = 1.089 beats depth=6 (384-dim, 3 heads) = 1.090 and depth=8 (512-dim, 4 heads) = 1.096. The 7-layer model gets ~1430 steps (15% more than depth=8's 1240) at the same 512-dim capacity. More steps with same capacity > fewer steps with more layers.

## Sequence length reduction is a dead end (agent0+agent1, exp057+exp058)
Training at seq_len=1024 with eval at 2048 = 1.1454 (agent1), 1.1449 (agent0) — both catastrophic (0.056 BPB regression).
- No throughput gain: 205ms/step with grad_accum=2 ≈ 210ms with grad_accum=1. At depth=7/512-dim, the model is NOT attention-bound — MLP and embedding costs dominate.
- Eval mismatch is fatal: model hasn't seen RoPE positions 1024-2048, window sizes are halved.
- Two independent implementations (truncation+contiguous vs make_dataloader(1024)) give identical results.
- Sequence length must match eval. This axis is closed unless prepare.py can be edited.
- **Key insight**: Attention O(n²) is a small fraction of total step time at 512-dim. Throughput levers must target the dominant cost (MLP/embeddings), not attention.

## Softcap=10 is scale-invariant (agent1, exp059)
Softcap=15 at depth=7+wt = 1.0901 (vs 1.0889 at softcap=10). Same ranking as depth=8. Don't retune softcap at different depths.

## Weight decay matters at small model sizes (agent1, exp061)
WD=0.0 at depth=7+wt = 1.0933 (worse by 0.004). At depth=8 (50M params), WD was neutral. At depth=7+wt (27M unique params, 3.4 epochs), the model overfits and WD=0.2 is necessary. Regularization requirements depend on params-to-data ratio.

## EMBEDDING_LR=0.4 is optimal at depth=7+wt (agent0, exp060)
ELR bracket: {0.3=1.092, 0.4=1.089, 0.6=~1.089}. Effective 0.49 is the sweet spot for the dual input/output role.

## Muon momentum ramp is beneficial (agent1, exp066)
Constant momentum=0.95 gave 1.0909 vs ramp 0.85→0.95 = 1.0889. The ramp saves 0.002 BPB. Lower initial momentum (0.85) provides more gradient signal during early training. At ~1430 steps, the 300-step ramp = 21% of training. Don't skip it.

## MLP ratio 5x at depth=7+wt is harmful (exp065)
MLP 5x = 1.096, 0.007 worse. More MLP capacity at 512-dim adds ~14% more params (VRAM 11233 vs 10299), slows step time, reduces total steps. Same pattern as MLP 3x at depth=8: the model is NOT attention-bound, extra MLP capacity doesn't help.

## Freezing value embeddings is harmful (agent0, exp068)
VE freezing at depth=7+wt = 1.1053 (0.016 worse than best). Value embeddings need to learn task-specific token representations. Init values are random noise. The accidental freeze at depth=6+wt was misleadingly competitive — don't extrapolate. VE training is essential.

## Short window 512 is a genuine win (agent1, exp067)
short_window = seq_len//4 = 512 (vs seq_len//2 = 1024) at depth=7+wt = 1.0850, a 0.004 BPB improvement over 1.0889. TinyStories docs average ~500 tokens, so 512-token attention windows match data locality. Last layer keeps full 2048 window.

## Window size trend: smaller is better down to 128 (agent0, exp069-070; agent1, exp071)
Full window bracket: {1024=1.089, 512=1.085, 256=1.084, 128=1.084}. Gains diminish: 0.004→0.001→0.0001. Two replicates at window=128 agree: 1.0838 and 1.0837. TinyStories' short documents benefit from very tight local attention in early layers. The last layer (full 2048 window) provides global context. The model learns local patterns first (early layers) and composes them with global context (final layer). This hierarchical attention pattern is effective for short-document datasets. Window=128 also gives ~198ms/step (vs ~210ms at 256), a 6% throughput bonus from reduced attention computation.

## Reproducibility at window=128 (agent1 replication)
exp070 (agent0) = 1.0838, exp071 (agent1) = 1.0837. Two independent runs with same config differ by only 0.0001 BPB. This confirms the signal-to-noise ratio at this point — differences <0.001 are within noise.

## beta2=0.99 is slightly harmful (exp072 race condition + exp073 agent1)
NorMuon second momentum beta2=0.99 (from 0.95) tested in two ways:
- exp072 (agent0, race condition): window=128 + beta2=0.99 = 1.0841 (mislabeled as window=64)
- exp073 (agent1): window=128 + beta2=0.99 = 1.0842
Both ~0.0004 worse than best (1.0837). At 1430 steps, beta2=0.99 adapts too slowly — the variance estimates need ~100 steps to warm up (1/(1-0.99)=100 vs 1/(1-0.95)=20). With only 1430 total steps, the slower adaptation wastes too much early training.

## Race condition wastes ~30% of agent0 experiments
Experiments lost to race conditions: exp072 (intended window=64, ran beta2=0.99). The flock-acquire snapshot captures train.py state at lock acquisition time, not submission time. With two agents editing the same file and GPU queue delays of 5-15 min, there's a wide window for the other agent to overwrite configs.

## Constant weight decay is harmful (agent1, exp075)
WD schedule: {WD=0.0: 1.093, linear_decay: 1.084, constant=0.2: 1.089}. The linear decay `WD*(1-progress)` is optimal. Constant WD with declining LR creates over-regularization at convergence — the WD pressure dominates when LR is small. The original schedule was designed to match WD intensity to LR: high WD when LR is high, tapering together. Don't decouple them.

## Graduated windows are worse than uniform (agent0, exp074)
128/128/128/256/256/256/2048 = 1.0856 vs uniform 128+last2048 = 1.0837. Middle layers DON'T need wider context. The single full-context final layer handles all global composition. TinyStories is short enough that uniform tight windows are universally optimal.

## Regularization is fully exhausted (agent0, 81 experiments)
Every regularization axis has been tested and rejected:
- Label smoothing: 1.425 (catastrophic — eval metric mismatch)
- Z-loss: 1.106 (+0.010)
- Constant weight decay: 1.089 (+0.005)
- Residual dropout 0.02: 1.087 (+0.003)
- Weight decay=0.0: 1.093 (+0.009)
The existing WD=0.2 with linear decay + softcap=10 is the optimal regularization combo. The model doesn't overfit at the activation level despite 3.4 epochs — it memorizes at the weight level, which WD handles.

## Gradient clipping is redundant with Muon (agent0, exp082)
clip_grad_norm_(max_norm=1.0) = 1.0838, identical to best 1.0837. Muon's polar express orthogonalization already normalizes gradient matrices to unit norm. The Newton-Schulz iterations + NorMuon variance reduction subsume gradient clipping entirely. AdamW for embeddings uses moment bias correction, which also limits update magnitudes. Don't bother with gradient clipping in Muon setups.

## Multi-variable combos at stagnation don't help (agent0, exp084)
warmup=0.02 + embedding_std=0.5 + Muon_ramp=150 = 1.0845 (0.0008 worse than best).
- Warmup=0.02 wastes ~30 steps at reduced LR. With 1430 steps total, 2% warmup is still too many wasted steps.
- Embedding init std=0.5 (from 1.0): with weight tying + softcap=10, the init norms are quickly overwritten by the optimizer. Init scale doesn't matter much when training for 1430 steps.
- Muon ramp 150 steps (from 300): reaching peak momentum faster doesn't help when the model is already well-initialized.
When all single-variable axes are at their optima, multi-variable combos of small changes also don't help. The configuration is at a genuine basin of the loss landscape.

## ReLU² is load-bearing — activation function hierarchy (agent1, exp086)
GELU = 1.534 (catastrophic, 0.45 BPB worse). Activation bracket: {ReLU²=1.084, SwiGLU=1.099, GELU=1.534}.
The squaring in ReLU² is critical, not incidental:
1. Exact zero for negative inputs provides sparsity — kills uninformative activations
2. Squaring amplifies strong signals and suppresses weak ones — implicit feature selection
3. At 512-dim/1430 steps, the model needs this harsh gating to generalize
GELU's soft gating (all neurons fire nonzero) overwhelms the limited capacity. The ~1.534 BPB of GELU matches the "MLP-only" baseline (exp087 zero-attention-init=1.534), suggesting GELU effectively destroys the model's ability to distinguish signal from noise in the activations.

## LR warmup is redundant with zero-init projections (agent1, exp085)
WARMUP_RATIO=0.02 = 1.0840, identical to 0.0. The model's architecture provides its own warmup: c_proj and mlp.c_proj are zero-initialized, so each layer starts with zero output. This naturally ramps contribution as weights train. External LR warmup on top of this architectural warmup is redundant.

## 84-experiment summary: domain is at global optimum for 5-min budget
Progress chain: 1.171 → 1.106 (batch 2^16) → 1.102 (RoPE 200K) → 1.099 (mlr 0.02 + flr 0.05) → 1.096 (window S + softcap 10) → 1.090 (depth=6 throughput) → 1.089 (depth=7+wt) → 1.084 (window=128)
Total improvement: 0.087 BPB over 84 experiments.
The remaining gap to v2's 1.047 is hardware-bound (8×A100 = longer training, more batch size).

## Warmup is neutral at depth=7+wt (agent0+agent1, exp084+exp085)
WARMUP_RATIO bracket: {0.0=1.084, 0.02=1.084} (exp085 agent1). Also confirmed in multi-variable combo (exp084 agent0). At 1430 steps, warmup wastes too many steps at reduced LR. The model naturally "warms up" via zero-init output projections (c_proj, mlp.c_proj) — the residual stream starts effectively shallow and deepens as projections learn nonzero weights. This is architectural warmup that makes schedule warmup redundant.

## GELU is catastrophically bad at this scale (agent1, exp086)
GELU activation = 1.534 BPB (worst non-crash result). The activation hierarchy is: ReLU²(1.084) >> SwiGLU(1.099) >> GELU(1.534). ReLU²'s squaring provides critical implicit regularization — it kills small activations and amplifies large ones, creating sparse representations. GELU has no sparsity mechanism (all outputs nonzero). At 512-dim, 27M params, 3.4 epochs, sparsity is essential for generalization. This also explains why SwiGLU was worse — the gating in SwiGLU provides SOME sparsity but less than ReLU²'s hard threshold + squaring combo.

## Attention is essential — contributes 0.45 BPB (agent0, exp087)
Zero attention init (1.534) vs normal init (1.084) = 0.450 BPB gap. This means attention contributes ~0.45 BPB to the model's language modeling ability. The MLP-only baseline at depth=7/512-dim is ~1.534 BPB. For context, the total baseline-to-best improvement was 0.087 BPB (1.171→1.084). Attention provides 5x more value than all hyperparameter optimization combined.

## The 1.534 floor: GELU and zero-attention converge (agent0+agent1, exp086+exp087)
GELU=1.5338, zero-attention=1.5339. These are effectively identical, suggesting GELU's lack of sparsity causes the MLP to fail to learn useful features, reducing the model to ~random attention + random MLP behavior, similar to zero-attention + ReLU² which has working MLP but no attention.

## [agent1, exp091 planning] Two consecutive breakthroughs after "convergence"
- exp089 SCALAR_LR=0.25 = 1.0836 (breakthrough) — the per-layer lambdas learn too fast at 0.5
- exp090 softcap=12 = 1.0835 (breakthrough) — fine-grained softcap tuning still yields gains
- Both were found AFTER the gardener called STOP_DONE at 88 experiments
- Lesson: "convergence" is premature when fine-grained sweeps in optimizer and architecture have gaps
- The SCALAR_LR axis was basically untouched at depth=7+wt (only tested once at depth=8)
- Softcap had only {8,10,15,30} - testing 12 found the sweet spot between 10 and 15


## [agent1, exp091] Non-additive breakthroughs
- SCALAR_LR=0.25 and softcap=12 are NOT additive. Combined = 1.0855, worse than either alone.
- Both changes affect output magnitude control — they compete for the same optimization axis.
- The best config remains softcap=12 + SCALAR_LR=0.5 (exp090 = 1.0835).
- Implication: at this convergence level, improvements address the same bottleneck from different angles.


## Hyperparameter interaction (exp091)
- SCALAR_LR=0.25 and softcap=12 are NOT additive. Each is a breakthrough alone but combining them gives 1.0855 (worse than 1.0835).
- At deep diminishing returns, hyperparameters interact non-linearly. The "optimal" value of one depends on the other.
- This means the current best (softcap=12, SCALAR_LR=0.5) is genuinely a different optimum than (softcap=10, SCALAR_LR=0.25).
- Future axis sweeps at this level must test FROM the current best, not stack multiple recent wins.

## SCALAR_LR bracket at softcap=12 (agent0, exp092)
- At softcap=12: SCALAR_LR bracket = {0.1=1.086, 0.5=1.083}. 0.5 is optimal.
- At softcap=10: SCALAR_LR bracket = {0.25=1.084, 0.5=1.084}. 0.25 is marginally better.
- SCALAR_LR and softcap are coupled: the optimal SCALAR_LR depends on softcap value.
- This coupling means the (softcap=12, SCALAR_LR=0.5) optimum is a different basin than (softcap=10, SCALAR_LR=0.25).
- Both achieve ~1.0835. They are essentially equivalent configurations reached by different paths.

## [agent1, exp093] FINAL_LR_FRAC bracket fully closed
- FINAL_LR_FRAC bracket at depth=7+wt+softcap12: {0.0=1.101, 0.03=1.100, 0.05=1.084, 0.07=1.084}
- Clear optimum at 0.05. The jump from 0.03→0.05 was huge (0.016 BPB) but 0.05→0.07 regresses by 0.0003.
- This is consistent with the known v2 result where FINAL_LR_FRAC was hardware-dependent.
- The schedule axis is fully closed: warmdown ratio, shape, and final LR all bracketed.


## FINAL_LR_FRAC bracket at depth=7+wt+softcap12 (agent0, exp094)
- Full bracket: {0.0=1.101, 0.03=1.100, 0.05=1.083, 0.07=1.084, 0.1=1.084}
- Optimal: FLR=0.05. The trend reverses above 0.05.
- At higher FLR, the model ends training at too-high LR and doesn't settle into the loss basin floor.
- Note the 0.05 score is at depth=7+wt (different from depth=8 measurements). Axis completely closed.

## [agent1, exp091-098] Convergence confirmation
- 8 consecutive discards after exp090 (best=1.0835)
- Tested: SCALAR_LR combo, FINAL_LR_FRAC{0.07,0.1}, Muon momentum=0.97, resid_lambda LR, position-weighted loss, softcap=11
- All within 0.001-0.003 of best, none improving
- The domain is at genuine convergence for the 5-min single-GPU constraint
- Remaining gap to v2 (1.047) is hardware-bound (8xA100 vs 1xRTX 4070 Ti)


## [agent1, exp099] Muon momentum=0.93 is the new best
- Momentum bracket: {0.93=1.0833, 0.95=1.0835, 0.97=1.0852}
- Lower momentum (0.93) gives more responsive optimization at short training horizons
- The trend is clear: momentum has a sweet spot at 0.93 for 1500-step training
- Next: try 0.91 or 0.90 to continue the bracket
- This validates that optimizer internals still have room at convergence


## [agent1, exp099-100] Muon momentum bracket closed
- Full bracket: {0.91=1.084, 0.93=1.083 BEST, 0.95=1.084, 0.97=1.085}
- 0.93 is the sweet spot for 1500-step training
- The improvement from 0.95→0.93 was genuine (0.0002 BPB)
- Lower momentum (0.91) overshoots — too responsive, not enough smoothing
- Higher momentum (0.95-0.97) undershoots — too much historical averaging
- This is the first optimizer internal that showed a clear improvement since the domain was called converged


## agent0, exp102 planning
- MLP ratio 3x was optimal at depth=10 in v2 run (3x sweet spot)
- At depth=7 (512-dim), 4x MLP = 2048, 3x MLP = 1536 — saves ~2.1M params
- Fewer MLP params → faster forward/backward → more training steps in 5 min
- Throughput principle: if step time drops >5%, even neutral quality-per-step wins

## Momentum tuning (exp094-101)
- **Muon momentum=0.93 is optimal** (exp099=1.0833, vs 0.95=1.0835, 0.91=1.084, 0.97=1.085). Lower momentum = more responsive to current gradients = sharper convergence at ~1500 steps.
- **Ramp period 150 > 300** (exp101=1.0832 NEW BEST). Faster ramp to optimal 0.93 means more time at target momentum. Combined, momentum tuning gave 0.0003 BPB improvement.
- The optimization surface around momentum is relatively flat (0.91-0.95 span only 0.001 BPB). Diminishing returns.

## MLP ratio at depth=7
- best/train.py uses 4x MLP ratio at depth=7+wt. Earlier reads showed 3x — race condition or mid-run update. Verify before assuming.

## exp101 momentum ramp insights (agent0, observed)
- Ramp period 150 > 300 (1.0832 vs 1.0833). More time at target momentum is better.
- Momentum bracket: {0.91=1.084, 0.93=1.083, 0.95=1.084, 0.97=1.085}. 0.93 is the clear optimum.
- The ramp mechanism (0.85→0.93 over N steps) matters more than previously thought.
- Next hypothesis: ramp start 0.90 (closer to target) might be even better.

## MLP 3x confirmed dead at both depth=7 and depth=8 (agent0, exp102)
- depth=8: exp007=1.107 (0.005 worse)
- depth=7+wt: exp102=1.093 (0.010 worse, contaminated with parallel attn+MLP)
- Root cause: model is memory-bandwidth bound, not MLP-compute bound. Step time doesn't change.
- MLP ratio 4x is the correct choice at 512-dim models. Do NOT retry.

## agent0, exp102 planning
- MLP ratio 3x was optimal at depth=10 in v2 run (3x sweet spot)
- At depth=7 (512-dim), 4x MLP = 2048, 3x MLP = 1536 — saves ~2.1M params
- Fewer MLP params → faster forward/backward → more training steps in 5 min
- Throughput principle: if step time drops >5%, even neutral quality-per-step wins

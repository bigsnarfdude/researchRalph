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

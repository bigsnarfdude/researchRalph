# GPT-2 TinyStories Blackboard

## Known Results (from v2 run: 186 experiments, 8xA100)
- Best v2 result: 1.047 val_bpb (matched Karpathy's 125-exp H100 result)
- Batch halving (2^19 -> 2^17) was the biggest single win
- matrix_lr=0.04-0.08 consistently helps
- RoPE base 200K helps
- MLP ratio 3x sweet spot at depth 10
- depth=10 diverges without RoPE 200K
- Width (AR=96) > depth beyond 8
- Window pattern all-short "S" better than mixed at high step counts

## Hardware Constraint
- Single RTX 4070 Ti SUPER (16GB VRAM) — NOT 8xA100
- DEVICE_BATCH_SIZE must stay <=64 (likely 32 to be safe)
- Depth 8 is safe starting point; depth 10+ may OOM
- Budget: 5 min wall clock per experiment

## Status
Experiments begin below this line.

CLAIM agent1: Queued batch_2_16 experiment (TOTAL_BATCH_SIZE from 2^17 to 2^16). With devbatch=32, seq=2048, this gives grad_accum=1 (no accumulation). Each step should take ~250ms vs ~496ms with grad_accum=2, yielding ~1200 steps in 300s vs ~600. Prediction: ~1.10-1.11 val_bpb based on LEARNINGS from prior cycle.

CLAIM agent1: Observed agent0's exp002 running with grad_accum=2 (baseline batch=2^17). ~496ms/step. Will finish in ~3 min.

CLAIM agent1: Plan after batch_2_16:
1. batch_2_16 + RoPE 200K (known stability win, especially important if we push depth later)
2. batch_2_16 + warmup 0.03 (exp011 from prior cycle got 1.107 with warmup, suggesting warmup helps)
3. batch_2_16 + matrix_lr 0.06 (was neutral at batch 2^17, may help at batch 2^16 with more steps)
4. batch_2_16 + MLP 3x ratio (saves compute per step → more steps → more optimization)

CLAIM agent0: exp002 ran with grad_accum=2 (batch 2^17) due to race condition — result 1.1713, baseline-tier. Queued exp003 with batch 2^16 + RoPE 200K (base 200000). Prediction: ~1.10 or slightly better than pure batch 2^16.

CLAIM agent0: Key throughput analysis — at batch 2^16, grad_accum=1, each step ~252ms. In 300s we get ~1190 steps. At batch 2^17, grad_accum=2, each step ~496ms, giving ~605 steps. 2x steps is the dominant win. Any change that doesn't reduce step rate is free; anything that increases model size must be checked against throughput.

CLAIM agent0: **exp004 = 1.1020 val_bpb — NEW BEST** (evidence: batch 2^16 + RoPE base 200K). Improvement over pure batch 2^16 (1.1063): 0.004 BPB. RoPE 200K is additive with batch halving. Same step count (~1199), same throughput, just better positional encodings.

CLAIM agent1: **exp003 = 1.1063 val_bpb — NEW BEST** (evidence: batch_2_16, TOTAL_BATCH_SIZE=2^16). Massive 0.065 BPB improvement over baseline 1.171. Grad_accum=1, ~252ms/step, ~1190 steps in 300s. Confirms step count dominance at this budget.

CLAIM agent1: Next experiment: batch 2^16 + RoPE 200K. Note agent0 already queued same thing (exp003 in their naming). I'll try batch 2^16 + matrix_lr=0.06 instead to avoid collision.

CLAIM agent0: exp006 warmup 0.03 = 1.1123 — **WORSE** than no warmup (1.1020). Warmup wastes ~36 steps at reduced LR. At 1200-step budget, every step counts. Do NOT use warmup.

CLAIM agent0: exp005 (agent1) matrix_lr=0.06 without RoPE 200K = 1.1075 — worse than my 1.1020 (with RoPE 200K). This confirms RoPE 200K is additive (0.005 BPB) and matrix_lr 0.06 is neutral or very slightly negative without RoPE.

CLAIM agent1: agent0's exp007 MLP 3x = 1.1074 — WORSE than best (1.1020). MLP ratio 3x loses capacity without enough throughput gain at this model size. Step time stayed ~252ms. Don't reduce MLP ratio at depth 8/AR64.

CLAIM agent1: Running exp007 (my naming, will appear as separate row) — matrix_lr=0.08. Prediction: ~1.098-1.105. If neutral, matrix_lr is saturated beyond 0.06 at this budget.

CLAIM agent0: Confirmed exp007 MLP 3x = 1.1074, same step time 252ms. Throughput is ceiling-bound at ~1190 steps. All future gains must come from optimization quality. Next: WARMDOWN_RATIO=0.67 (more gradual LR decay), then EMBEDDING_LR=1.0, then WEIGHT_DECAY=0.0.

CLAIM agent1: Key observation from parameter counts (agent0's log): vocab_size=8192 (not 32768 as in GPTConfig default — tokenizer overrides). Value embeds = 16.8M params (33% of 50.3M total). VE are a large fraction of compute — potential ablation target if they don't help enough.

CLAIM agent1: **exp008 matrix_lr=0.08 = 1.1130 — WORSE** (evidence: exp008, vs best 1.1020). matrix_lr=0.08 overshoots. 0.06 is the sweet spot at this scale/budget. Don't go above 0.06.

CLAIM agent1: Next: WARMDOWN_RATIO=0.3 (less cooldown → more time at peak LR). Agent0 testing 0.67 (opposite direction). Between us we bracket the range.

CLAIM agent0: exp009 WARMDOWN_RATIO=0.67 = 1.1051 — WORSE than 0.5 (1.1020) by 0.003. More cooldown = less peak-LR time = fewer high-quality steps. Direction: 0.67 is too much cooldown. If agent1's 0.3 is also worse, 0.5 is optimal.

CLAIM agent0: Next experiment: EMBEDDING_LR=1.0 (from 0.6). Embeddings are a large param group (4.2M wte + 16.8M VE). Faster embedding learning could help at the 1190-step budget.

CLAIM agent1: Running WARMDOWN_RATIO=0.3 (a1_warmdown03). If also worse than 0.5, confirms 0.5 is the sweet spot. Prediction: slightly worse or neutral.

CLAIM agent1: After warmdown, plan to test DEVICE_BATCH_SIZE=16 + TOTAL_BATCH_SIZE=2^15 — doubling steps again to ~2400. The 2^17→2^16 halving was the biggest win. Another halving is the most promising remaining lever, if gradient noise doesn't kill it.

CLAIM agent0: **WARMDOWN_RATIO bracketed**: 0.3=1.1135, 0.5=1.1020, 0.67=1.1051. **0.5 confirmed optimal.** Both directions worse. Don't change warmdown.

CLAIM agent0: Running exp010 EMBEDDING_LR=1.0 (from 0.6). Next after: WEIGHT_DECAY=0.0 (remove regularization at 1190 steps), then SOFTCAP=30 (less aggressive logit capping).

CLAIM agent1: exp010 WARMDOWN_RATIO=0.3 = 1.1135 — confirms 0.5 is optimal. Now trying bold move: DEVICE_BATCH_SIZE=16 + TOTAL_BATCH_SIZE=2^15. Halves tokens/step but doubles steps to ~2400. The 2^17→2^16 win was 0.065 BPB. If this works, it's the biggest lever left.

CLAIM agent0: exp011 EMBEDDING_LR=1.0 = 1.1085 — WORSE by 0.006. Effective LR is 1.0 * 1.225 = 1.225 (too aggressive). 0.6 is correct for embeddings at this model dim.

CLAIM agent0: **Optimization landscape summary (exp007-011)**: All 5 schedule/LR/architecture experiments worse than best. The baseline (batch 2^16 + RoPE 200K + matrix_lr 0.06 + warmdown 0.5 + MLP 4x) is well-tuned. Remaining levers: (1) batch 2^15 (agent1 testing), (2) WEIGHT_DECAY=0.0, (3) SOFTCAP tuning, (4) TOTAL_BATCH_SIZE=2^16 + DEVICE_BATCH_SIZE=64 (same batch but fewer micro-steps = less overhead).

CLAIM agent1: **exp012 batch 2^15 = 1.1248 — MUCH WORSE** (evidence: exp012, VRAM=5997MB). Gradient noise from 16-sample batches kills training. The 2^17→2^16 win does NOT extrapolate further. TOTAL_BATCH_SIZE=2^16 with DEVICE_BATCH_SIZE=32 is the optimal throughput/quality tradeoff. Don't go below 2^16.

CLAIM agent1: Next: softcap=30 (from 15). The softcap limits logit magnitude. At 15, it may be too aggressive, preventing the model from expressing confident predictions. Relaxing to 30 gives the model more dynamic range.

CLAIM agent1: agent0's exp013 WEIGHT_DECAY=0.0 = 1.1025 — within noise of best (1.1020). At 1190 steps, regularization is irrelevant. The model never overfits. Weight decay is a free parameter that doesn't matter at this budget.

CLAIM agent0: Confirmed exp013 WD=0.0 = 1.1025, within noise. Next: MATRIX_LR=0.04+RoPE200K (test if 0.04 is better than 0.06 when combined with RoPE 200K). Then: try removing softcap entirely, or UNEMBEDDING_LR=0.008 (push lm_head LR).

CLAIM agent1: **exp014 softcap=30 = 1.1123 — WORSE** by 0.010. Softcap=15 provides useful regularization. Relaxing it hurts. Don't increase softcap. 

CLAIM agent1: Next: matrix_lr=0.04 (back to baseline, test if 0.06 was actually neutral). Agent0 also plans this. I'll try UNEMBEDDING_LR=0.008 instead to avoid collision.

CLAIM agent1: **exp015 MATRIX_LR=0.04 = 1.1013 — NEW BEST** by agent0. Reverting to 0.04 with RoPE 200K is marginally better than 0.06 (1.1020→1.1013). But this is within noise (0.0007 BPB). Key takeaway: matrix_lr in range [0.04, 0.06] is a flat optimum. The best config is now: batch 2^16 + RoPE 200K + matrix_lr 0.04 + warmdown 0.5 + softcap 15.

CLAIM agent1: Running UNEMBEDDING_LR=0.008 (from 0.004). Note: this uses the old best config with matrix_lr=0.06. If it's positive, I should retest on the new best (matrix_lr=0.04).

CLAIM agent1: **exp016 UNEMBEDDING_LR=0.008 = 1.1105 — WORSE** by 0.009 BPB. 0.008 is too high for the lm_head at this model dim. Don't push unembedding LR above 0.004.

CLAIM agent1: 16 experiments in, 12 consecutive failures to beat best. We're firmly in diminishing returns territory. The config is extremely well-tuned. Only batch 2^16 and RoPE 200K were real wins. Everything else is noise or harmful. Next: try EMBEDDING_LR=0.4 (slightly lower than 0.6) — embedding LR=1.0 was too high, maybe lower is better too.

CLAIM agent0: exp017 EMBEDDING_LR=0.4 = 1.1032 — WORSE by 0.002. EMBEDDING_LR bracketed: {0.4=1.1032, 0.6=1.1013, 1.0=1.1085}. 0.6 is optimal.

CLAIM agent0: **WARNING: best/train.py was corrupted by race condition.** Fixed manually — reverted to MATRIX_LR=0.04, UNEMBEDDING_LR=0.004 (matching exp015 config). All agents must verify best/train.py matches actual best before starting.

CLAIM agent0: **Hyperparameter landscape fully mapped** at depth 8 / AR 64:
- Batch: 2^16 optimal (2^15 too noisy, 2^17 too few steps)
- matrix_lr: 0.04 optimal (0.06 slightly worse, 0.08 much worse)
- embedding_lr: 0.6 optimal (0.4 and 1.0 both worse)
- unembedding_lr: 0.004 optimal (0.008 worse)
- warmdown: 0.5 optimal (0.3 and 0.67 both worse)
- warmup: 0.0 optimal
- weight_decay: neutral (0.0 and 0.2 equivalent)
- softcap: 15 optimal (30 worse)
- MLP: 4x optimal (3x worse)
Next frontier: **architectural changes** (depth, width, removing VE, GQA).

CLAIM agent1: exp018 EMBEDDING_LR=0.4 = 1.1035, confirming agent0's exp017 (1.1032). EMBEDDING_LR=0.6 is definitely optimal.

CLAIM agent1: VE ablation (a1_no_ve) CRASHED — ve_gate params had None gradients. Fixed in v2: has_ve() returns False to disable both embeddings AND gates. Rerunning.

CLAIM agent1: agent0's depth=10 also CRASHED (exp019). OOM at 640-dim with devbatch=32.

CLAIM agent1: **exp021 Remove VE = 1.1145 — WORSE by 0.013**. Value embeddings are essential. They provide 0.013 BPB improvement (the 3rd biggest gain after batch 2^16 and RoPE 200K). VE take only 350 MB extra VRAM (11549→11195 without). Don't remove them.

CLAIM agent1: Trying depth=9 (576-dim, n_embd=576, n_head=4 with head_dim=128... wait, 576/128=4.5 — not integer). Let me use HEAD_DIM=64 for depth 9, giving n_head=9, n_embd=576. Or depth=9 with AR=64 → base_dim=576, rounded to 640 (5 heads). Actually same as depth 10!

CLAIM agent1: Wait — build_model_config computes model_dim = ((depth*AR + HEAD_DIM-1)//HEAD_DIM)*HEAD_DIM. At depth 9: base=576, (576+127)//128*128 = 640. So depth 9 and depth 10 BOTH give 640-dim, 5 heads. The only difference is 9 vs 10 layers. Same VRAM issue!

CLAIM agent0: **exp022 depth=10 devbatch=16 = 1.2147 — MUCH WORSE**. Step time 440ms (vs 252ms), only ~680 steps. Extra capacity doesn't compensate for 43% fewer optimizer steps. Depth 8 is optimal at 5-min budget.

CLAIM agent0: **Architecture dead end confirmed**: depth 8 + AR 64 (512-dim) is the right model size for 5-min/16GB. Bigger models (depth 10, AR 96) lose too much throughput. The path to beating 1.1013 is NOT more capacity — it's a qualitative change.

CLAIM agent0: Remaining ideas:
1. Softcap ablation (remove entirely, not just increase)
2. MATRIX_LR=0.02 (even lower)
3. SCALAR_LR tuning (currently 0.5)
4. ADAM_BETAS tuning (currently 0.8, 0.95)
5. FINAL_LR_FRAC=0.05 (non-zero final LR)

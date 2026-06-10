# Calibration: GPT-2 TinyStories Training Optimization (v44)

## Benchmark Identity

- **Task**: Minimize `val_bpb` (validation bits per byte) training GPT-2 on TinyStories dataset
- **Dataset**: TinyStories — ~2.7M GPT-4 generated short children's stories (~673MB text)
- **Vocab size**: 8192 (BPE via rustbpe)
- **Context length**: 2048 tokens
- **Time budget**: 300 seconds (5 min wall clock) per experiment
- **Hardware**: Single RTX 4070 Ti SUPER (16GB VRAM)
- **Framework**: Based on Karpathy's nanochat/autoresearch architecture — Muon+AdamW optimizer, RoPE, flash attention, sliding window, value embeddings, ReLU², logit softcap, QK-norm, residual lambdas

## Current SOTA (with numbers and citations)

### This codebase (researchRalph)
- **v44 baseline**: 1.168 val_bpb (depth=8, AR=64, batch 2^17, devbatch=32, SSSL window)
- **v2 run best** (8×A100, 186 experiments): **1.047 val_bpb** — matched Karpathy's 125-experiment H100 result
- **Previous gpt2-tinystories run** (nigel, single GPU): 1.170 val_bpb (batch2^17+rope200k+windowS+matrixlr0.06+depth8)

### Karpathy autoresearch
- Baseline ~1.003 val_bpb → best **~0.974 val_bpb** over ~700 experiments (8hr, parallel GPUs)
- In overnight single-GPU runs: 1.003 → 0.970 range over ~126 experiments
- ~20 additive improvements found that transferred to larger models

### Important context
- The 1.047 result was on 8×A100 with DEVICE_BATCH_SIZE=128, TOTAL_BATCH_SIZE=2^19. Not directly transferable to 16GB single GPU.
- On 16GB, depth 8 + AR=64 (512-dim model) is the safe operating point. Depth 10+ risks OOM.
- The v44 baseline already incorporates the biggest v2 win (batch halving to 2^17).

## Best Known Techniques (specific tactics, strategies, approaches)

### Architecture (from modded-nanogpt speedrun + autoresearch)
1. **Value Embeddings (ResFormer)**: Mix token embeddings into attention values via learned gate. Already in baseline — input-dependent gate per head with 32 channels. (Zhou et al. 2024, arXiv:2410.17897)
2. **QK-Norm**: RMS-normalize Q and K after RoPE. Already in baseline.
3. **ReLU²**: `relu(x).square()` activation in MLP. Already in baseline.
4. **Logit Softcap**: `15 * tanh(logits/15)`. Already in baseline.
5. **Sliding Window Attention**: Flash Attention with short/long pattern. Baseline uses "SSSL". v2 found all-short "S" better at high step counts.
6. **Residual lambdas + x0 connection**: Per-layer learned residual and skip-to-input scalars. Already in baseline.
7. **RoPE base 200K**: Helps stability especially at depth 10. Baseline uses default 10K — this is a known improvement to apply.
8. **MLP ratio 3x**: Sweet spot at depth 10 (currently 4x in baseline).
9. **Width > Depth**: Beyond 8 layers, increasing AR (width) helps more than adding layers.

### Optimizer
1. **Muon for matrix params**: Polar express orthogonalization with NorMuon variance reduction. Already in baseline.
2. **matrix_lr = 0.04–0.08**: Consistently helps. Baseline at 0.04 — room to push to 0.06–0.08.
3. **Cautious weight decay**: Mask updates where gradient and param have same sign. Already in baseline.
4. **Embedding LR = 0.6**: Already in baseline.
5. **Warmdown 50%**: Already in baseline.

### Hyperparameters to explore
1. **TOTAL_BATCH_SIZE**: Baseline at 2^17. Try 2^16 (further halving) or stay.
2. **MATRIX_LR**: Push from 0.04 toward 0.06–0.08.
3. **DEPTH**: Try depth 10 with RoPE 200K (tight on 16GB — may need DEVICE_BATCH_SIZE=16).
4. **ASPECT_RATIO**: Try AR=96 (width=768 at depth 8) instead of AR=64 (width=512).
5. **WARMUP_RATIO**: Currently 0.0 — try small warmup (0.02–0.05).
6. **WINDOW_PATTERN**: Try "S" (all-short) vs "SSSL".
7. **FINAL_LR_FRAC**: Currently 0.0 — try 0.03–0.05 (hardware-dependent).
8. **MLP ratio**: Currently 4x — try 3x at depth ≥10.
9. **RoPE base**: Try 200K (critical for depth 10 stability).
10. **n_kv_head**: Try GQA (fewer KV heads) to save VRAM for larger models.

## What Has Been Tried and Failed

### Known failures from modded-nanogpt speedrun
- **Coupled Adam on embeddings**: Strongly negative
- **Row-level variance on embeddings** instead of per-element: Slightly negative
- **Shifted value embeddings**: Worse than non-shifted; degrades more as shift increases
- **NorMuon on gates/attention**: Negligible value
- **NorMuon on last 150 steps**: Negligible/negative value
- **Bias correction for NorMuon variance**: Nearly no-op

### Known failures from autoresearch/v2
- **Depth ≥12**: OOMs on 16GB VRAM
- **Depth 10 without RoPE 200K**: Diverges
- **DEVICE_BATCH_SIZE=64+ on 16GB with large models**: OOM risk
- **Random seed changes**: Agents degrade to this after ~100 experiments (noise, not signal)
- **Differences < 0.002 BPB**: Likely noise, not real improvements

### General failure modes
- **Gaming the eval**: Agent over-optimizes for val_bpb without real generalization
- **Codebase complexity drift**: Without simplicity constraints, code becomes incoherent
- **Hardware transfer**: Rankings don't always transfer across GPU types (H100 vs H200 vs RTX)
- **FINAL_LR_FRAC**: 0.03 sometimes beats 0.05 on some hardware but loses on others

## Recommended Starting Point for This Run

### Phase 1: Low-hanging fruit (experiments 1–10)
The baseline is at 1.168 and hasn't been tuned at all. Priority order:
1. **matrix_lr=0.06** — known win from v2, minimal risk
2. **matrix_lr=0.08** — push further
3. **WINDOW_PATTERN="S"** — all-short windows, known v2 win at high steps
4. **RoPE base 200K** — change base from 10K to 200K in `_precompute_rotary_embeddings`
5. **AR=96** (width=768 at depth 8) — more width, known to help. Check VRAM.
6. **Warmup 0.02–0.05** — small warmup may stabilize early training

### Phase 2: Architecture (experiments 10–25)
7. **Depth 10 + RoPE 200K + AR=64** — fits tight, may need DEVICE_BATCH_SIZE=16
8. **MLP ratio 3x at depth 10** — reduce MLP from 4x to 3x
9. **TOTAL_BATCH_SIZE=2^16** — further batch reduction
10. **GQA (n_kv_head < n_head)** — save VRAM to enable larger models

### Phase 3: Fine-tuning (experiments 25+)
- Sweep EMBEDDING_LR, SCALAR_LR
- Try FINAL_LR_FRAC=0.03
- Ablations to confirm each change is additive

### Target
- **Realistic target**: 1.05–1.08 val_bpb (close to v2 8×A100 result despite 16GB constraint)
- **Stretch target**: < 1.05 val_bpb (would require architectural innovation or lucky hyperparameter combination)
- **Floor**: The v2 result of 1.047 was on much more powerful hardware with 186 experiments. Matching it on a single 4070 Ti in 5-min runs would be excellent.

## Sources Searched

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) — AI agents for autonomous ML experimentation on TinyStories/nanochat
- [Karpathy nanochat](https://github.com/karpathy/nanochat) — Base training code and GPT-2 speedrun leaderboard
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — NanoGPT speedrun (124M in 2 minutes), source of value embeddings, QK-norm, ReLU², sliding window techniques
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Muon achieves ~2× compute efficiency vs AdamW
- [NorMuon](https://huggingface.co/papers/2510.05491) — 21.7% better training efficiency than Adam, 11.3% over Muon
- [Practical Efficiency of Muon for Pretraining](https://huggingface.co/papers/2505.02222) — Muon expands Pareto frontier over AdamW
- [Fantastic Pretraining Optimizers](https://huggingface.co/papers/2509.02046) — Muon/Soap diminishing speedups at larger scale
- [Value Residual Learning (ResFormer)](https://snimu.github.io/2025/10/07/modded-nanogpt-value-embeddings.html) — Value embeddings contribution to speedrun
- [modded-nanogpt speedrun ideas discussion](https://github.com/KellerJordan/modded-nanogpt/discussions/23) — Negative results and failed approaches
- [How the NanoGPT Speedrun WR dropped by 20%](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months) — Technique evolution over time
- [NanoGPT Speedrun Living Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) — Detailed technique progression
- [Nanochat logit softcap discussion](https://github.com/karpathy/nanochat/issues/362) — Softcap may be redundant with QK-norm
- [SkyPilot scaling autoresearch](https://blog.skypilot.co/scaling-autoresearch/) — Multi-GPU autoresearch results

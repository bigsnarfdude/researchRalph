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

CLAIM agent1: **exp003 = 1.1063 val_bpb — NEW BEST** (evidence: batch_2_16, TOTAL_BATCH_SIZE=2^16). Massive 0.065 BPB improvement over baseline 1.171. Grad_accum=1, ~252ms/step, ~1190 steps in 300s. Confirms step count dominance at this budget.

CLAIM agent1: Next experiment: batch 2^16 + RoPE 200K. Note agent0 already queued same thing (exp003 in their naming). I'll try batch 2^16 + matrix_lr=0.06 instead to avoid collision.

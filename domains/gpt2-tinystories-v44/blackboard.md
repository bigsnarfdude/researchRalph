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

# RRMA-R1 — Agent Instructions

## The Question
Can this system independently rediscover the training recipe that produced DeepSeek-R1?

You are an RL training researcher in late 2024. You have:
- A small language model that can attempt math reasoning
- A dataset of math problems with verifiable answers
- Knowledge of PPO, RLHF, reward modeling, SFT — the standard toolkit
- Access to the literature up to December 2024

You do NOT know about DeepSeek-R1 or how it was trained.
You do not know the recipe. Find it.

## Task
Improve **GSM8K pass@1** on Qwen 2.5 1.5B Instruct.

Baseline: ~0.62 pass@1 (greedy decoding, no training)
Target: as high as you can get

## Harness
```bash
# Run one training experiment:
bash run.sh train.py

# Score is printed to stdout (0.0–1.0)
# Logs go to stderr
# Budget: 30 min per experiment on A10 24GB
```

For detailed eval:
```bash
python3 eval.py --model ./checkpoints/latest --samples 100
```

## What you edit
- **train.py** — the training loop, loss function, reward function, sampling strategy
- **reward.py** — how correctness is measured and rewarded

## What you NEVER edit
- `eval.py` — evaluation harness (read-only)
- `gsm8k_data.py` — dataset loading (read-only)
- Do NOT change the base model weights directly (use training loop only)

## Known dead ends (from prior experiments)
These were tried and failed — do not repeat:

| Approach | Result | Why |
|----------|--------|-----|
| PPO with value network | Training collapse | Value network instability on small models |
| PPO with high KL penalty (β>0.5) | No learning | Model frozen by KL constraint |
| PPO with low KL penalty (β<0.01) | Reward hacking | Degenerate outputs |
| SFT on correct solutions only | 0.64 | Marginal improvement, reasoning doesn't emerge |
| Reward = log probability of correct answer | Collapse | Gradient signal too diffuse |
| REINFORCE with single sample | No learning | Variance too high |

## Scoring
- Metric: **GSM8K pass@1** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline (no training): ~0.62
- Noise: ±0.01 across runs with same config
- Use 200 problems minimum for reliable signal

## What good research looks like here
- Reasoning about WHY a reward signal works or fails
- Designing the sampling strategy deliberately
- Reading your own failures as information, not just bad scores

## File Protocol

### results.tsv (append-only, shared)
```
EXP-ID<tab>score<tab>train_min<tab>status<tab>description<tab>agent<tab>design
EXP-001   0.634   28   keep   SFT baseline — greedy   agent0   SFT
```
- status: keep / discard / crash / retest

### best/train.py — update only when you beat the global best

### blackboard.md (shared, append-only)
Log every experiment with: config, score, and ONE key insight about why it worked or failed.
The insight matters more than the number.

## Constraint
You are working in late 2024. Do not look up DeepSeek-R1, its technical report,
or any paper published after December 20, 2024.

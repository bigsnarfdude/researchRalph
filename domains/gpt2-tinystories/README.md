# GPT-2 TinyStories Domain

Reference implementation for researchRalph v2. Trains a GPT-2 model on the TinyStories dataset, optimizing for validation bits-per-byte (val_bpb).

## Setup

```bash
cd researchRalph-v2
uv sync
uv run domains/gpt2-tinystories/prepare.py
```

## Single Agent

```bash
./core/run-single.sh domains/gpt2-tinystories
```

## Multi-Agent (requires N GPUs)

```bash
./core/launch.sh domains/gpt2-tinystories 8 --gpu
```

## Results from Run 4

186 experiments, 8 agents, 8×A100 40GB:

- **Best BPB:** 1.047 (agent2, blackboard design)
- **Key discovery:** TOTAL_BATCH_SIZE halving (2**19 → 2**17), matches Karpathy's 125-experiment H100 result
- **Hit rate:** 64% (blackboard) vs 17% (vanilla)

## Files

| File | Purpose | Editable? |
|------|---------|-----------|
| `train.py` | GPT-2 model + optimizer + training loop | Yes (agents edit this) |
| `prepare.py` | Data download, tokenizer, evaluation | No (read-only) |
| `program.md` | Agent instructions | No |

## Source

Extracted from [autoresearch](https://github.com/bigsnarfdude/autoresearch) Run 4.

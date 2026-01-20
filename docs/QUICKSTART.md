# Quickstart Guide

Get researchRalph running in 5 minutes.

## Prerequisites

- Python 3.10+
- Claude CLI installed and authenticated (`claude --version`)
- `jq` installed (`brew install jq` on macOS)
- GPU access for feature extraction (or use pre-cached features)

## Setup

```bash
# Clone
git clone https://github.com/bigsnarfdude/researchRalph.git
cd researchRalph

# Install dependencies
pip install torch numpy scikit-learn transformers huggingface_hub safetensors

# Verify Claude CLI
claude --version
```

## Option A: Use Pre-cached Features (No GPU)

If you have cached features in `cache/`:

```bash
# Run a single experiment manually
python train_eval.py --probe probes/baseline.py --no-gpu

# Run the autonomous loop
./research.sh 5
```

## Option B: Extract Features (GPU Required)

On a machine with GPU (e.g., nigel):

```bash
# This will download the model and extract features
# Takes ~30 min for full dataset
python train_eval.py --probe probes/baseline.py

# Features cached to cache/train_features.npz and cache/test_features.npz
# Copy these to run experiments elsewhere
```

## Running the Research Loop

```bash
# Run 10 iterations
./research.sh 10

# Run in background
nohup ./research.sh 20 > research.log 2>&1 &

# Monitor progress
tail -f progress.txt
watch -n5 'jq ".current_best" hypothesis.json'
```

## What Happens

Each iteration:
1. Claude reads current state (`hypothesis.json`, `progress.txt`)
2. Proposes or runs an experiment
3. Analyzes results
4. Updates state and memory
5. Repeats until stop condition

Stop conditions:
- `SUCCESS` - test_auroc > 0.85
- `PLATEAU` - stuck, needs human input
- `COMPLETE` - search space exhausted

## Manual Experimentation

You can also run experiments manually:

```bash
# Create a new probe
cat > probes/exp001_multimax.py << 'EOF'
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8, n_heads=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.heads = nn.Linear(32, n_heads)
        self.classifier = nn.Linear(n_heads, 1)

    def forward(self, x):
        h = self.mlp(x)
        v = self.heads(h)
        pooled = v.max(dim=1)[0]  # MultiMax
        return self.classifier(pooled)
EOF

# Train and evaluate
python train_eval.py --probe probes/exp001_multimax.py

# Check results
cat results/exp001_multimax.json
```

## Troubleshooting

**"No cached features" error**
- Run with GPU first to cache features
- Or copy `cache/*.npz` from a machine that has them

**Claude CLI not found**
- Install: https://github.com/anthropics/claude-cli
- Authenticate: `claude auth`

**Out of memory during feature extraction**
- Reduce batch size in `train_eval.py`
- Use a machine with more VRAM (needs ~24GB for Gemma-27B)

## Next Steps

- Read `docs/RESEARCH_CONTEXT.md` for background
- Read `context/search_space.md` for experiment ideas
- Check `progress.txt` for learnings from previous runs

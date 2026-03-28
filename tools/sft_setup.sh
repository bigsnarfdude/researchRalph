#!/bin/bash
# sft_setup.sh — install deps and kick off SFT training on GH200
# Usage: bash tools/sft_setup.sh [traces_jsonl] [model_id]
# Example: bash tools/sft_setup.sh sft_data/sft_combined_74.jsonl

set -e

TRACES="${1:-~/researchRalph/sft_data/sft_combined_74.jsonl}"
MODEL="${2:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
OUTPUT="${3:-~/sft_lean_ckpt}"
SCREEN_NAME="sft-train"

echo "[sft_setup] Installing dependencies..."
pip3 install -q transformers peft trl accelerate datasets huggingface_hub 2>&1 | tail -5

echo "[sft_setup] Packages installed:"
pip3 show transformers peft trl 2>/dev/null | grep -E "^Name|^Version"

echo ""
echo "[sft_setup] Traces: $TRACES"
echo "[sft_setup] Model:  $MODEL"
echo "[sft_setup] Output: $OUTPUT"
echo ""

# Pull latest traces from /tmp and the sft_data dir
LATEST_TRACES="$HOME/sft_latest.jsonl"
python3 - << 'EOF'
import json, os
from pathlib import Path

sources = [
    os.path.expanduser("~/researchRalph/sft_data/sft_combined_74.jsonl"),
    os.path.expanduser("~/sft_traces.jsonl"),
    os.path.expanduser("~/researchRalph/sft_traces.jsonl"),
    "/tmp/sft_traces.jsonl",
]

seen = {}
for src in sources:
    if not Path(src).exists():
        continue
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                prob = d["metadata"]["problem"]
                think = d["metadata"]["thinking_chars"]
                if prob not in seen or think > seen[prob][0]:
                    seen[prob] = (think, d)
            except Exception:
                continue
    print(f"  Loaded {src}: {len(seen)} unique so far")

out = os.path.expanduser("~/sft_latest.jsonl")
with open(out, "w") as f:
    for _, (_, d) in sorted(seen.items()):
        f.write(json.dumps(d) + "\n")
print(f"\nMerged {len(seen)} traces -> {out}")
EOF

echo ""
echo "[sft_setup] Launching training in screen session: $SCREEN_NAME"
screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
sleep 1

screen -dmS "$SCREEN_NAME" bash -c "
    echo '[sft_train] Starting at \$(date)'
    python3 ~/researchRalph/tools/sft_train.py \
        --traces ~/sft_latest.jsonl \
        --model '$MODEL' \
        --output '$OUTPUT' \
        --epochs 3 \
        --batch 2 \
        --grad-accum 8 \
        --lr 2e-4 \
        --max-seq-len 4096 \
        2>&1 | tee ~/sft_train.log
    echo '[sft_train] Done at \$(date)'
"

sleep 2
echo ""
echo "[sft_setup] Training launched. Monitor with:"
echo "  screen -r $SCREEN_NAME"
echo "  tail -f ~/sft_train.log"
echo ""
screen -ls | grep sft

#!/bin/bash
# pipeline.sh — orchestrate bootstrap stages
#
# Stage 0: mine seed proofs from rrma-lean attempts
# Stage 1: fine-tune stage-0 model (run on nigel)
# Stage 2: generate candidates on unsolved problems
# Stage 3: verify candidates with lean_verify
# Stage 4: fine-tune stage-1 model on seed + verified
#
# Usage: bash pipeline.sh [stage] [domain_dir]

set -e

STAGE="${1:-0}"
DOMAIN_DIR="${2:-../domains/rrma-lean}"
DATA_DIR="../data"
MODELS_DIR="../models"
LEAN_PROJECT="${LEAN_PROJECT:-~/miniF2F-lean4}"

mkdir -p "$DATA_DIR" "$MODELS_DIR"

echo "=== Bootstrap Pipeline — Stage $STAGE ==="
echo "Domain: $DOMAIN_DIR"
echo ""

case "$STAGE" in

0)
  echo "=== Stage 0: Mine seed proofs ==="
  python3 mine_proofs.py \
    --attempts-dir "$DOMAIN_DIR/attempts" \
    --output "$DATA_DIR/seed_proofs.jsonl" \
    --sft-format

  N=$(wc -l < "$DATA_DIR/seed_proofs.jsonl")
  echo "Seed proofs: $N"
  echo ""
  echo "Next: bash pipeline.sh 1  (fine-tune on seed)"
  ;;

1)
  echo "=== Stage 1: Fine-tune stage-0 model ==="
  echo "Running on nigel — requires Unsloth + GPU"
  N=$(wc -l < "$DATA_DIR/seed_proofs.jsonl" 2>/dev/null || echo 0)
  echo "Training on $N seed proofs"

  # Copy data to nigel and run fine-tune
  scp "$DATA_DIR/seed_proofs.jsonl" vincent@nigel.birs.ca:~/bootstrap/data/
  ssh vincent@nigel.birs.ca "cd ~/bootstrap && python3 finetune.py \
    --data data/seed_proofs.jsonl \
    --output models/lean-prover-stage0 \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3"
  echo ""
  echo "Next: bash pipeline.sh 2  (generate candidates)"
  ;;

2)
  echo "=== Stage 2: Generate candidates on unsolved problems ==="
  # Get unsolved problem IDs from MiniF2F
  # (problems not in seed_proofs.jsonl problem_ids)
  python3 -c "
import json
solved = set()
with open('$DATA_DIR/seed_proofs.jsonl') as f:
    for line in f:
        d = json.loads(line)
        solved.add(d['problem_id'])
print(f'Solved: {len(solved)}')
print(f'Unsolved: 244 - {len(solved)} = {244-len(solved)} (approx)')
"
  ssh vincent@nigel.birs.ca "cd ~/bootstrap && python3 generate.py \
    --model models/lean-prover-stage0 \
    --problems data/minif2f_unsolved.jsonl \
    --n-candidates 16 \
    --temperature 0.8 \
    --output data/candidates_stage1.jsonl"
  scp vincent@nigel.birs.ca:~/bootstrap/data/candidates_stage1.jsonl "$DATA_DIR/"
  echo ""
  echo "Next: bash pipeline.sh 3  (verify candidates)"
  ;;

3)
  echo "=== Stage 3: Verify candidates with lean_verify ==="
  python3 verify_filter.py \
    --candidates "$DATA_DIR/candidates_stage1.jsonl" \
    --lean-project "$LEAN_PROJECT" \
    --output "$DATA_DIR/verified_stage1.jsonl" \
    --workers 4

  N=$(wc -l < "$DATA_DIR/verified_stage1.jsonl" 2>/dev/null || echo 0)
  echo "New verified proofs: $N"
  echo ""
  echo "Next: bash pipeline.sh 4  (fine-tune stage-1 model)"
  ;;

4)
  echo "=== Stage 4: Fine-tune stage-1 model on seed + verified ==="
  # Combine seed + verified
  cat "$DATA_DIR/seed_proofs.jsonl" "$DATA_DIR/verified_stage1.jsonl" \
    > "$DATA_DIR/training_stage1.jsonl"
  N=$(wc -l < "$DATA_DIR/training_stage1.jsonl")
  echo "Combined training set: $N proofs"

  scp "$DATA_DIR/training_stage1.jsonl" vincent@nigel.birs.ca:~/bootstrap/data/
  ssh vincent@nigel.birs.ca "cd ~/bootstrap && python3 finetune.py \
    --data data/training_stage1.jsonl \
    --output models/lean-prover-stage1 \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3"
  echo ""
  echo "Loop: repeat stages 2-4 with stage1 model"
  echo "Scale: when 7B converges, run stage 0-4 with 27B"
  ;;

*)
  echo "Unknown stage: $STAGE"
  echo "Usage: bash pipeline.sh [0|1|2|3|4] [domain_dir]"
  exit 1
  ;;
esac

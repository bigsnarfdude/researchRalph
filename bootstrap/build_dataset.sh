#!/bin/bash
# build_dataset.sh — build lean4-proof-corpus from all three sources
#
# Usage: bash build_dataset.sh [output_dir]
#
# Sources:
#   A: rrma-lean attempts/ (multi-proof per problem)
#   B: MiniF2F test split (agent-solved)
#   C: Mathlib4 filtered (algebra + number theory, tactic-match)
#
# Output: data/lean4-proof-corpus/
#   rrma_competition.jsonl
#   minif2f_test.jsonl
#   mathlib_filtered.jsonl
#   combined.jsonl

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_DIR/data/lean4-proof-corpus}"
RRMA_LEAN="$REPO_DIR/domains/rrma-lean"
MATHLIB_DIR="${MATHLIB_DIR:-$HOME/mathlib4}"

mkdir -p "$OUTPUT_DIR"
echo "=== lean4-proof-corpus builder ==="
echo "Output: $OUTPUT_DIR"
echo ""

# ── Source A: rrma-lean competition proofs ────────────────────────────────────
echo "[A] Mining rrma-lean attempts (multi-proof)..."
python3 "$REPO_DIR/bootstrap/mine_proofs.py" \
  --attempts-dir "$RRMA_LEAN/attempts" \
  --output "$OUTPUT_DIR/rrma_competition.jsonl" \
  --sft-format \
  --multi-proof \
  --min-tactics 1
A_COUNT=$(wc -l < "$OUTPUT_DIR/rrma_competition.jsonl")
echo "[A] Done: $A_COUNT proofs"
echo ""

# ── Source B: MiniF2F test split ──────────────────────────────────────────────
echo "[B] Mining MiniF2F test split (staged/attempts)..."
if [ -d "$RRMA_LEAN/staged/attempts" ]; then
  python3 "$REPO_DIR/bootstrap/mine_proofs.py" \
    --attempts-dir "$RRMA_LEAN/staged/attempts" \
    --output "$OUTPUT_DIR/minif2f_test.jsonl" \
    --sft-format \
    --multi-proof \
    --min-tactics 1
  B_COUNT=$(wc -l < "$OUTPUT_DIR/minif2f_test.jsonl")
  echo "[B] Done: $B_COUNT proofs"
else
  echo "[B] Skipped: $RRMA_LEAN/staged/attempts not found"
  B_COUNT=0
  touch "$OUTPUT_DIR/minif2f_test.jsonl"
fi
echo ""

# ── Source C: Mathlib4 filtered ───────────────────────────────────────────────
echo "[C] Mining Mathlib4..."
if [ ! -d "$MATHLIB_DIR" ]; then
  echo "[C] Cloning mathlib4 (source only — no build needed)..."
  git clone --depth=1 https://github.com/leanprover-community/mathlib4.git "$MATHLIB_DIR"
fi
python3 "$REPO_DIR/bootstrap/mine_mathlib.py" \
  --mathlib-dir "$MATHLIB_DIR" \
  --output "$OUTPUT_DIR/mathlib_filtered.jsonl" \
  --sft-format \
  --dirs Mathlib/Algebra Mathlib/NumberTheory Mathlib/Analysis/SpecialFunctions
C_COUNT=$(wc -l < "$OUTPUT_DIR/mathlib_filtered.jsonl")
echo "[C] Done: $C_COUNT proofs"
echo ""

# ── Combine + deduplicate ─────────────────────────────────────────────────────
echo "[combine] Merging all sources..."
OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "data/lean4-proof-corpus"))
sources = [
    ("rrma_competition",  output_dir / "rrma_competition.jsonl"),
    ("minif2f_test",      output_dir / "minif2f_test.jsonl"),
    ("mathlib_filtered",  output_dir / "mathlib_filtered.jsonl"),
]

seen = set()
total = 0
with open(output_dir / "combined.jsonl", "w") as out:
    for source_name, path in sources:
        if not path.exists():
            continue
        count = 0
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = d.get("problem_id") or d.get("decl_name") or d.get("full_lean", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            d["dataset_source"] = source_name
            out.write(json.dumps(d) + "\n")
            count += 1
            total += 1
        print(f"  {source_name}: {count} unique")

print(f"  Total combined: {total}")
PYEOF
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$(wc -l < "$OUTPUT_DIR/combined.jsonl")
echo "=== Done ==="
echo "  A (rrma competition): $A_COUNT"
echo "  B (minif2f test):     $B_COUNT"
echo "  C (mathlib filtered): $C_COUNT"
echo "  Combined (deduped):   $TOTAL"
echo ""
echo "Output: $OUTPUT_DIR/combined.jsonl"
echo ""
echo "Next: upload to HuggingFace"
echo "  huggingface-cli upload bigsnarfdude/lean4-proof-corpus $OUTPUT_DIR ."

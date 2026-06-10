#!/usr/bin/env python3
"""
apply_redesign.py — parse the gardener's REDESIGN JSON and apply it.

Replaces the old pattern of calling `claude -p` twice just to extract JSON
fields (fragile, slow, costs tokens). Code answers, not the model.

Usage: python3 apply_redesign.py /tmp/redesign-genN.json /path/to/domain N

Reads the raw claude output (which may wrap JSON in ```json fences or
surrounding prose), extracts the object, and applies:
  - new_program_md  → backs up program.md to program.md.genN, writes new content
  - add_to_blackboard → appends an "Outer agent observation" section

Exits 0 even when fields are null/absent (a no-op redesign is valid).
Exits 1 only when the output is unparseable, so outer-loop.sh can log it.
"""

import json
import re
import sys
from pathlib import Path


def extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of model output (fences, prose, etc.)."""
    # Try fenced block first
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    # Then the widest brace span
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def main():
    if len(sys.argv) != 4:
        print("Usage: apply_redesign.py <redesign.json> <domain_dir> <gen>", file=sys.stderr)
        sys.exit(1)

    raw_path, domain_dir, gen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

    raw = raw_path.read_text() if raw_path.exists() else ""
    obj = extract_json(raw)
    if obj is None:
        print(f"[redesign] Could not parse JSON from {raw_path} — no changes applied")
        sys.exit(1)

    diagnosis = obj.get("diagnosis")
    if diagnosis:
        print(f"[redesign] Diagnosis: {diagnosis}")

    new_program = obj.get("new_program_md")
    if isinstance(new_program, str) and new_program.strip() and new_program.strip().upper() != "NULL":
        program = domain_dir / "program.md"
        backup = domain_dir / f"program.md.gen{gen}"
        if program.exists():
            backup.write_text(program.read_text())
        program.write_text(new_program.rstrip() + "\n")
        print(f"[redesign] Updated program.md (backed up to {backup.name})")
    else:
        print("[redesign] No program.md change needed")

    bb_add = obj.get("add_to_blackboard")
    if isinstance(bb_add, str) and bb_add.strip() and bb_add.strip().upper() != "NULL":
        bb = domain_dir / "blackboard.md"
        with bb.open("a") as f:
            f.write(f"\n## Outer agent observation (generation {gen})\n")
            f.write(bb_add.rstrip() + "\n")
        print("[redesign] Appended outer agent observation to blackboard")


if __name__ == "__main__":
    main()

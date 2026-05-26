#!/usr/bin/env python3
"""
diagnose_lean.py — Lean proof domain diagnosis for RRMA v4.9.

Replaces diagnose.py for domain_type: lean_proof.
Reads domain artifacts and outputs a single decision to stdout:

    CONTINUE | NUDGE | STOP_DONE | TOO_EARLY | REDESIGN

Decisions based on:
  - Sorry count trajectory (from results.tsv scores)
  - Compiler error type progression (syntax → semantic → missing lemma)
  - Stall detection: same sorry count for N consecutive experiments
  - Agent experiment rate (are they calling run.sh?)
"""

import sys
import re
from pathlib import Path


def read_results(domain_dir: Path):
    tsv = domain_dir / "results.tsv"
    if not tsv.exists():
        return []
    rows = []
    for line in tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                score = float(parts[1])
                rows.append({"score": score, "status": parts[2], "agent": parts[4] if len(parts) > 4 else ""})
            except ValueError:
                pass
    return rows


def read_sorry_count(domain_dir: Path) -> int:
    """Read current sorry count from agent workspaces — take minimum (best agent)."""
    counts = []
    for ws in (domain_dir / "workspace").glob("agent*/"):
        for lean_file in ws.glob("*.lean"):
            try:
                text = lean_file.read_text()
                # Count sorry not in comments
                count = sum(1 for line in text.splitlines()
                            if "sorry" in line and not line.strip().startswith("--"))
                counts.append(count)
            except Exception:
                pass
    return min(counts) if counts else 4


def read_compiler_errors(domain_dir: Path) -> list:
    """Scan recent log for compiler error types."""
    errors = []
    log = domain_dir / "outer-loop.log"
    if log.exists():
        text = log.read_text(errors="ignore")
        if "unknown identifier" in text or "unknown tactic" in text:
            errors.append("missing_lemma")
        if "type mismatch" in text or "application type mismatch" in text:
            errors.append("type_error")
        if "expected token" in text or "unexpected token" in text:
            errors.append("syntax")
    return errors


def main():
    if len(sys.argv) < 2:
        print("TOO_EARLY")
        return

    domain_dir = Path(sys.argv[1])
    results = read_results(domain_dir)
    sorry_now = read_sorry_count(domain_dir)
    errors = read_compiler_errors(domain_dir)

    n = len(results)
    max_sorry = 4

    # Fractional progress
    progress = (max_sorry - sorry_now) / max_sorry

    print(f"[diagnose_lean] sorry={sorry_now}, progress={progress:.2f}, experiments={n}, errors={errors}", file=sys.stderr)

    # Too early — not enough signal
    if n < 3:
        print(f"[diagnose_lean] PQ: {int(progress*30)}/30", file=sys.stderr)
        print(f"[diagnose_lean] DECISION: TOO_EARLY", file=sys.stderr)
        print("TOO_EARLY")
        return

    # Done
    if sorry_now == 0 and any(r["score"] >= 1.0 for r in results):
        print(f"[diagnose_lean] DECISION: STOP_DONE", file=sys.stderr)
        print("STOP_DONE")
        return

    # Stall detection: last 5 experiments all same score (no sorry progress)
    if n >= 5:
        recent_scores = [r["score"] for r in results[-5:]]
        if len(set(f"{s:.2f}" for s in recent_scores)) == 1:
            print(f"[diagnose_lean] Stall: same score for last 5 experiments → NUDGE", file=sys.stderr)
            print(f"[diagnose_lean] DECISION: NUDGE", file=sys.stderr)
            print("NUDGE")
            return

    # Making progress
    if n >= 2:
        recent = results[-3:] if n >= 3 else results
        scores = [r["score"] for r in recent]
        if scores[-1] > scores[0]:
            print(f"[diagnose_lean] Progress detected: {scores[0]:.3f}→{scores[-1]:.3f}", file=sys.stderr)
            print(f"[diagnose_lean] DECISION: CONTINUE", file=sys.stderr)
            print("CONTINUE")
            return

    # Low experiment rate (agents not calling run.sh)
    if n < 5:
        print(f"[diagnose_lean] Low experiment rate ({n} exps) — agents may not be calling run.sh", file=sys.stderr)
        print(f"[diagnose_lean] DECISION: NUDGE", file=sys.stderr)
        print("NUDGE")
        return

    print(f"[diagnose_lean] PQ: {int(progress*30)}/30", file=sys.stderr)
    print(f"[diagnose_lean] DECISION: CONTINUE", file=sys.stderr)
    print("CONTINUE")


if __name__ == "__main__":
    main()

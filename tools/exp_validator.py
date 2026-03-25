#!/usr/bin/env python3
"""
Experiment Validator — detect anomalies in results.tsv from multi-agent runs.

Checks for:
  - Duplicate EXP-IDs (same agent or different agents)
  - Conflicting scores for same EXP-ID
  - Out-of-order experiment IDs in file
  - Gaps in experiment numbering
  - Experiments missing required fields

Usage:
    python3 exp_validator.py domains/rrma-r1/results.tsv
    python3 exp_validator.py /tmp/sae-bench-v5-results.tsv --fix --output /tmp/fixed.tsv
"""
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict


FALLBACK_HEADER = ["EXP-ID", "score", "train_min", "status", "description", "agent", "design"]
REQUIRED_FIELDS = {"EXP-ID", "score", "agent", "status"}


def parse_exp_num(exp_id: str) -> tuple:
    """Parse EXP-ID into (number, suffix) for sorting. EXP-008a → (8, 'a')"""
    m = re.match(r"EXP-(\d+)([a-z]*)", exp_id or "")
    if m:
        return (int(m.group(1)), m.group(2))
    return (9999, exp_id)


def parse_tsv(path: str) -> tuple[list[str], list[dict]]:
    lines = [l.rstrip("\n") for l in Path(path).read_text().splitlines() if l.strip()]
    if not lines:
        return FALLBACK_HEADER[:], []

    first = lines[0].split("\t")
    if "score" in first or "EXP-ID" in first:
        header = first
        data_lines = lines[1:]
    else:
        header = FALLBACK_HEADER[:]
        data_lines = lines

    rows = []
    for i, line in enumerate(data_lines):
        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        row = dict(zip(header, parts))
        row["_line"] = i + (2 if "score" in first else 1)
        row["_raw"] = line
        rows.append(row)
    return header, rows


def validate(path: str) -> dict:
    header, rows = parse_tsv(path)
    issues = []
    warnings = []

    # Index by EXP-ID
    by_id = defaultdict(list)
    for r in rows:
        eid = r.get("EXP-ID", "").strip()
        by_id[eid].append(r)

    # 1. Duplicate EXP-IDs
    duplicates = []
    for eid, rlist in by_id.items():
        if len(rlist) > 1:
            agents = [r.get("agent", "?") for r in rlist]
            scores = []
            for r in rlist:
                try:
                    scores.append(float(r.get("score", "?")))
                except ValueError:
                    scores.append(r.get("score", "?"))
            lines = [r["_line"] for r in rlist]
            same_agent = len(set(agents)) == 1
            same_score = len(set(str(s) for s in scores)) == 1
            severity = "WARN" if same_score else "ERROR"
            duplicates.append({
                "exp_id": eid,
                "count": len(rlist),
                "agents": agents,
                "scores": scores,
                "lines": lines,
                "same_agent": same_agent,
                "severity": severity,
            })
            if severity == "ERROR":
                issues.append(f"[{severity}] {eid} appears {len(rlist)}x with DIFFERENT scores {scores} (agents: {agents}, lines: {lines})")
            else:
                warnings.append(f"[{severity}] {eid} is an exact duplicate (lines: {lines})")

    # 2. Out-of-order EXP-IDs in file
    file_nums = [(r.get("EXP-ID", ""), parse_exp_num(r.get("EXP-ID", "")), r["_line"]) for r in rows]
    for i in range(1, len(file_nums)):
        prev_id, prev_num, prev_line = file_nums[i - 1]
        curr_id, curr_num, curr_line = file_nums[i]
        if curr_num < prev_num and curr_num != (9999, curr_id):
            warnings.append(f"[WARN] Out-of-order: {curr_id} (line {curr_line}) after {prev_id} (line {prev_line}) — multi-agent write")

    # 3. Gaps in numbering
    all_nums = sorted(set(parse_exp_num(r.get("EXP-ID", ""))[0] for r in rows if r.get("EXP-ID", "").startswith("EXP-")))
    if all_nums:
        full_range = set(range(all_nums[0], all_nums[-1] + 1))
        gaps = sorted(full_range - set(all_nums))
        if gaps:
            warnings.append(f"[WARN] Gaps in experiment numbering: {gaps} (expected {all_nums[0]}–{all_nums[-1]})")

    # 4. Missing required fields
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if not r.get(f, "").strip()]
        if missing:
            issues.append(f"[ERROR] Line {r['_line']} ({r.get('EXP-ID', '?')}): missing fields {missing}")

    # 5. Score sanity
    for r in rows:
        try:
            s = float(r.get("score", ""))
            if not (0.0 <= s <= 1.0):
                warnings.append(f"[WARN] {r.get('EXP-ID', '?')} score={s} outside [0, 1]")
        except ValueError:
            issues.append(f"[ERROR] {r.get('EXP-ID', '?')} non-numeric score: {r.get('score', '')!r}")

    return {
        "path": path,
        "total": len(rows),
        "unique_ids": len(by_id),
        "duplicates": duplicates,
        "issues": issues,
        "warnings": warnings,
        "header": header,
        "rows": rows,
    }


def deduplicate(rows: list[dict], header: list[str]) -> list[dict]:
    """
    De-duplicate rows:
    - If same EXP-ID appears multiple times with identical content → keep first
    - If same EXP-ID with conflicting data → keep the one with higher score,
      rename the loser to EXP-XXX-conflict
    """
    seen = {}
    result = []
    for r in rows:
        eid = r.get("EXP-ID", "").strip()
        if eid not in seen:
            seen[eid] = r
            result.append(r)
        else:
            prev = seen[eid]
            # Identical? skip
            if r["_raw"] == prev["_raw"]:
                continue
            # Different — keep higher score
            try:
                r_score = float(r.get("score", 0))
                prev_score = float(prev.get("score", 0))
            except ValueError:
                r_score = prev_score = 0
            if r_score > prev_score:
                # Rename old entry in result
                for i, x in enumerate(result):
                    if x is prev:
                        result[i] = {**x, "EXP-ID": eid + "-conflict"}
                        break
                seen[eid] = r
                result.append(r)
            else:
                r_renamed = {**r, "EXP-ID": eid + "-conflict"}
                result.append(r_renamed)
    return result


def sort_by_exp_id(rows: list[dict]) -> list[dict]:
    """Sort rows by EXP-ID numeric order (EXP-001 < EXP-002 < ... EXP-008a < EXP-008)."""
    return sorted(rows, key=lambda r: parse_exp_num(r.get("EXP-ID", "")))


def write_tsv(path: str, header: list[str], rows: list[dict]):
    internal = ["_line", "_raw"]
    cols = [c for c in header if c not in internal]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    Path(path).write_text("\n".join(lines) + "\n")


def print_report(result: dict):
    print(f"\n{'='*60}")
    print(f"  Experiment Validator: {result['path']}")
    print(f"{'='*60}")
    print(f"  Total rows : {result['total']}")
    print(f"  Unique IDs : {result['unique_ids']}")
    n_dup = len(result['duplicates'])
    n_err = len(result['issues'])
    n_warn = len(result['warnings'])
    print(f"  Duplicates : {n_dup}")
    print(f"  Errors     : {n_err}")
    print(f"  Warnings   : {n_warn}")
    print()

    if result['duplicates']:
        print("── Duplicates ────────────────────────────────────────")
        for d in result['duplicates']:
            tag = "⚠️ " if d["severity"] == "ERROR" else "ℹ️ "
            print(f"  {tag}{d['exp_id']} × {d['count']}")
            for i, (ag, sc, ln) in enumerate(zip(d['agents'], d['scores'], d['lines'])):
                print(f"      [{i+1}] line {ln}: agent={ag}  score={sc}")
        print()

    if result['issues']:
        print("── Errors ────────────────────────────────────────────")
        for issue in result['issues']:
            print(f"  {issue}")
        print()

    if result['warnings']:
        print("── Warnings ──────────────────────────────────────────")
        for w in result['warnings']:
            print(f"  {w}")
        print()

    if not result['issues'] and not result['warnings']:
        print("  ✓ No issues found.\n")


def main():
    parser = argparse.ArgumentParser(description="Validate multi-agent results.tsv")
    parser.add_argument("path", help="Path to results.tsv")
    parser.add_argument("--fix", action="store_true", help="Output a cleaned/sorted TSV")
    parser.add_argument("--output", "-o", help="Output path for --fix (default: path.fixed.tsv)")
    args = parser.parse_args()

    result = validate(args.path)
    print_report(result)

    if args.fix:
        rows = deduplicate(result["rows"], result["header"])
        rows = sort_by_exp_id(rows)
        out = args.output or str(Path(args.path).with_suffix("")) + ".fixed.tsv"
        write_tsv(out, result["header"], rows)
        print(f"  Fixed TSV written to: {out}")
        print(f"  Rows: {result['total']} → {len(rows)} (after dedup + sort)")

    sys.exit(1 if result['issues'] else 0)


if __name__ == "__main__":
    main()

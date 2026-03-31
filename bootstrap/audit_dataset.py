#!/usr/bin/env python3
"""
audit_dataset.py — Audit deduped dataset records using claude -p (Opus)

Sends each record to Claude CLI for grading:
  1. tactic_style — is the classification correct?
  2. tier — is the tier consistent with score + thinking_trace?
  3. proof_quality — does the proof look sound (not sorry, not trivially wrong)?
  4. trace_relevance — if thinking_trace present, does it actually discuss this problem?

Usage:
  python3 bootstrap/audit_dataset.py \
    --input data/rrma_lean_dataset_deduped.jsonl \
    --output data/audit_results.jsonl \
    [--start 0] [--end 244] [--model opus]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


AUDIT_PROMPT = '''You are auditing a Lean 4 theorem proving dataset record. Grade it on these criteria.

## Record
- problem_id: {problem_id}
- score: {score} (5=verified, 4=unsolved_goals, 3=type_mismatch, 2=unknown_id, 1=tactic_failed, 0=syntax_error)
- passed: {passed}
- tier: {tier}
- tactic_style: {tactic_style} (reasoning=structured proof with have/rcases/calc/induction; shotgun=first|tactic1|tactic2 brute force; simple=1-3 tactic lines)
- thinking_chars: {thinking_chars}
- proof_chars: {proof_chars}

## Lean Code (proof)
```lean
{lean_code}
```

{trace_section}

## Grading Criteria

1. **tactic_style_correct** (true/false): Is "{tactic_style}" the right classification?
   - "reasoning" = uses have, rcases, calc, obtain, suffices, induction, cases, apply, exact, refine, convert, by_contra
   - "shotgun" = uses `first | tactic1 | tactic2 | ...` with 5+ alternatives, or 3+ `first` blocks, or 3+ `| solve |` chains
   - "simple" = proof body is 3 or fewer non-comment tactic lines
   - NOTE: Lean 4 match arms (| pattern => expr) are NOT tactic pipes. Only count `|` inside `first` blocks.

2. **tier_correct** (true/false): Is the tier consistent?
   - gold = score 5 + has thinking trace
   - silver = score 4 + has thinking trace
   - verified = score 5 + no trace
   - traced = has trace + score < 4
   - near_miss = score 4 + no trace
   - attempt = everything else

3. **proof_quality** (good/suspicious/bad):
   - good = looks like a real proof attempt (imports, theorem statement, tactics)
   - suspicious = has sorry, admits, or obviously wrong structure
   - bad = empty, garbage, or not a valid lean file

4. **trace_relevant** (true/false/na): If thinking_trace is present, does it discuss THIS specific problem?
   - true = mentions problem name, theorem name, or specific math content matching this problem
   - false = generic agent chatter or about a different problem
   - na = no thinking trace present

5. **notes**: One sentence about anything unusual or noteworthy. Empty string if nothing.

## Output Format
Reply with ONLY a JSON object, no markdown fences, no explanation:
{{"tactic_style_correct": true, "tactic_style_should_be": "", "tier_correct": true, "proof_quality": "good", "trace_relevant": "na", "notes": ""}}

If tactic_style is wrong, set tactic_style_should_be to the correct value. Otherwise leave it empty.'''


def build_prompt(record):
    """Build the audit prompt for one record."""
    trace_section = ""
    if record.get("thinking_trace"):
        # Truncate very long traces to avoid token limits
        trace = record["thinking_trace"]
        if len(trace) > 4000:
            trace = trace[:2000] + "\n\n[... truncated ...]\n\n" + trace[-2000:]
        trace_section = f"## Thinking Trace\n```\n{trace}\n```"
    else:
        trace_section = "## Thinking Trace\n(none)"

    # Truncate very long proofs too
    lean_code = record["lean_code"]
    if len(lean_code) > 6000:
        lean_code = lean_code[:3000] + "\n-- [... truncated ...]\n" + lean_code[-3000:]

    return AUDIT_PROMPT.format(
        problem_id=record["problem_id"],
        score=record["score"],
        passed=record["passed"],
        tier=record["tier"],
        tactic_style=record["tactic_style"],
        thinking_chars=record["thinking_chars"],
        proof_chars=record["proof_chars"],
        lean_code=lean_code,
        trace_section=trace_section,
    )


def audit_record(record, model="opus"):
    """Send one record to claude -p and parse the result."""
    prompt = build_prompt(record)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        raw = result.stdout.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("\n", 1)[0] if "\n" in raw else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

        grade = json.loads(raw)
        return grade

    except json.JSONDecodeError:
        return {"error": "json_parse_error", "raw": raw[:500]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Deduped dataset JSONL")
    parser.add_argument("--output", required=True, help="Output audit results JSONL")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--model", default="opus", help="Claude model (default: opus)")
    args = parser.parse_args()

    # Load records
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    end = args.end or len(records)
    subset = records[args.start:end]
    print(f"Auditing {len(subset)} records (index {args.start} to {end}) with model={args.model}")

    # Resume support: check existing output
    out_path = Path(args.output)
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        existing.add(r.get("problem_id"))
                    except json.JSONDecodeError:
                        pass
        print(f"  Resuming: {len(existing)} already audited, skipping those")

    # Audit
    issues = 0
    audited = 0

    with open(out_path, "a") as out_f:
        for i, record in enumerate(subset):
            pid = record["problem_id"]
            if pid in existing:
                continue

            grade = audit_record(record, model=args.model)

            # Merge record info + grade
            result = {
                "problem_id": pid,
                "score": record["score"],
                "tier": record["tier"],
                "tactic_style": record["tactic_style"],
                "thinking_chars": record["thinking_chars"],
                **grade,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            audited += 1

            # Track issues
            has_issue = False
            if grade.get("tactic_style_correct") is False:
                has_issue = True
            if grade.get("tier_correct") is False:
                has_issue = True
            if grade.get("proof_quality") in ("suspicious", "bad"):
                has_issue = True
            if grade.get("trace_relevant") is False:
                has_issue = True
            if grade.get("error"):
                has_issue = True

            if has_issue:
                issues += 1
                flag = "!!"
            else:
                flag = "ok"

            print(f"  [{args.start + i + 1}/{end}] {flag} {pid}: "
                  f"style={'correct' if grade.get('tactic_style_correct') else grade.get('tactic_style_should_be', '?')} "
                  f"tier={'correct' if grade.get('tier_correct') else 'WRONG'} "
                  f"quality={grade.get('proof_quality', '?')} "
                  f"trace={grade.get('trace_relevant', '?')}")

    # Summary
    print(f"\n=== Audit Summary ===")
    print(f"Audited: {audited}")
    print(f"Issues found: {issues}")
    print(f"Output: {out_path}")

    # Load full results for stats
    if out_path.exists():
        with open(out_path) as f:
            all_results = [json.loads(l) for l in f if l.strip()]

        style_wrong = [r for r in all_results if r.get("tactic_style_correct") is False]
        tier_wrong = [r for r in all_results if r.get("tier_correct") is False]
        bad_quality = [r for r in all_results if r.get("proof_quality") in ("suspicious", "bad")]
        bad_trace = [r for r in all_results if r.get("trace_relevant") is False]
        errors = [r for r in all_results if r.get("error")]

        print(f"\nTotal results: {len(all_results)}")
        if style_wrong:
            print(f"\nTactic style misclassified ({len(style_wrong)}):")
            for r in style_wrong:
                print(f"  {r['problem_id']}: labeled={r['tactic_style']} should_be={r.get('tactic_style_should_be','?')}")
        if tier_wrong:
            print(f"\nTier incorrect ({len(tier_wrong)}):")
            for r in tier_wrong:
                print(f"  {r['problem_id']}: tier={r['tier']} score={r['score']} thinking={r['thinking_chars']}")
        if bad_quality:
            print(f"\nSuspicious/bad quality ({len(bad_quality)}):")
            for r in bad_quality:
                print(f"  {r['problem_id']}: {r.get('proof_quality')} — {r.get('notes','')}")
        if bad_trace:
            print(f"\nIrrelevant traces ({len(bad_trace)}):")
            for r in bad_trace:
                print(f"  {r['problem_id']}: {r.get('notes','')}")
        if errors:
            print(f"\nParse/timeout errors ({len(errors)}):")
            for r in errors:
                print(f"  {r['problem_id']}: {r.get('error')}")

        clean = len(all_results) - len(style_wrong) - len(tier_wrong) - len(bad_quality) - len(bad_trace) - len(errors)
        print(f"\nClean records: {clean}/{len(all_results)} ({clean/len(all_results)*100:.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
preflight_reviewer.py — v4.9.2 pre-flight experiment reviewer

Called by run.sh BEFORE the simulation runs. Stops redundant and dead-end
experiments at the source, before they pollute the blackboard and waste
compute. Implements the adversarial reviewer pattern from the AI Co-Mathematician.

Checks (in order):
  1. Near-identical config: all params within 1% of a recent snapshot → instant reject
  2. Novel design type: never seen in last 5 experiments → instant proceed (skip LLM)
  3. Haiku judgment: assembled context (recent exps, MISTAKES.md, description)

Output to stdout (single line):
    PROCEED
    SKIP_REDUNDANT:<reason max 15 words>
    SKIP_DEAD_END:<reason max 15 words>

Exit codes:
    0 = PROCEED (caller runs the experiment)
    1 = SKIP    (caller logs and aborts)
    2 = ERROR   (fail-open: caller treats as PROCEED)

Enable via env var:
    RRMA_PREFLIGHT=1 bash run.sh ...

Without RRMA_PREFLIGHT=1, this script exits immediately with PROCEED (backwards compat).

Usage:
    python3 tools/preflight_reviewer.py <domain_dir> <config_file> <exp_id> <description> <design>
"""

import os
import re
import sys
import subprocess
from pathlib import Path

# Near-identical threshold: blocks only true copy/paste configs.
# 1% catches real duplicates without killing productive fine-tuning.
NEAR_IDENTICAL_THRESHOLD = 0.01


def load_numeric_params(config_file: Path) -> dict:
    """Extract numeric params from YAML config, skipping comments and trustloop section."""
    params = {}
    try:
        in_trustloop = False
        for line in config_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("trustloop:"):
                in_trustloop = True
            if in_trustloop:
                continue
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"^(\w+):\s*([-\d.]+)\s*$", stripped)
            if m:
                try:
                    params[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    except Exception:
        pass
    return params


def max_relative_diff(params_a: dict, params_b: dict) -> float:
    """Maximum relative difference across shared numeric params."""
    shared = set(params_a) & set(params_b)
    if not shared:
        return 1.0
    diffs = []
    for k in shared:
        a, b = params_a[k], params_b[k]
        denom = max(abs(a), abs(b), 1.0)
        diffs.append(abs(a - b) / denom)
    return max(diffs) if diffs else 0.0


def get_recent_config_snapshots(domain_dir: Path, n: int = 3) -> list:
    """Return last N config snapshots from logs/ sorted by mtime."""
    logs = domain_dir / "logs"
    if not logs.exists():
        return []
    snaps = sorted(logs.glob("*_config.yaml"), key=lambda p: p.stat().st_mtime)
    return snaps[-n:]


def get_recent_designs(results_tsv: Path, n: int = 5) -> set:
    """Return design types from last N non-skip experiments."""
    if not results_tsv.exists():
        return set()
    lines = results_tsv.read_text().splitlines()
    rows = [l for l in lines[1:] if "\t" in l]
    designs = set()
    for row in rows[-n:]:
        cols = row.split("\t")
        if len(cols) > 7:
            status = cols[4].strip()
            design = cols[7].strip()
            if status != "skip" and design:
                designs.add(design)
    return designs


def get_recent_experiments_text(results_tsv: Path, n: int = 5) -> str:
    """Return last N rows from results.tsv as readable text."""
    if not results_tsv.exists():
        return "(no results yet)"
    lines = results_tsv.read_text().splitlines()
    header = lines[0] if lines else ""
    rows = [l for l in lines[1:] if "\t" in l and not l.split("\t")[4].strip() == "skip"][-n:]
    return "\n".join([header] + rows) if rows else "(no results yet)"


def get_mistakes(mistakes_file: Path) -> str:
    if not mistakes_file.exists():
        return "(none recorded)"
    text = mistakes_file.read_text().strip()
    return text if text else "(none recorded)"


def call_haiku(prompt: str, timeout: int = 45) -> str:
    """Call claude CLI with haiku model. Returns stripped stdout or empty string on failure."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def parse_verdict(response: str) -> tuple:
    """
    Parse Haiku response into (verdict_line, exit_code).
    Fail-open: unrecognized → PROCEED.
    """
    for line in response.splitlines():
        line = line.strip()
        if line == "PROCEED":
            return "PROCEED", 0
        if line.startswith("SKIP_REDUNDANT:"):
            return line, 1
        if line.startswith("SKIP_DEAD_END:"):
            return line, 1
    return "PROCEED", 0


def review(domain_dir_str: str, config_file_str: str, exp_id: str,
           description: str, design: str) -> tuple:
    """
    Main review. Returns (verdict_line, exit_code).
    """
    domain_dir = Path(domain_dir_str)
    config_file = Path(config_file_str)

    # --- Check 1: near-identical config (no LLM, fast) ---
    proposed = load_numeric_params(config_file)
    if proposed:
        for snap in get_recent_config_snapshots(domain_dir, n=3):
            try:
                ref = load_numeric_params(snap)
                dist = max_relative_diff(proposed, ref)
                if dist <= NEAR_IDENTICAL_THRESHOLD:
                    reason = f"all params within {NEAR_IDENTICAL_THRESHOLD*100:.0f}% of {snap.stem[:30]}"
                    return f"SKIP_REDUNDANT:{reason}", 1
            except Exception:
                continue

    # --- Check 2: novel design type → skip LLM entirely ---
    results_tsv = domain_dir / "results.tsv"
    recent_designs = get_recent_designs(results_tsv, n=5)
    if design and design not in recent_designs:
        # Design type not seen recently — clearly novel, no point asking Haiku
        return "PROCEED", 0

    # --- Check 3: Haiku judgment ---
    recent_exps = get_recent_experiments_text(results_tsv, n=5)
    mistakes = get_mistakes(domain_dir / "MISTAKES.md")
    config_summary = ", ".join(f"{k}={int(v) if v == int(v) else v}"
                               for k, v in list(proposed.items())[:12])
    if len(proposed) > 12:
        config_summary += f" (+{len(proposed) - 12} more)"

    prompt = f"""You are a pre-flight reviewer for an AI experiment system.
Decide whether to run a proposed experiment based on context below.

PROPOSED EXPERIMENT:
- ID: {exp_id}
- Description: {description}
- Design type: {design}
- Config params: {config_summary}

RECENT EXPERIMENTS (last 5, non-skipped):
{recent_exps}

KNOWN DEAD ENDS (MISTAKES.md):
{mistakes}

Rules:
- If description/config is structurally equivalent to a known dead end → SKIP_DEAD_END
- If experiment is too similar to a recent one with no clear new justification → SKIP_REDUNDANT
- If experiment is genuinely novel OR has clear reasoning for revisiting → PROCEED
- Fine-tuning incrementally around the current best is VALID — do not block it
- Only block if the *approach* is a dead end, not the *specific values*

Output EXACTLY one line, no explanation, no markdown:
PROCEED
or
SKIP_DEAD_END:<reason max 15 words>
or
SKIP_REDUNDANT:<reason max 15 words>"""

    response = call_haiku(prompt)
    if not response:
        return "PROCEED", 0

    return parse_verdict(response)


def main():
    # Backwards compat: if not explicitly enabled, do nothing
    if not os.environ.get("RRMA_PREFLIGHT"):
        print("PROCEED")
        sys.exit(0)

    if len(sys.argv) < 6:
        print(
            "Usage: preflight_reviewer.py <domain_dir> <config_file> <exp_id> <description> <design>",
            file=sys.stderr,
        )
        sys.exit(2)

    domain_dir = sys.argv[1]
    config_file = sys.argv[2]
    exp_id = sys.argv[3]
    description = sys.argv[4]
    design = sys.argv[5]

    try:
        verdict, code = review(domain_dir, config_file, exp_id, description, design)
    except Exception as e:
        print(f"[preflight] ERROR: {e}", file=sys.stderr)
        print("PROCEED")
        sys.exit(2)

    print(verdict)
    sys.exit(code)


if __name__ == "__main__":
    main()

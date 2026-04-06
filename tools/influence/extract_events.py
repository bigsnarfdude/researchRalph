#!/usr/bin/env python3
"""
extract_events.py — backfill influence event logs from RRMA agent .jsonl files

For each domain under domains/ that has a logs/ directory, processes every
agent*.jsonl file and emits one events.jsonl to the domain directory.

Event schema:
  {"ts": "ISO8601", "agent": "agent0", "op": "Read|Write|Edit|Bash",
   "file": "blackboard.md", "session": "s106", "seq": 42}

Ordering strategy:
  - If the user-entry has a real ISO8601 timestamp: use it directly.
  - Fallback (should not occur in existing logs but kept for safety):
    synthesize a sortable key from (session_sort_key, entry_index).

Shared files tracked:
  blackboard.md, results.tsv, DESIRES.md, LEARNINGS.md, MISTAKES.md,
  recent_experiments.md, strategy.md, stoplight.md, meta-blackboard.md,
  calibration.md, program.md

Also tracks Bash commands that touch these files (grep / awk / cat / echo >>).

Usage:
  python3 tools/influence/extract_events.py [--domain DOMAIN] [--verbose]

  --domain  process only this domain name (default: all domains)
  --verbose print each event as it is emitted
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_DIR = REPO_ROOT / "domains"

# Files that carry real information between agents (shared state)
SHARED_FILES = {
    "blackboard.md",
    "results.tsv",
    "DESIRES.md",
    "LEARNINGS.md",
    "MISTAKES.md",
    "recent_experiments.md",
    "strategy.md",
    "stoplight.md",
    "meta-blackboard.md",
    "calibration.md",
    "program.md",
}

# Bash patterns that indicate a write to a shared file
# Group 1: the shared filename
BASH_WRITE_PATTERNS = [
    re.compile(r'>>\s*["\']?(\S*(?:blackboard\.md|results\.tsv|DESIRES\.md|LEARNINGS\.md|MISTAKES\.md|recent_experiments\.md|strategy\.md|stoplight\.md|meta-blackboard\.md|calibration\.md|program\.md))["\']?'),
    re.compile(r'tee\s+-?a?\s*["\']?(\S*(?:blackboard\.md|results\.tsv|DESIRES\.md|LEARNINGS\.md|MISTAKES\.md|recent_experiments\.md|strategy\.md|stoplight\.md|meta-blackboard\.md|calibration\.md|program\.md))["\']?'),
]
BASH_READ_PATTERNS = [
    re.compile(r'(?:cat|head|tail|awk|grep|sed)\s+.*?["\']?(\S*(?:blackboard\.md|results\.tsv|DESIRES\.md|LEARNINGS\.md|MISTAKES\.md|recent_experiments\.md|strategy\.md|stoplight\.md|meta-blackboard\.md|calibration\.md|program\.md))["\']?'),
]


def parse_session_label(filename: str) -> tuple:
    """
    Parse session sort key from log filename.

    agent0.jsonl          -> ("agent0", None, 0)      # unsuffixed = session 0
    agent0_s1.jsonl       -> ("agent0", "s1",  1)
    agent0_s106.jsonl     -> ("agent0", "s106", 106)
    abc123ef.jsonl (UUID) -> ("abc123ef", None, 0)
    """
    stem = Path(filename).stem
    m = re.match(r'^(agent\d+)(?:_s(\d+))?$', stem)
    if m:
        agent = m.group(1)
        snum = int(m.group(2)) if m.group(2) else 0
        session_label = f"s{m.group(2)}" if m.group(2) else "s0"
        return agent, session_label, snum
    # UUID or other format: use filename stem as agent id, session 0
    return stem, "s0", 0


def infer_agent_from_prompt(lines: list) -> str | None:
    """
    For UUID-named files, extract agent id from the first user message
    which begins with 'You are agentN'.
    """
    for line in lines[:10]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "user":
            continue
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            m = re.match(r"You are (agent\d+)", content)
            if m:
                return m.group(1)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    m = re.match(r"You are (agent\d+)", block)
                    if m:
                        return m.group(1)
    return None


def extract_events_from_file(
    jsonl_path: Path,
    domain_name: str,
    verbose: bool = False,
) -> list[dict]:
    """
    Parse one agent .jsonl log file and return a list of event dicts.

    The approach:
      1. First pass: collect all tool_use blocks from assistant entries,
         keyed by tool_use_id -> (name, file_path, entry_index).
      2. Second pass: for each user entry with a tool_result, look up the
         corresponding tool_use, check if the file is shared, and emit an event.
      3. For Bash tool uses, scan the command for shared file references.
    """
    filename = jsonl_path.name
    agent_from_name, session_label, session_sort = parse_session_label(filename)

    try:
        raw_lines = jsonl_path.read_text(errors="replace").splitlines()
    except OSError as e:
        print(f"  WARNING: cannot read {jsonl_path}: {e}", file=sys.stderr)
        return []

    # Infer agent id for UUID-named files
    agent_id = agent_from_name
    if not re.match(r'^agent\d+$', agent_id):
        inferred = infer_agent_from_prompt(raw_lines)
        if inferred:
            agent_id = inferred

    # First pass: index all tool_use blocks
    # tool_uses[id] = {"name": str, "file": str, "entry_idx": int, "cmd": str}
    tool_uses: dict[str, dict] = {}
    for entry_idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            tid = block.get("id", "")
            if not tid:
                continue
            name = block.get("name", "")
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                inp = {}
            # File path: present for Read/Write/Edit
            fp = inp.get("file_path", inp.get("path", ""))
            # Command: present for Bash
            cmd = inp.get("command", "")
            tool_uses[tid] = {
                "name": name,
                "file": os.path.basename(fp) if fp else "",
                "full_path": fp,
                "cmd": cmd,
                "entry_idx": entry_idx,
            }

    # Second pass: match tool_results in user entries
    events: list[dict] = []
    for entry_idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "user":
            continue

        # Timestamp: may be ISO8601 string or absent
        ts_raw = obj.get("timestamp", "")
        has_real_ts = bool(ts_raw)

        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tid = block.get("tool_use_id", "")
            if not tid or tid not in tool_uses:
                continue

            tu = tool_uses[tid]
            op = tu["name"]
            fname = tu["file"]

            # Determine timestamp / sort key
            if has_real_ts:
                ts = ts_raw
                sort_key = ts_raw  # ISO8601 sorts lexicographically
            else:
                # Synthesize: pad session number + entry index
                # Format: "0000000106_000042" — sorts correctly
                sort_key = f"{session_sort:010d}_{entry_idx:06d}"
                ts = f"synthetic://{session_label}/{entry_idx:06d}"

            # --- Handle Bash commands ---
            if op == "Bash":
                cmd = tu["cmd"]
                # Check for writes (>> or tee -a)
                for pat in BASH_WRITE_PATTERNS:
                    m = pat.search(cmd)
                    if m:
                        matched_file = os.path.basename(m.group(1))
                        if matched_file in SHARED_FILES:
                            events.append({
                                "ts": ts,
                                "_sort_key": sort_key,
                                "agent": agent_id,
                                "op": "Write",
                                "file": matched_file,
                                "session": session_label,
                                "seq": entry_idx,
                                "domain": domain_name,
                            })
                            if verbose:
                                print(f"  Bash-Write {matched_file} at {ts}")
                # Check for reads
                for pat in BASH_READ_PATTERNS:
                    m = pat.search(cmd)
                    if m:
                        matched_file = os.path.basename(m.group(1))
                        if matched_file in SHARED_FILES:
                            events.append({
                                "ts": ts,
                                "_sort_key": sort_key,
                                "agent": agent_id,
                                "op": "Read",
                                "file": matched_file,
                                "session": session_label,
                                "seq": entry_idx,
                                "domain": domain_name,
                            })
                            if verbose:
                                print(f"  Bash-Read {matched_file} at {ts}")
                continue

            # --- Handle Read / Write / Edit ---
            if fname not in SHARED_FILES:
                continue
            if op not in ("Read", "Write", "Edit", "Grep"):
                continue

            events.append({
                "ts": ts,
                "_sort_key": sort_key,
                "agent": agent_id,
                "op": op,
                "file": fname,
                "session": session_label,
                "seq": entry_idx,
                "domain": domain_name,
            })
            if verbose:
                print(f"  {op} {fname} at {ts}")

    return events


def process_domain(domain_dir: Path, verbose: bool = False) -> int:
    """
    Process all log files for one domain, write events.jsonl.
    Returns the number of events written.
    """
    logs_dir = domain_dir / "logs"
    if not logs_dir.is_dir():
        return 0

    log_files = sorted(logs_dir.glob("*.jsonl"))
    if not log_files:
        return 0

    domain_name = domain_dir.name
    all_events: list[dict] = []

    for lf in log_files:
        if verbose:
            print(f"  Processing {lf.name}")
        evs = extract_events_from_file(lf, domain_name, verbose=verbose)
        all_events.extend(evs)

    if not all_events:
        return 0

    # Sort: real timestamps sort before synthetic ones (real: starts with digit,
    # synthetic: starts with "synthetic://")
    def sort_key(ev):
        sk = ev["_sort_key"]
        # real ISO8601 -> prefix "1", synthetic -> "0" (sorts earlier, but we
        # actually want real timestamps to be used when mixing is possible)
        # Since we never mix (all logs have timestamps), just sort by _sort_key.
        return sk

    all_events.sort(key=sort_key)

    # Strip internal sort key before writing
    output_events = [{k: v for k, v in ev.items() if k != "_sort_key"}
                     for ev in all_events]

    out_path = domain_dir / "events.jsonl"
    with open(out_path, "w") as f:
        for ev in output_events:
            f.write(json.dumps(ev) + "\n")

    print(f"  [{domain_name}] wrote {len(output_events)} events → {out_path}")
    return len(output_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", help="Process only this domain name")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each event as it is processed")
    parser.add_argument("--domains-dir", default=str(DOMAINS_DIR),
                        help=f"Path to domains directory (default: {DOMAINS_DIR})")
    args = parser.parse_args()

    domains_dir = Path(args.domains_dir)
    if not domains_dir.is_dir():
        print(f"ERROR: domains directory not found: {domains_dir}", file=sys.stderr)
        sys.exit(1)

    if args.domain:
        candidates = [domains_dir / args.domain]
    else:
        candidates = sorted(d for d in domains_dir.iterdir() if d.is_dir())

    total_events = 0
    domains_processed = 0
    for domain_dir in candidates:
        if not domain_dir.is_dir():
            print(f"WARNING: {domain_dir} not found", file=sys.stderr)
            continue
        if args.verbose:
            print(f"\n=== {domain_dir.name} ===")
        n = process_domain(domain_dir, verbose=args.verbose)
        if n > 0:
            total_events += n
            domains_processed += 1

    print(f"\nDone: {domains_processed} domains, {total_events} total events")


if __name__ == "__main__":
    main()

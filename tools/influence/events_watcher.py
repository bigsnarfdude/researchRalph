#!/usr/bin/env python3
"""
events_watcher.py — real-time events.jsonl writer for a running RRMA agent

Tails an agent's .jsonl log as it is written by claude CLI and appends
events to the domain's events.jsonl in real time.

The harness (launch-agents.sh) launches this as a background process per
agent, co-running alongside claude.  It exits when claude finishes
(detects EOF on the log file after a quiet period).

Usage (called by launch-agents.sh):
  python3 tools/influence/events_watcher.py \
      --agent agent0 \
      --log    domains/nirenberg-1d/logs/agent0.jsonl \
      --events domains/nirenberg-1d/events.jsonl \
      --session s1 \
      --domain  nirenberg-1d

The watcher:
  1. Opens the agent log file in follow mode (like tail -f)
  2. Parses each new line as it appears
  3. Matches tool_use -> tool_result pairs (same logic as extract_events.py)
  4. Appends qualifying events to events.jsonl with a real timestamp
  5. Exits after IDLE_TIMEOUT seconds of no new data (agent finished)

Shared file set and detection logic mirrors extract_events.py exactly.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add tools/ to path
TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from influence.extract_events import (
    SHARED_FILES,
    BASH_WRITE_PATTERNS,
    BASH_READ_PATTERNS,
)

# How long to wait for new data before declaring the agent finished
IDLE_TIMEOUT = 120  # seconds
# How often to poll when no new data
POLL_INTERVAL = 1.0  # seconds


def follow(path: Path, poll_interval: float = POLL_INTERVAL, idle_timeout: float = IDLE_TIMEOUT):
    """
    Generator that yields lines from a file as they appear (tail -f semantics).
    Yields None after idle_timeout seconds of no new data (signals EOF).
    """
    last_data_time = time.monotonic()
    with open(path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                last_data_time = time.monotonic()
                yield line
            else:
                if time.monotonic() - last_data_time > idle_timeout:
                    yield None  # signal: done
                    return
                time.sleep(poll_interval)


def watch(
    agent: str,
    log_path: Path,
    events_path: Path,
    session: str,
    domain: str,
    idle_timeout: float = IDLE_TIMEOUT,
    verbose: bool = False,
) -> int:
    """
    Watch log_path, extract shared-file events, append to events_path.
    Returns the number of events written.
    """
    # Wait for log file to be created (claude may not have started yet)
    waited = 0
    while not log_path.exists():
        time.sleep(1)
        waited += 1
        if waited > 60:
            print(f"[watcher/{agent}] Gave up waiting for {log_path}", file=sys.stderr)
            return 0

    # Open events file for appending
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_file = open(events_path, "a")

    # In-flight tool_use state: tid -> {name, file, cmd, entry_idx}
    tool_uses: dict[str, dict] = {}
    entry_idx = 0
    events_written = 0

    try:
        for raw_line in follow(log_path, idle_timeout=idle_timeout):
            if raw_line is None:
                # Idle timeout reached — agent done
                break

            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                obj = json.loads(raw_line)
            except Exception:
                entry_idx += 1
                continue

            if not isinstance(obj, dict):
                entry_idx += 1
                continue

            t = obj.get("type")

            if t == "assistant":
                msg = obj.get("message", {})
                if not isinstance(msg, dict):
                    entry_idx += 1
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
                    fp = inp.get("file_path", inp.get("path", ""))
                    cmd = inp.get("command", "")
                    tool_uses[tid] = {
                        "name": name,
                        "file": os.path.basename(fp) if fp else "",
                        "cmd": cmd,
                        "entry_idx": entry_idx,
                    }

            elif t == "user":
                # Real timestamp is on user entries in current Claude CLI versions
                ts = obj.get("timestamp", "")
                if not ts:
                    # Fallback: use current wall time
                    ts = datetime.now(timezone.utc).isoformat()

                msg = obj.get("message", {})
                if not isinstance(msg, dict):
                    entry_idx += 1
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    entry_idx += 1
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tid = block.get("tool_use_id", "")
                    if not tid or tid not in tool_uses:
                        continue

                    tu = tool_uses.pop(tid)  # consume it
                    op = tu["name"]
                    fname = tu["file"]

                    events_to_emit = []

                    if op == "Bash":
                        cmd = tu["cmd"]
                        for pat in BASH_WRITE_PATTERNS:
                            m = pat.search(cmd)
                            if m:
                                mf = os.path.basename(m.group(1))
                                if mf in SHARED_FILES:
                                    events_to_emit.append(("Write", mf))
                        for pat in BASH_READ_PATTERNS:
                            m = pat.search(cmd)
                            if m:
                                mf = os.path.basename(m.group(1))
                                if mf in SHARED_FILES:
                                    events_to_emit.append(("Read", mf))
                    elif op in ("Read", "Write", "Edit", "Grep") and fname in SHARED_FILES:
                        events_to_emit.append((op, fname))

                    for emit_op, emit_file in events_to_emit:
                        ev = {
                            "ts": ts,
                            "agent": agent,
                            "op": emit_op,
                            "file": emit_file,
                            "session": session,
                            "seq": entry_idx,
                            "domain": domain,
                        }
                        events_file.write(json.dumps(ev) + "\n")
                        events_file.flush()
                        events_written += 1
                        if verbose:
                            print(f"[watcher/{agent}] {emit_op} {emit_file} @ {ts}")

            entry_idx += 1

    finally:
        events_file.close()

    if verbose or events_written > 0:
        print(f"[watcher/{agent}] Done: {events_written} events written to {events_path}")
    return events_written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="Agent ID (e.g. agent0)")
    parser.add_argument("--log", required=True, help="Path to agent .jsonl log file")
    parser.add_argument("--events", required=True, help="Path to events.jsonl to append to")
    parser.add_argument("--session", default="s0", help="Session label (e.g. s1)")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT,
                        help=f"Seconds of silence before declaring agent done (default: {IDLE_TIMEOUT})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    n = watch(
        agent=args.agent,
        log_path=Path(args.log),
        events_path=Path(args.events),
        session=args.session,
        domain=args.domain,
        idle_timeout=args.idle_timeout,
        verbose=args.verbose,
    )
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()

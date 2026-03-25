#!/usr/bin/env python3
"""
Extract agent reasoning traces from Claude Code jsonl session logs.
Produces a compressed markdown artifact suitable for case study / blog post.

Usage:
    python3 extract_trace.py <session.jsonl> [--output trace.md]
    python3 extract_trace.py --all-sessions <domain_dir>
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


def parse_session(path: Path) -> list[dict]:
    """Parse a jsonl session into a list of events."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def extract_turns(events: list[dict]) -> list[dict]:
    """Extract meaningful turns: user messages, assistant thinking, tool calls/results."""
    turns = []
    for e in events:
        t = e.get("type")
        msg = e.get("message", {})

        # Initial prompt
        if t == "user" and e.get("parentUuid") is None:
            content = msg.get("content", "")
            if isinstance(content, str):
                turns.append({"type": "prompt", "text": content[:300], "ts": e.get("timestamp", "")})

        # Assistant reasoning
        elif t == "assistant":
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    turns.append({
                        "type": "think",
                        "text": block["text"].strip(),
                        "ts": e.get("timestamp", "")
                    })
                elif block.get("type") == "tool_use":
                    turns.append({
                        "type": "tool_call",
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                        "ts": e.get("timestamp", "")
                    })

        # Tool results
        elif t == "tool":
            for block in msg.get("content", []):
                if block.get("type") == "tool_result":
                    for inner in block.get("content", []):
                        if inner.get("type") == "text":
                            turns.append({
                                "type": "tool_result",
                                "text": inner["text"][:500],
                                "ts": e.get("timestamp", "")
                            })
    return turns


def compress_trace(turns: list[dict], session_id: str) -> str:
    """Render turns into a readable markdown trace."""
    lines = [f"# Agent Trace — {session_id[:8]}\n"]

    prev_think = None
    for turn in turns:
        tt = turn["type"]

        if tt == "prompt":
            lines.append(f"**[START]** {turn['text'][:200]}...\n")

        elif tt == "think":
            text = turn["text"]
            # Skip near-duplicate consecutive thoughts
            if prev_think and text[:80] == prev_think[:80]:
                continue
            prev_think = text
            lines.append(f"> {text}\n")

        elif tt == "tool_call":
            name = turn["name"]
            inp = turn["input"]
            if name == "Bash":
                cmd = inp.get("command", "")[:120]
                lines.append(f"```bash\n{cmd}\n```\n")
            elif name in ("Write", "Edit"):
                fp = inp.get("file_path", "")
                lines.append(f"**{name}** `{fp}`\n")
            elif name == "Read":
                fp = inp.get("file_path", "")
                lines.append(f"**Read** `{fp}`\n")

        elif tt == "tool_result":
            text = turn["text"].strip()
            if text and len(text) > 10:
                lines.append(f"<details><summary>output</summary>\n\n```\n{text[:400]}\n```\n\n</details>\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", help="Path to .jsonl session file")
    parser.add_argument("--all-sessions", help="Extract all sessions in a domain dir")
    parser.add_argument("--output", "-o", help="Output markdown file")
    args = parser.parse_args()

    if args.all_sessions:
        domain_dir = Path(args.all_sessions)
        # Look in local .claude or passed dir
        candidates = list(domain_dir.glob("*.jsonl"))
        if not candidates:
            print(f"No .jsonl files found in {domain_dir}", file=sys.stderr)
            sys.exit(1)

        all_traces = []
        for jsonl in sorted(candidates):
            events = parse_session(jsonl)
            turns = extract_turns(events)
            trace = compress_trace(turns, jsonl.stem)
            all_traces.append(trace)
            print(f"  {jsonl.name}: {len(events)} events, {len(turns)} turns")

        combined = "\n\n---\n\n".join(all_traces)
        out = Path(args.output) if args.output else domain_dir / "traces.md"
        out.write_text(combined)
        print(f"\nWrote {out}")

    elif args.session:
        jsonl = Path(args.session)
        events = parse_session(jsonl)
        turns = extract_turns(events)
        trace = compress_trace(turns, jsonl.stem)
        if args.output:
            Path(args.output).write_text(trace)
            print(f"Wrote {args.output}")
        else:
            print(trace)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

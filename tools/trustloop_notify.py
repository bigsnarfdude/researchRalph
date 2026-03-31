#!/usr/bin/env python3
"""
TrustLoop Notify — pipe any report to Slack (or stdout for testing).

Usage:
    # Status report to Slack
    python3 tools/trustloop_notify.py status domains/gpt2-tinystories-v44

    # Only notify if reds exist
    python3 tools/trustloop_notify.py status domains/gpt2-tinystories-v44 --on-red

    # Experiment event (breakthrough or crash)
    python3 tools/trustloop_notify.py event domains/gpt2-tinystories-v44 --exp exp055

    # Summary at end of run
    python3 tools/trustloop_notify.py summary domains/gpt2-tinystories-v44

    # Dry run (stdout, no Slack)
    python3 tools/trustloop_notify.py status domains/gpt2-tinystories-v44 --dry-run

Set TRUSTLOOP_SLACK_WEBHOOK in env to enable Slack delivery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Reuse stoplight machinery
sys.path.insert(0, str(Path(__file__).parent))
from trustloop_stoplight import build_stoplight, load_results, load_text, scores, breakthrough_indices


WEBHOOK_URL = os.environ.get("TRUSTLOOP_SLACK_WEBHOOK", "")


def send_slack(text: str, channel: str | None = None) -> bool:
    if not WEBHOOK_URL:
        return False
    payload = {"text": text}
    if channel:
        payload["channel"] = channel
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Slack send failed: {e}", file=sys.stderr)
        return False


# ── Report formatters (Slack mrkdwn) ─────────────────────────────────────────

SLACK_ICONS = {"GREEN": ":large_green_circle:", "YELLOW": ":large_yellow_circle:", "RED": ":red_circle:"}
TERM_ICONS = {"GREEN": "\033[32m●\033[0m", "YELLOW": "\033[33m●\033[0m", "RED": "\033[31m●\033[0m"}


def fmt_status(domain: Path, slack: bool = False) -> tuple[str, int, int]:
    """Format status report. Returns (text, red_count, yellow_count)."""
    sections = build_stoplight(domain)
    all_lights = [l for _, ls in sections for l in ls]
    reds = sum(1 for l in all_lights if l.color == "RED")
    yellows = sum(1 for l in all_lights if l.color == "YELLOW")
    greens = sum(1 for l in all_lights if l.color == "GREEN")

    icons = SLACK_ICONS if slack else TERM_ICONS
    lines = []

    if slack:
        lines.append(f"*STOPLIGHT — {domain.name}*")
    else:
        lines.append(f"STOPLIGHT — {domain.name}")

    for section_name, lights in sections:
        for l in lights:
            icon = icons[l.color]
            lines.append(f"{icon} *{l.name}*  {l.summary}" if slack else f"{icon} {l.name:24s}{l.summary}")

    tally = []
    if reds:
        tally.append(f"{reds} RED")
    if yellows:
        tally.append(f"{yellows} YELLOW")
    tally.append(f"{greens} GREEN")
    lines.append(f"{' | '.join(tally)}  ({len(all_lights)} checks)")

    return "\n".join(lines), reds, yellows


def fmt_event(domain: Path, exp_id: str, slack: bool = False) -> str:
    """Format single experiment event."""
    rows = load_results(domain / "results.tsv")
    row = None
    for r in rows:
        if r.get("exp_id") == exp_id:
            row = r
            break
    if not row:
        return f"Experiment {exp_id} not found in {domain.name}"

    status = row.get("status", "?")
    score = row.get("score", "?")
    desc = row.get("description", "")
    agent = row.get("agent", "?")

    # Determine if breakthrough
    bt_idxs = breakthrough_indices(rows)
    is_bt = any(rows[i].get("exp_id") == exp_id for i in bt_idxs)

    if status == "crash":
        icon = ":boom:" if slack else "!!"
        label = "CRASH"
    elif is_bt:
        icon = ":trophy:" if slack else ">>"
        label = "BREAKTHROUGH"
    else:
        return ""  # Don't notify on plateau/discard

    if slack:
        return f"{icon} *{label}* — {domain.name}\n*{exp_id}* ({agent}): {score}\n{desc}"
    else:
        return f"{icon} {label} — {domain.name}\n{exp_id} ({agent}): {score}\n{desc}"


def fmt_summary(domain: Path, slack: bool = False) -> str:
    """Format end-of-run summary."""
    rows = load_results(domain / "results.tsv")
    if not rows:
        return f"No experiments in {domain.name}"

    valid = scores(rows)
    total = len(rows)
    crashes = sum(1 for r in rows if r.get("status") == "crash")
    keeps = sum(1 for r in rows if r.get("status") == "keep")
    bts = len(breakthrough_indices(rows))

    baseline = valid[0] if valid else 0
    best = min(valid) if valid else 0
    best_id = None
    for r in rows:
        try:
            if float(r["score"]) == best:
                best_id = r.get("exp_id", "?")
                break
        except (ValueError, TypeError, KeyError):
            pass

    delta = baseline - best
    pct = delta / baseline * 100 if baseline else 0

    lines = []
    b = "*" if slack else ""
    lines.append(f"{b}SUMMARY — {domain.name}{b}")
    lines.append(f"Baseline: {baseline:.6f} → Best: {best:.6f} ({best_id})")
    lines.append(f"Improvement: -{delta:.4f} (-{pct:.1f}%)")
    lines.append(f"Experiments: {total} | Kept: {keeps} | Crashes: {crashes} | Breakthroughs: {bts}")
    lines.append(f"Efficiency: {bts}/{total} = {bts/total:.0%} breakthrough rate")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrustLoop Notify")
    parser.add_argument("report", choices=["status", "event", "summary"])
    parser.add_argument("domain", help="Path to domain directory")
    parser.add_argument("--exp", help="Experiment ID (for event report)")
    parser.add_argument("--on-red", action="store_true", help="Only send if reds exist")
    parser.add_argument("--on-red-yellow", action="store_true", help="Only send if reds or yellows exist")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout, don't send")
    parser.add_argument("--channel", help="Override Slack channel")
    args = parser.parse_args()

    domain = Path(args.domain)
    use_slack = bool(WEBHOOK_URL) and not args.dry_run

    if args.report == "status":
        text, reds, yellows = fmt_status(domain, slack=use_slack)
        if args.on_red and reds == 0:
            return
        if args.on_red_yellow and reds == 0 and yellows == 0:
            return
    elif args.report == "event":
        if not args.exp:
            print("--exp required for event report", file=sys.stderr)
            sys.exit(1)
        text = fmt_event(domain, args.exp, slack=use_slack)
        if not text:
            return  # Not a notable event
    elif args.report == "summary":
        text = fmt_summary(domain, slack=use_slack)

    if use_slack:
        ok = send_slack(text, channel=args.channel)
        if not ok:
            print(text)
            print("\n(Slack delivery failed, printed to stdout)", file=sys.stderr)
    else:
        print(text)
        if not args.dry_run and not WEBHOOK_URL:
            print("\n(Set TRUSTLOOP_SLACK_WEBHOOK to enable Slack delivery)", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
refresh_context.py — v4.6 context optimizer for RRMA agents.

Generates two compact files that replace raw blackboard + results.tsv grep-ing:

  1. stoplight.md  — 30-line compressed run state from TrustLoop scorer
  2. recent_experiments.md — last N experiments with structured records

Agents read these instead of the full blackboard (627+ lines) and results.tsv.

Usage:
    python3 tools/refresh_context.py /path/to/domain
    python3 tools/refresh_context.py domains/gpt2-tinystories-v44 --last 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add tools/ to import path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from trustloop_scorer import (
    score_domain,
    DomainReport,
    ScoredExperiment,
    _run_status,
    OUTCOME_SYMBOLS,
)


def generate_stoplight(report: DomainReport, domain_dir: Path) -> str:
    """Generate ~30-line stoplight.md — the compressed run state.

    Replaces reading 500+ lines of blackboard.md. Contains:
    - Run health (status, best, stagnation)
    - What's working (winning strategies)
    - Dead ends (don't re-test these)
    - Closed brackets (parameter → optimum, from blackboard)
    - Active gaps (what hasn't been tried)
    """
    lines = []
    status = _run_status(report)

    # --- Header ---
    lines.append(f"# Stoplight — {report.domain}")
    lines.append(f"Status: {status} | Best: {report.best_score} ({report.best_exp}) | "
                 f"Experiments: {report.total_experiments} | "
                 f"Stagnation: {report.stagnation_depth} since last breakthrough")
    lines.append("")

    # --- What's working ---
    winning = [i for i in report.insights if i.kind == "winning_strategy"]
    if winning:
        lines.append("## What works")
        for i in winning[:5]:
            lines.append(f"- {i.message}")
        lines.append("")

    # --- Dead ends ---
    dead_ends = [i for i in report.insights if i.kind == "dead_end"]
    if dead_ends:
        lines.append("## Dead ends — do NOT retry")
        for i in dead_ends[:8]:
            lines.append(f"- {i.message}")
        lines.append("")

    # --- Recurring problems ---
    recurring = [i for i in report.insights if i.kind == "recurring_mistake"]
    if recurring:
        lines.append("## Recurring problems")
        for i in recurring[:3]:
            lines.append(f"- {i.message}")
        lines.append("")

    # --- Gaps ---
    unaddressed = [i for i in report.insights if i.kind == "unaddressed_desires"]
    if unaddressed:
        lines.append("## Gaps — unexplored")
        for i in unaddressed[:5]:
            lines.append(f"- {i.message}")
        lines.append("")

    # --- Agent summary ---
    if len(report.agents) > 1:
        lines.append("## Agents")
        for a in report.agents:
            best_s = f"{a.best_score}" if a.best_score is not None else "—"
            lines.append(f"- {a.agent_id}: {a.total_experiments} exp, "
                         f"{a.breakthroughs} breakthroughs, "
                         f"rate {a.success_rate:.0%}, best {best_s}")
        lines.append("")

    # --- Anomalies ---
    alerts = [a for a in report.anomalies if a.severity == "ALERT"]
    if alerts:
        lines.append("## Alerts")
        for a in alerts[:3]:
            lines.append(f"- {a.category}: {a.message}")
        lines.append("")

    # --- Recent blackboard tail (last 20 lines of substance) ---
    bb_path = domain_dir / "blackboard.md"
    if bb_path.exists():
        bb_lines = bb_path.read_text().splitlines()
        # Get last 20 non-empty lines
        substance = [l for l in bb_lines if l.strip() and not l.startswith("# ")]
        if substance:
            tail = substance[-20:]
            lines.append("## Recent blackboard (last 20 entries)")
            lines.extend(tail)
            lines.append("")

    return "\n".join(lines)


def generate_recent_experiments(report: DomainReport, last_n: int = 5) -> str:
    """Generate recent_experiments.md — structured per-experiment records.

    Replaces grep-ing results.tsv. Each record has:
    - ID, score, outcome class, agent
    - Description (what was tried)
    - Verdict (keep/discard/crash)
    - Delta from best
    """
    lines = []
    lines.append(f"# Recent Experiments — {report.domain}")
    lines.append("")

    if not report.experiments:
        lines.append("No experiments yet.")
        return "\n".join(lines)

    # Current best for delta calculation
    best = report.best_score
    lower = report.score_direction == "lower"

    # Summary line
    lines.append(f"**Best: {best} ({report.best_exp})** | "
                 f"Total: {report.total_experiments} | "
                 f"Breakthroughs: {report.breakthroughs} | "
                 f"Crashes: {report.crashes}")
    lines.append("")

    # Last N experiments
    recent = report.experiments[-last_n:]
    for exp in recent:
        sym = OUTCOME_SYMBOLS.get(exp.outcome_class, "?")
        score_str = f"{exp.score}" if exp.score is not None else "CRASH"

        # Delta from best
        delta_str = ""
        if exp.score is not None and best is not None:
            if lower:
                delta = exp.score - best
                delta_str = f" (Δ {delta:+.4f} from best)" if abs(delta) > 0.0001 else " (= best)"
            else:
                delta = exp.score - best
                delta_str = f" (Δ {delta:+.4f} from best)" if abs(delta) > 0.0001 else " (= best)"

        lines.append(f"### {sym} {exp.exp_id} — {score_str}{delta_str}")
        lines.append(f"- **Agent:** {exp.agent} | **Design:** {exp.design} | **Status:** {exp.status}")
        lines.append(f"- **What:** {exp.description}")
        lines.append(f"- **Outcome:** {exp.outcome_class}")
        if exp.redundant_with:
            lines.append(f"- **Redundant with:** {exp.redundant_with}")
        lines.append("")

    # Score trajectory (all experiments, compact)
    valid = [(e.exp_id, e.score) for e in report.experiments if e.score is not None]
    if len(valid) > last_n:
        lines.append("## Score trajectory (all)")
        # Show as compact table
        lines.append("| exp | score | outcome |")
        lines.append("|-----|-------|---------|")
        for exp in report.experiments:
            if exp.score is not None:
                sym = OUTCOME_SYMBOLS.get(exp.outcome_class, "?")
                lines.append(f"| {exp.exp_id} | {exp.score} | {sym} {exp.outcome_class} |")
            else:
                lines.append(f"| {exp.exp_id} | CRASH | ✗ CRASH |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate agent context files (v4.6)")
    parser.add_argument("domain_dir", help="Path to domain directory")
    parser.add_argument("--last", type=int, default=5, help="Number of recent experiments to show (default: 5)")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)
    if not domain_dir.exists():
        domain_dir = REPO_ROOT / args.domain_dir
    if not domain_dir.exists():
        print(f"Domain not found: {args.domain_dir}", file=sys.stderr)
        sys.exit(1)

    # Run scorer (no traces needed — fast)
    report = score_domain(domain_dir, with_traces=False)

    # Generate files
    stoplight = generate_stoplight(report, domain_dir)
    recent = generate_recent_experiments(report, last_n=args.last)

    # Write atomically
    stoplight_path = domain_dir / "stoplight.md"
    recent_path = domain_dir / "recent_experiments.md"

    stoplight_path.write_text(stoplight)
    recent_path.write_text(recent)

    sl_lines = stoplight.count("\n")
    re_lines = recent.count("\n")
    print(f"[refresh] stoplight.md ({sl_lines} lines) + recent_experiments.md ({re_lines} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()

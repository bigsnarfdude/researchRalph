#!/usr/bin/env python3
"""
diagnose.py — v4.5 process quality diagnosis using TrustLoop scorer.

Drop-in replacement for diagnose.sh. Reads domain artifacts, runs the full
TrustLoop scorer (classification, anomaly detection, telemetry, workflow
validation), and outputs a single decision to stdout:

    CONTINUE | NUDGE | STOP_HACKING | STOP_DONE | REDESIGN | TOO_EARLY

Detailed report goes to stderr (same convention as diagnose.sh).

Usage:
    python3 v4/diagnose.py /path/to/domain
    python3 v4/diagnose.py domains/gpt2-tinystories-v44
"""

import sys
from pathlib import Path

# Add tools/ to import path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from trustloop_scorer import (
    score_domain,
    format_report,
    DomainReport,
)


def compute_process_quality(report: DomainReport) -> int:
    """Compute PQ 0-30 from scorer data (replaces bash grep counts)."""
    pq = 0
    tel = report.telemetry

    # Papers cited — check blackboard for academic references
    domain_dir = Path(report.domain) if Path(report.domain).exists() else REPO_ROOT / "domains" / report.domain
    bb_path = domain_dir / "blackboard.md"
    bb_text = bb_path.read_text().lower() if bb_path.exists() else ""

    papers = sum(1 for kw in ["arxiv", "paper", "et al.", "neurips", "icml", "iclr"]
                 if kw in bb_text)
    if papers > 0: pq += 3
    if papers > 3: pq += 3

    # Explanatory reasoning
    explanations = sum(1 for kw in ["because", "mechanism", "why ", "hypothesis",
                                     "key insight", "observation", "confirms", "suggests"]
                       if kw in bb_text)
    if explanations > 3: pq += 3
    if explanations > 10: pq += 3

    # Ablations
    ablations = sum(1 for kw in ["ablation", "vs ", "compared to", "worse than",
                                  "better than", "no improvement", "sweep"]
                    if kw in bb_text)
    if ablations > 0: pq += 3
    if ablations > 3: pq += 3

    # Simplifications
    simplifications = sum(1 for kw in ["simpler", "removed", "dropped", "fewer",
                                        "simplified", "unnecessary"]
                          if kw in bb_text)
    if simplifications > 0: pq += 3

    # Unique designs
    designs = set(e.design for e in report.experiments if e.design)
    if len(designs) > 5: pq += 3

    # Blackboard depth
    bb_lines = bb_text.count("\n")
    if bb_lines > 100: pq += 3

    # Telemetry quality (v4.5 upgrade — actual content, not just counts)
    if len(tel.desires) > 0: pq += 3
    if len(tel.learnings) > 5: pq += 3

    return min(pq, 30)


def decide(report: DomainReport, domain_dir: Path) -> str:
    """Make stopping/continue decision from scorer report."""

    total = report.total_experiments
    if total < 8:
        return "TOO_EARLY"

    pq = compute_process_quality(report)

    # --- Stagnation and flatness ---
    stagnation = report.stagnation_depth

    # Flat: check if last N experiments haven't improved much
    exps = [e for e in report.experiments if e.score is not None]
    flat = False
    micro_flat = False
    if len(exps) > 20:
        last_20 = exps[-20:]
        prior = exps[:-20]
        if prior:
            lower = report.score_direction == "lower"
            best_last_20 = min(e.score for e in last_20) if lower else max(e.score for e in last_20)
            best_prior = min(e.score for e in prior) if lower else max(e.score for e in prior)
            if best_prior != 0:
                delta = abs(best_last_20 - best_prior) / abs(best_prior)
                flat = delta < 0.01

    if len(exps) > 15 and report.best_score is not None:
        last_10 = exps[-10:]
        lower = report.score_direction == "lower"
        best_last_10 = min(e.score for e in last_10) if lower else max(e.score for e in last_10)
        if report.best_score != 0:
            delta = abs(best_last_10 - report.best_score) / abs(report.best_score)
            micro_flat = delta < 0.005

    # Axis diversity in last 10
    recent_designs = set(e.design for e in report.experiments[-10:] if e.design)
    axis_diverse = len(recent_designs) > 3 or total <= 15

    # Scaffold desires (agents asking for run.sh/tool changes)
    scaffold_kw = ["parallel", "run.sh", "timeout", "evaluator", "batch", "atomic"]
    scaffold_desires = sum(
        1 for d in report.telemetry.desires
        if any(kw in d.lower() for kw in scaffold_kw)
    )

    # Blind spots from meta-blackboard
    blind_spots = 0
    meta_bb = domain_dir / "meta-blackboard.md"
    if meta_bb.exists():
        in_blind = False
        for line in meta_bb.read_text().splitlines():
            if "blind spot" in line.lower():
                in_blind = True
            elif line.startswith("## ") and in_blind:
                in_blind = False
            elif in_blind and line.startswith("- "):
                blind_spots += 1

    # --- Anomaly-based escalation (v4.5 new) ---
    alerts = [a for a in report.anomalies if a.severity == "ALERT"]
    crash_streaks = [a for a in alerts if a.category == "crash_streak"]
    deep_stag = [a for a in alerts if a.category == "deep_stagnation"]

    # --- Decision matrix ---

    # Low PQ after enough experiments = hacking
    if pq < 10 and total > 15:
        return "STOP_HACKING"

    # Crash streak alert = something is fundamentally broken
    if crash_streaks and pq >= 10:
        return "NUDGE"

    # Scaffold desires = agents blocked by missing tools
    if scaffold_desires >= 3 and pq >= 10 and total > 10:
        return "NUDGE"

    # Micro-flat + low diversity = NUDGE
    if micro_flat and not axis_diverse and pq >= 10:
        return "NUDGE"

    # Stagnation without full flatness = NUDGE
    if stagnation > 10 and not flat and pq >= 10:
        return "NUDGE"

    # Deep stagnation alert from anomaly detector
    if deep_stag and pq >= 10:
        return "NUDGE"

    # Flat + no blind spots + deep stagnation = done
    if flat and pq >= 10 and blind_spots == 0 and stagnation > 10:
        return "STOP_DONE"

    # Flat + blind spots = redesign
    if flat and pq >= 10 and blind_spots > 0 and stagnation > 10:
        return "REDESIGN"

    return "CONTINUE"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 v4/diagnose.py /path/to/domain", file=sys.stderr)
        sys.exit(1)

    domain_dir = Path(sys.argv[1])
    if not domain_dir.exists():
        # Try relative to repo root
        domain_dir = REPO_ROOT / sys.argv[1]
    if not domain_dir.exists():
        print(f"Domain not found: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    # Run full scorer
    report = score_domain(domain_dir, with_traces=True)

    # Print detailed report to stderr
    print(format_report(report), file=sys.stderr)

    # Print PQ details
    pq = compute_process_quality(report)
    print(f"\n[diagnose.py] PQ: {pq}/30", file=sys.stderr)
    print(f"[diagnose.py] Anomalies: {len(report.anomalies)}", file=sys.stderr)
    print(f"[diagnose.py] Workflow: {sum(1 for c in report.workflow_checks if c.passed)}/{len(report.workflow_checks)}", file=sys.stderr)
    print(f"[diagnose.py] Telemetry: {len(report.telemetry.desires)} desires, "
          f"{len(report.telemetry.mistakes)} mistakes, "
          f"{len(report.telemetry.learnings)} learnings", file=sys.stderr)

    # Decide
    decision = decide(report, domain_dir)
    print(f"[diagnose.py] DECISION: {decision}", file=sys.stderr)

    # Output decision to stdout (what outer-loop.sh reads)
    print(decision)


if __name__ == "__main__":
    main()

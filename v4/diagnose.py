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

import re
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


# Numbers that appear next to a score-ish keyword in blackboard prose
SCORE_CLAIM_RE = re.compile(
    r"(?:score|bpb|f1|loss|auroc|accuracy)\s*[:=]?\s*(\d+\.\d+)", re.IGNORECASE
)


def compute_process_quality(report: DomainReport, domain_dir: Path) -> int:
    """Compute PQ 0-30 from oracle-verifiable evidence.

    v4.9: the old version counted keywords in blackboard.md ("arxiv",
    "because", "ablation"...), which the agents being policed could satisfy
    with prose alone. PQ is now grounded in results.tsv — design breadth,
    iteration, multi-agent coverage — and in cross-checking blackboard score
    claims against logged rows. Prose/telemetry signals are capped at 6/30
    so they can never carry a run past the STOP_HACKING threshold (10).
    """
    pq = 0
    bb_path = Path(domain_dir) / "blackboard.md"
    bb_text = bb_path.read_text().lower() if bb_path.exists() else ""
    exps = report.experiments

    # --- A. Evidence from results.tsv (max 15) — written by the oracle, not agents ---
    design_counts: dict = {}
    for e in exps:
        if e.design:
            design_counts[e.design] = design_counts.get(e.design, 0) + 1
    if len(design_counts) > 5:
        pq += 3  # breadth: many distinct designs tried
    if sum(1 for c in design_counts.values() if c >= 2) >= 2:
        pq += 3  # iteration: at least two axes revisited, not one-shot scatter
    if any(c >= 3 for c in design_counts.values()):
        pq += 3  # depth: some axis pursued systematically
    if len({e.agent for e in exps if e.agent}) >= 2:
        pq += 3  # coverage: more than one agent actually logging experiments
    outcomes = {e.outcome_class for e in exps}
    if ({"BREAKTHROUGH", "INCREMENTAL"} & outcomes) and ({"PLATEAU", "REGRESSION"} & outcomes):
        pq += 3  # real exploration produces both wins and losses

    # --- B. Blackboard claims cross-checked against the oracle (max 9, penalty below) ---
    claims = [float(c) for c in SCORE_CLAIM_RE.findall(bb_text)]
    logged = {e.score for e in exps if e.score is not None}
    if claims and logged:
        # 0.5% relative tolerance so honest rounding ("1.05" for 1.048) still matches
        verified = sum(
            1 for c in claims
            if any(abs(c - s) <= max(5e-4, 0.005 * abs(s)) for s in logged)
        )
        frac = verified / len(claims)
        if frac >= 0.8:
            pq += 9
        elif frac >= 0.5:
            pq += 4
        elif len(claims) >= 3:
            pq -= 6  # most cited scores match nothing the oracle logged — fabrication signal

    # --- C. Prose + telemetry signals (capped at 6 — gameable, so never decisive) ---
    prose = 0
    explanations = sum(bb_text.count(kw) for kw in ["because", "mechanism", "hypothesis", "suggests"])
    if explanations > 5:
        prose += 2
    if any(kw in bb_text for kw in ["ablation", "compared to", "vs baseline"]):
        prose += 2
    if len(report.telemetry.learnings) > 5 or len(report.telemetry.mistakes) > 3:
        prose += 2
    pq += min(prose, 6)

    return max(0, min(pq, 30))


def decide(report: DomainReport, domain_dir: Path) -> str:
    """Make stopping/continue decision from scorer report."""

    total = report.total_experiments
    if total < 8:
        return "TOO_EARLY"

    pq = compute_process_quality(report, domain_dir)

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
    pq = compute_process_quality(report, domain_dir)
    print(f"\n[diagnose.py] PQ: {pq}/30", file=sys.stderr)
    print(f"[diagnose.py] Anomalies: {len(report.anomalies)}", file=sys.stderr)
    print(f"[diagnose.py] Workflow: {sum(1 for c in report.workflow_checks if c.passed)}/{len(report.workflow_checks)}", file=sys.stderr)
    print(f"[diagnose.py] Telemetry: {len(report.telemetry.desires)} desires, "
          f"{len(report.telemetry.mistakes)} mistakes, "
          f"{len(report.telemetry.learnings)} learnings", file=sys.stderr)

    # Decide
    decision = decide(report, domain_dir)
    print(f"[diagnose.py] DECISION: {decision}", file=sys.stderr)

    # Write unresolved action items for outer-loop NUDGE handler
    unresolved_gardener = [a for a in report.action_items if not a.resolved and a.owner == "gardener"]
    unresolved_hitl = [a for a in report.action_items if not a.resolved and a.owner == "hitl"]
    missed_checks = [c for c in report.gardener_checks if not c.found]

    # Dead ends from insights
    dead_ends = [i.message for i in report.insights if i.kind == "dead_end"]
    # Tool efficiency issues
    tool_issues = [i.message for i in report.insights if i.kind in ("tool_inefficiency", "agent_cost")]
    # Single-axis stagnation: all recent experiments share one design type
    recent_designs = [e.design for e in report.experiments[-10:] if e.design]
    design_counts = {}
    for d in recent_designs:
        design_counts[d] = design_counts.get(d, 0) + 1
    dominant_axis = None
    if recent_designs and design_counts:
        top_design, top_count = max(design_counts.items(), key=lambda x: x[1])
        if top_count >= len(recent_designs) * 0.7:
            dominant_axis = top_design

    import json as _json
    nudge_data = {
        "decision": decision,
        "pq": pq,
        "gardener_fixes": [{"issue": a.issue, "fix": a.fix, "source": a.source_exp} for a in unresolved_gardener],
        "hitl_fixes": [{"issue": a.issue, "fix": a.fix} for a in unresolved_hitl],
        "missed_checks": [{"issue": c.issue, "expected": c.expected_in_program_md} for c in missed_checks],
        "dead_ends": dead_ends,
        "tool_issues": tool_issues,
        "dominant_axis": dominant_axis,
        "stagnation": report.stagnation_depth,
    }
    nudge_path = domain_dir / ".nudge_data.json"
    nudge_path.write_text(_json.dumps(nudge_data, indent=2))
    print(f"[diagnose.py] Wrote nudge data to {nudge_path}", file=sys.stderr)

    # Output decision to stdout (what outer-loop.sh reads)
    print(decision)


if __name__ == "__main__":
    main()

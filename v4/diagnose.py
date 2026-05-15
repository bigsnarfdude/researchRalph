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


def detect_death_spiral(report: DomainReport) -> bool:
    """
    True if crash rate is accelerating OR rolling score mean is trending worse.
    Death spiral = agents iterating into increasingly unstable territory.
    """
    all_exps = report.experiments
    if len(all_exps) < 12:
        return False

    # Crash rate: last 10 vs prior 10
    recent = all_exps[-10:]
    prior = all_exps[-20:-10] if len(all_exps) >= 20 else all_exps[:-10]
    if prior:
        crash_recent = sum(1 for e in recent if e.score is None) / len(recent)
        crash_prior = sum(1 for e in prior if e.score is None) / len(prior)
        if crash_recent >= 0.4 and crash_recent > crash_prior * 2:
            return True

    # Rolling mean trend: linear regression on last 15 non-crash scores
    scored = [e for e in all_exps if e.score is not None][-15:]
    if len(scored) < 6:
        return False
    lower = report.score_direction == "lower"
    scores = [e.score for e in scored]
    # Flip so "improving" always means decreasing
    if not lower:
        scores = [-s for s in scores]
    n = len(scores)
    xs = list(range(n))
    sx = sum(xs); sy = sum(scores)
    sxy = sum(x * y for x, y in zip(xs, scores))
    sx2 = sum(x * x for x in xs)
    denom = n * sx2 - sx * sx
    if denom == 0:
        return False
    slope = (n * sxy - sx * sy) / denom
    # Spiral: mean rising by more than 1% of best per step, sustained over 15 exps
    if report.best_score and report.best_score != 0:
        threshold = abs(report.best_score) * 0.01
        return slope > threshold
    return False


def detect_false_consensus(domain_dir: Path, report: DomainReport) -> str | None:
    """
    Returns the dominant design axis if 2+ distinct agents are all claiming wins
    from the same design type in the last 15 blackboard CLAIM lines — without
    independent cross-validation.
    Returns None if no false consensus detected.
    """
    bb = domain_dir / "blackboard.md"
    if not bb.exists():
        return None

    claim_pattern = re.compile(r"^CLAIM\s+(\S+):", re.IGNORECASE)
    lines = bb.read_text().splitlines()

    # Collect (agent, exp_id) pairs from last 15 CLAIM lines
    claims = []
    for line in lines:
        m = claim_pattern.match(line.strip())
        if m:
            agent = m.group(1)
            exp_m = re.search(r"exp\d+", line, re.IGNORECASE)
            exp_id = exp_m.group(0).lower() if exp_m else None
            claims.append((agent, exp_id))
    recent_claims = claims[-15:]
    if len(recent_claims) < 4:
        return None

    # Map exp_id → design from results
    exp_design = {e.exp_id: e.design for e in report.experiments if e.design}

    # Tally designs per agent from recent claims
    agent_designs: dict[str, set] = {}
    for agent, exp_id in recent_claims:
        design = exp_design.get(exp_id, "") if exp_id else ""
        if not design:
            continue
        agent_designs.setdefault(agent, set()).add(design)

    if len(agent_designs) < 2:
        return None

    # Find design claimed by 2+ agents with no cross-design claims
    from collections import Counter
    all_designs = [d for ds in agent_designs.values() for d in ds]
    counts = Counter(all_designs)
    for design, count in counts.most_common():
        if count >= 2:
            # Check that agents claiming this design aren't also exploring others
            exclusive = sum(1 for ds in agent_designs.values()
                            if design in ds and len(ds) == 1)
            if exclusive >= 2:
                return design
    return None


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

    # --- v4.9.4: death spiral + false consensus ---
    death_spiral = detect_death_spiral(report)
    false_consensus_axis = detect_false_consensus(domain_dir, report)

    # --- Decision matrix ---

    # Low PQ after enough experiments = hacking
    if pq < 10 and total > 15:
        return "STOP_HACKING"

    # Death spiral: crash rate accelerating or rolling mean worsening
    if death_spiral and pq >= 10:
        return "NUDGE"

    # False consensus: 2+ agents locked on same axis without cross-validation
    if false_consensus_axis and pq >= 10:
        return "NUDGE"

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
    _ds = detect_death_spiral(report)
    _fc = detect_false_consensus(domain_dir, report)
    print(f"[diagnose.py] Death spiral: {_ds} | False consensus: {_fc}", file=sys.stderr)

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
        "death_spiral": _ds,
        "false_consensus_axis": _fc,
    }
    nudge_path = domain_dir / ".nudge_data.json"
    nudge_path.write_text(_json.dumps(nudge_data, indent=2))
    print(f"[diagnose.py] Wrote nudge data to {nudge_path}", file=sys.stderr)

    # Output decision to stdout (what outer-loop.sh reads)
    print(decision)


if __name__ == "__main__":
    main()

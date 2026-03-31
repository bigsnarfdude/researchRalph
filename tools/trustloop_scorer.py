#!/usr/bin/env python3
"""
TrustLoop Experiment Scorer — automated classification and scoring of RRMA outputs.

Designed for multi-agent scale (2-8 agents). Reads results.tsv + agent JSONL logs
and produces structured verdicts without requiring Claude API calls.

Scoring dimensions:
  1. Outcome class: BREAKTHROUGH / INCREMENTAL / PLATEAU / REGRESSION / CRASH
  2. Design novelty: 0.0-1.0 (vs prior experiments)
  3. Agent efficiency: success_rate, waste_ratio, best_contribution
  4. Redundancy detection: near-duplicate configs across agents
  5. Influence score: did blackboard reads precede improvements

Usage:
    # Score a domain's results
    python3 tools/trustloop_scorer.py domains/gpt2-tinystories-v44

    # JSON output for programmatic use
    python3 tools/trustloop_scorer.py domains/gpt2-tinystories-v44 --json

    # Score with agent trace analysis (slower, reads JSONL logs)
    python3 tools/trustloop_scorer.py domains/gpt2-tinystories-v44 --traces
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════

OutcomeClass = Literal[
    "BREAKTHROUGH",  # New best score (keep)
    "INCREMENTAL",   # Improved but not best
    "PLATEAU",       # Within 1% of current best, not new best
    "REGRESSION",    # Worse than median of last 5
    "CRASH",         # Experiment failed
]


@dataclass
class ScoredExperiment:
    exp_id: str
    score: float | None
    status: str  # keep/discard/crash
    description: str
    agent: str
    design: str
    train_min: float
    # Computed fields
    outcome_class: OutcomeClass = "PLATEAU"
    is_best_ever: bool = False
    improvement_pct: float = 0.0       # vs previous best (negative = better for lower-is-better)
    novelty: float = 0.0               # 0-1: how different from prior designs
    redundant_with: str | None = None  # exp_id of near-duplicate


@dataclass
class AgentReport:
    agent_id: str
    total_experiments: int = 0
    keeps: int = 0
    crashes: int = 0
    breakthroughs: int = 0
    best_score: float | None = None
    best_exp: str | None = None
    success_rate: float = 0.0      # keeps / total
    waste_ratio: float = 0.0       # (crashes + redundant) / total
    redundant_count: int = 0
    # Trace-derived (optional)
    tool_calls: int = 0
    thinking_blocks: int = 0
    blackboard_reads: int = 0
    blackboard_writes: int = 0


@dataclass
class Anomaly:
    """A detected anomaly in the experiment run."""
    severity: Literal["INFO", "WARN", "ALERT"]
    category: str   # score_jump, protocol_violation, resource_waste, suspicious, stagnation
    exp_id: str | None
    agent: str | None
    message: str


@dataclass
class WorkflowCheck:
    """Result of a workflow validation check."""
    check: str
    passed: bool
    detail: str


@dataclass
class Insight:
    """Actionable insight from analyzing the run."""
    kind: str  # winning_strategy, dead_end, agent_drift, resource_suggestion, desire, mistake, learning
    message: str
    source: str | None = None  # which file/agent produced this


@dataclass
class AgentTelemetry:
    """Parsed content from DESIRES.md, MISTAKES.md, LEARNINGS.md."""
    desires: list[str] = field(default_factory=list)       # what agents want/need
    mistakes: list[dict] = field(default_factory=list)     # {exp, what, result, lesson}
    learnings: list[str] = field(default_factory=list)     # discovered facts


@dataclass
class DomainReport:
    domain: str
    total_experiments: int = 0
    best_score: float | None = None
    best_exp: str | None = None
    score_direction: str = "lower"  # lower or higher
    breakthroughs: int = 0
    crashes: int = 0
    redundant_pairs: int = 0
    active_agents: int = 0
    experiments: list[ScoredExperiment] = field(default_factory=list)
    agents: list[AgentReport] = field(default_factory=list)
    stagnation_depth: int = 0      # experiments since last breakthrough
    efficiency_score: float = 0.0  # 0-1 composite
    anomalies: list[Anomaly] = field(default_factory=list)
    workflow_checks: list[WorkflowCheck] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


# ═══════════════════════════════════════════════════════════════════════════════
# Score direction detection
# ═══════════════════════════════════════════════════════════════════════════════

# Metrics where lower is better
LOWER_IS_BETTER = {"bpb", "loss", "perplexity", "error", "mse", "mae", "ce"}
# Metrics where higher is better
HIGHER_IS_BETTER = {"f1", "accuracy", "auroc", "auc", "precision", "recall", "score"}


def detect_score_direction(domain_name: str, descriptions: list[str]) -> str:
    """Guess whether lower or higher scores are better."""
    name_lower = domain_name.lower()
    for kw in LOWER_IS_BETTER:
        if kw in name_lower:
            return "lower"
    for kw in HIGHER_IS_BETTER:
        if kw in name_lower:
            return "higher"
    # Check descriptions for hints
    desc_text = " ".join(descriptions).lower()
    if any(kw in desc_text for kw in ["bpb", "loss", "perplexity"]):
        return "lower"
    if any(kw in desc_text for kw in ["f1", "accuracy", "auroc"]):
        return "higher"
    # Default: lower is better (most training metrics)
    return "lower"


# ═══════════════════════════════════════════════════════════════════════════════
# Results parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_results_tsv(path: Path) -> list[dict]:
    """Parse results.tsv with flexible header detection."""
    rows = []
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return rows

    lines = text.split("\n")
    # Detect header
    first = lines[0].split("\t")
    if any(h in first for h in ["exp_id", "EXP-ID", "score"]):
        header = [h.strip().lower().replace("-", "_") for h in first]
        data_lines = lines[1:]
    else:
        header = ["exp_id", "score", "vram", "status", "description", "agent", "design", "train_min"]
        data_lines = lines

    for line in data_lines:
        if not line.strip():
            continue
        fields = line.split("\t")
        row = {}
        for i, h in enumerate(header):
            row[h] = fields[i].strip() if i < len(fields) else ""
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Novelty scoring (description + design similarity)
# ═══════════════════════════════════════════════════════════════════════════════

def description_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two experiment descriptions."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_novelty(desc: str, design: str, prior_descs: list[str], prior_designs: list[str]) -> float:
    """Novelty score 0-1. 1.0 = completely novel, 0.0 = exact duplicate."""
    if not prior_descs:
        return 1.0

    # Max similarity to any prior experiment
    max_desc_sim = max(description_similarity(desc, p) for p in prior_descs)
    max_design_sim = max(
        (1.0 if design == p else 0.0) for p in prior_designs
    ) if prior_designs else 0.0

    # Weighted: description similarity matters more than design label
    combined_sim = 0.7 * max_desc_sim + 0.3 * max_design_sim
    return max(0.0, 1.0 - combined_sim)


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_experiments(
    rows: list[dict], direction: str
) -> list[ScoredExperiment]:
    """Classify each experiment's outcome."""
    lower = direction == "lower"
    experiments = []
    best_so_far = None
    prior_descs = []
    prior_designs = []
    recent_scores = []  # last 5 valid scores for regression detection

    for row in rows:
        exp_id = row.get("exp_id", "?")
        status = row.get("status", "discard")
        desc = row.get("description", "")
        agent = row.get("agent", "unknown")
        design = row.get("design", "")
        try:
            score = float(row.get("score", "nan"))
        except (ValueError, TypeError):
            score = None
        try:
            train_min = float(row.get("train_min", "0"))
        except (ValueError, TypeError):
            train_min = 0.0

        exp = ScoredExperiment(
            exp_id=exp_id, score=score, status=status,
            description=desc, agent=agent, design=design,
            train_min=train_min,
        )

        # Classify
        if status == "crash" or score is None:
            exp.outcome_class = "CRASH"
        elif best_so_far is None:
            exp.outcome_class = "BREAKTHROUGH"
            exp.is_best_ever = True
            best_so_far = score
        else:
            is_better = (score < best_so_far) if lower else (score > best_so_far)
            pct_diff = abs(score - best_so_far) / abs(best_so_far) * 100 if best_so_far != 0 else 0

            if is_better:
                exp.outcome_class = "BREAKTHROUGH"
                exp.is_best_ever = True
                exp.improvement_pct = pct_diff
                best_so_far = score
            elif pct_diff < 1.0:
                exp.outcome_class = "PLATEAU"
            elif recent_scores:
                median_recent = sorted(recent_scores)[len(recent_scores) // 2]
                is_worse_than_median = (score > median_recent) if lower else (score < median_recent)
                if is_worse_than_median:
                    exp.outcome_class = "REGRESSION"
                else:
                    exp.outcome_class = "INCREMENTAL"
            else:
                exp.outcome_class = "INCREMENTAL"

        # Novelty
        exp.novelty = compute_novelty(desc, design, prior_descs, prior_designs)

        # Redundancy detection (>85% description similarity to any prior)
        for i, prior_desc in enumerate(prior_descs):
            if description_similarity(desc, prior_desc) > 0.85:
                exp.redundant_with = experiments[i].exp_id
                break

        prior_descs.append(desc)
        prior_designs.append(design)
        if score is not None:
            recent_scores.append(score)
            if len(recent_scores) > 5:
                recent_scores.pop(0)

        experiments.append(exp)

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
# Agent reports
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_reports(
    experiments: list[ScoredExperiment], direction: str
) -> list[AgentReport]:
    """Aggregate per-agent statistics."""
    lower = direction == "lower"
    agents: dict[str, AgentReport] = {}

    for exp in experiments:
        aid = exp.agent
        if aid not in agents:
            agents[aid] = AgentReport(agent_id=aid)
        r = agents[aid]
        r.total_experiments += 1

        if exp.status == "crash" or exp.outcome_class == "CRASH":
            r.crashes += 1
        elif exp.status == "keep":
            r.keeps += 1

        if exp.outcome_class == "BREAKTHROUGH":
            r.breakthroughs += 1

        if exp.redundant_with:
            r.redundant_count += 1

        if exp.score is not None:
            if r.best_score is None:
                r.best_score = exp.score
                r.best_exp = exp.exp_id
            elif (exp.score < r.best_score) if lower else (exp.score > r.best_score):
                r.best_score = exp.score
                r.best_exp = exp.exp_id

    for r in agents.values():
        if r.total_experiments > 0:
            r.success_rate = r.keeps / r.total_experiments
            r.waste_ratio = (r.crashes + r.redundant_count) / r.total_experiments

    return sorted(agents.values(), key=lambda a: a.agent_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Trace analysis (optional, reads agent JSONL logs)
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_from_traces(agents: list[AgentReport], logs_dir: Path) -> None:
    """Read agent JSONL logs and count tool calls, thinking, artifact access."""
    for report in agents:
        # Find matching log files
        pattern = f"{report.agent_id}*.jsonl"
        # Current session log (e.g. agent0.jsonl)
        current_log = logs_dir / f"{report.agent_id}.jsonl"
        if not current_log.exists():
            # Try alternate names
            matches = list(logs_dir.glob(f"{report.agent_id}*.jsonl"))
            if matches:
                current_log = max(matches, key=lambda p: p.stat().st_mtime)
            else:
                continue

        try:
            with open(current_log) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    # Count tool calls
                    if etype == "assistant":
                        content = event.get("message", {}).get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get("type") == "tool_use":
                                        report.tool_calls += 1
                                        tool_name = block.get("name", "")
                                        inp = block.get("input", {})
                                        # Detect artifact reads/writes
                                        if tool_name == "Read":
                                            fp = inp.get("file_path", "")
                                            if "blackboard" in fp:
                                                report.blackboard_reads += 1
                                        elif tool_name in ("Edit", "Write"):
                                            fp = inp.get("file_path", "")
                                            if "blackboard" in fp:
                                                report.blackboard_writes += 1
                                    elif block.get("type") == "thinking":
                                        report.thinking_blocks += 1
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry parsing (DESIRES, MISTAKES, LEARNINGS)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_telemetry(domain_dir: Path) -> AgentTelemetry:
    """Parse agent self-telemetry files."""
    tel = AgentTelemetry()

    # DESIRES.md — bullet list of wants/needs
    desires_path = domain_dir / "DESIRES.md"
    if desires_path.exists():
        for line in desires_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- ") and len(line) > 5:
                tel.desires.append(line[2:].strip())

    # MISTAKES.md — structured sections per experiment
    mistakes_path = domain_dir / "MISTAKES.md"
    if mistakes_path.exists():
        current: dict | None = None
        for line in mistakes_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("## "):
                if current:
                    tel.mistakes.append(current)
                # Parse "## EXP-003: matrix_lr=0.06 (agent1)"
                m = re.match(r"##\s+(EXP-\d+|exp\d+):?\s*(.*)", line, re.IGNORECASE)
                current = {
                    "exp": m.group(1) if m else line[3:],
                    "what": m.group(2).strip() if m else "",
                    "result": "",
                    "lesson": "",
                }
            elif current:
                if line.startswith("- **What**:"):
                    current["what"] = line.split(":", 1)[1].strip().strip("*")
                elif line.startswith("- **Result**:"):
                    current["result"] = line.split(":", 1)[1].strip().strip("*")
                elif line.startswith("- **Lesson**:"):
                    current["lesson"] = line.split(":", 1)[1].strip().strip("*")
                elif line.startswith("- **Why it failed**:"):
                    current["why"] = line.split(":", 1)[1].strip().strip("*")
        if current:
            tel.mistakes.append(current)

    # LEARNINGS.md — bullet points and section content
    learnings_path = domain_dir / "LEARNINGS.md"
    if learnings_path.exists():
        for line in learnings_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- ") and len(line) > 10:
                tel.learnings.append(line[2:].strip())

    return tel


# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(
    experiments: list[ScoredExperiment],
    agents: list[AgentReport],
    direction: str,
) -> list[Anomaly]:
    """Detect anomalous patterns in experiment results and agent behavior."""
    anomalies = []
    lower = direction == "lower"
    valid = [e for e in experiments if e.score is not None]

    if len(valid) < 2:
        return anomalies

    scores = [e.score for e in valid]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5 if variance > 0 else 0.001

    # 1. Score jumps: >3 sigma improvement in a single step
    prev_best = None
    for e in valid:
        if prev_best is not None:
            diff = prev_best - e.score if lower else e.score - prev_best
            if diff > 3 * std_score and diff > 0:
                anomalies.append(Anomaly(
                    severity="WARN", category="score_jump",
                    exp_id=e.exp_id, agent=e.agent,
                    message=f"Score jumped {diff:.4f} (>{3*std_score:.4f} = 3σ) — verify not cherry-picked or data leak",
                ))
        if e.is_best_ever:
            prev_best = e.score

    # 2. Agent monopoly: one agent does >80% of experiments
    total = len(experiments)
    for a in agents:
        if total >= 5 and a.total_experiments / total > 0.8:
            anomalies.append(Anomaly(
                severity="WARN", category="agent_monopoly",
                exp_id=None, agent=a.agent_id,
                message=f"Agent runs {a.total_experiments}/{total} experiments ({a.total_experiments/total:.0%}) — other agents may be stuck",
            ))

    # 3. Crash streaks: 3+ consecutive crashes from same agent
    agent_streaks: dict[str, int] = defaultdict(int)
    for e in experiments:
        if e.outcome_class == "CRASH":
            agent_streaks[e.agent] += 1
            if agent_streaks[e.agent] >= 3:
                anomalies.append(Anomaly(
                    severity="ALERT", category="crash_streak",
                    exp_id=e.exp_id, agent=e.agent,
                    message=f"Agent has {agent_streaks[e.agent]} consecutive crashes — likely broken config or OOM",
                ))
        else:
            agent_streaks[e.agent] = 0

    # 4. Redundancy burst: >50% of last 10 experiments are near-duplicates
    last_10 = experiments[-10:] if len(experiments) >= 10 else experiments
    dup_count = sum(1 for e in last_10 if e.redundant_with is not None)
    if len(last_10) >= 5 and dup_count / len(last_10) > 0.5:
        anomalies.append(Anomaly(
            severity="WARN", category="redundancy_burst",
            exp_id=None, agent=None,
            message=f"{dup_count}/{len(last_10)} recent experiments are near-duplicates — agents not reading blackboard or repeating work",
        ))

    # 5. Score regression after breakthrough: if score gets worse right after a new best
    for i in range(1, len(valid)):
        if valid[i-1].is_best_ever and valid[i].outcome_class == "REGRESSION":
            anomalies.append(Anomaly(
                severity="INFO", category="post_breakthrough_regression",
                exp_id=valid[i].exp_id, agent=valid[i].agent,
                message=f"Regression immediately after breakthrough {valid[i-1].exp_id} — agent may not have read updated best/",
            ))

    # 6. Stagnation: no improvement in last 15+ experiments
    stag = 0
    for e in reversed(valid):
        if e.is_best_ever:
            break
        stag += 1
    if stag >= 15:
        anomalies.append(Anomaly(
            severity="ALERT", category="deep_stagnation",
            exp_id=None, agent=None,
            message=f"No improvement in {stag} experiments — search space may be exhausted or agents are stuck",
        ))

    # 7. Zero-novelty agent: agent running all low-novelty experiments
    for a in agents:
        agent_exps = [e for e in experiments if e.agent == a.agent_id]
        if len(agent_exps) >= 3:
            avg_nov = sum(e.novelty for e in agent_exps) / len(agent_exps)
            if avg_nov < 0.2:
                anomalies.append(Anomaly(
                    severity="WARN", category="low_novelty_agent",
                    exp_id=None, agent=a.agent_id,
                    message=f"Average novelty {avg_nov:.2f} — agent is tweaking knobs instead of exploring",
                ))

    return anomalies


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_workflow(
    domain_dir: Path,
    agents: list[AgentReport],
    experiments: list[ScoredExperiment],
) -> list[WorkflowCheck]:
    """Validate that the RRMA workflow is being followed correctly."""
    checks = []

    # 1. Required files exist
    for f in ["program.md", "blackboard.md", "results.tsv", "run.sh"]:
        path = domain_dir / f
        checks.append(WorkflowCheck(
            check=f"file_exists:{f}",
            passed=path.exists(),
            detail=f"{f} {'found' if path.exists() else 'MISSING'}",
        ))

    # 2. Results.tsv has header
    results_path = domain_dir / "results.tsv"
    if results_path.exists():
        with open(results_path) as f:
            first_line = f.readline().strip()
        has_header = "score" in first_line.lower() or "exp_id" in first_line.lower()
        checks.append(WorkflowCheck(
            check="results_header",
            passed=has_header,
            detail=f"Header {'valid' if has_header else 'missing or malformed'}",
        ))

    # 3. Blackboard is being written to
    bb = domain_dir / "blackboard.md"
    if bb.exists():
        bb_lines = bb.read_text().count("\n")
        active = bb_lines > 5
        checks.append(WorkflowCheck(
            check="blackboard_active",
            passed=active,
            detail=f"Blackboard has {bb_lines} lines {'(active)' if active else '(nearly empty — agents may not be writing)'}",
        ))

    # 4. Agents read blackboard (from trace data)
    for a in agents:
        if a.tool_calls > 0:  # trace data available
            reads_bb = a.blackboard_reads > 0
            checks.append(WorkflowCheck(
                check=f"agent_reads_blackboard:{a.agent_id}",
                passed=reads_bb,
                detail=f"{a.agent_id}: {'reads' if reads_bb else 'NEVER reads'} blackboard ({a.blackboard_reads} reads)",
            ))

    # 5. Agents write blackboard
    for a in agents:
        if a.tool_calls > 0:
            writes_bb = a.blackboard_writes > 0
            checks.append(WorkflowCheck(
                check=f"agent_writes_blackboard:{a.agent_id}",
                passed=writes_bb,
                detail=f"{a.agent_id}: {'writes' if writes_bb else 'NEVER writes'} blackboard ({a.blackboard_writes} writes)",
            ))

    # 6. Agent telemetry files (DESIRES, MISTAKES, LEARNINGS)
    for f in ["DESIRES.md", "MISTAKES.md", "LEARNINGS.md"]:
        path = domain_dir / f
        exists = path.exists() and path.stat().st_size > 10
        checks.append(WorkflowCheck(
            check=f"telemetry:{f}",
            passed=exists,
            detail=f"{f} {'present' if exists else 'missing — agents not writing self-telemetry'}",
        ))

    # 7. Score consistency: all keep experiments actually improve
    wrong_keeps = []
    best = None
    lower = True  # default
    for e in experiments:
        if e.score is None:
            continue
        if e.status == "keep":
            if best is not None:
                is_better = (e.score < best) if lower else (e.score > best)
                if not is_better:
                    wrong_keeps.append(e.exp_id)
            best = e.score if best is None else (min(best, e.score) if lower else max(best, e.score))
    checks.append(WorkflowCheck(
        check="keep_consistency",
        passed=len(wrong_keeps) == 0,
        detail=f"{'All keeps improve' if not wrong_keeps else f'Wrong keeps: {wrong_keeps} — run.sh may have race condition'}",
    ))

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# Insight generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    experiments: list[ScoredExperiment],
    agents: list[AgentReport],
    anomalies: list[Anomaly],
    direction: str,
    telemetry: AgentTelemetry | None = None,
) -> list[Insight]:
    """Generate actionable insights from the data."""
    insights = []
    lower = direction == "lower"

    # 1. Winning strategies: what designs produce breakthroughs
    bt_designs = defaultdict(list)
    for e in experiments:
        if e.outcome_class == "BREAKTHROUGH":
            bt_designs[e.design].append(e)

    if bt_designs:
        best_design = max(bt_designs.items(), key=lambda x: len(x[1]))
        if len(best_design[1]) > 1:
            insights.append(Insight(
                kind="winning_strategy",
                message=f"Design '{best_design[0]}' produced {len(best_design[1])} breakthroughs — double down here",
            ))

    # 2. Dead ends: designs that never produce keeps
    design_keeps = defaultdict(int)
    design_total = defaultdict(int)
    for e in experiments:
        design_total[e.design] += 1
        if e.status == "keep":
            design_keeps[e.design] += 1
    for d, total in design_total.items():
        if total >= 3 and design_keeps[d] == 0:
            insights.append(Insight(
                kind="dead_end",
                message=f"Design '{d}' has {total} experiments, 0 keeps — abandon this approach",
            ))

    # 3. Best agent
    if len(agents) > 1:
        producing = [a for a in agents if a.breakthroughs > 0]
        if producing:
            top = max(producing, key=lambda a: a.breakthroughs)
            insights.append(Insight(
                kind="top_agent",
                message=f"{top.agent_id} leads with {top.breakthroughs} breakthroughs (best: {top.best_score})",
            ))

    # 4. Resource waste
    total_time = sum(e.train_min for e in experiments)
    crash_time = sum(e.train_min for e in experiments if e.outcome_class == "CRASH")
    if total_time > 0 and crash_time / total_time > 0.2:
        insights.append(Insight(
            kind="resource_waste",
            message=f"{crash_time:.0f}/{total_time:.0f} min spent on crashes ({crash_time/total_time:.0%}) — fix common failure mode first",
        ))

    # 5. Agent drift: agent's recent experiments much worse than their best
    for a in agents:
        agent_exps = [e for e in experiments if e.agent == a.agent_id and e.score is not None]
        if len(agent_exps) >= 5 and a.best_score is not None:
            recent = agent_exps[-3:]
            recent_avg = sum(e.score for e in recent) / len(recent)
            drift = abs(recent_avg - a.best_score) / abs(a.best_score) * 100 if a.best_score != 0 else 0
            if drift > 10:
                insights.append(Insight(
                    kind="agent_drift",
                    message=f"{a.agent_id} drifting: recent avg {recent_avg:.4f} vs best {a.best_score:.4f} ({drift:.0f}% off)",
                ))

    # 6. Collaboration signal: does blackboard sharing correlate with breakthroughs
    readers = [a for a in agents if a.blackboard_reads > 0 and a.breakthroughs > 0]
    non_readers = [a for a in agents if a.blackboard_reads == 0 and a.total_experiments > 0]
    if readers and non_readers:
        insights.append(Insight(
            kind="collaboration",
            message=f"Agents reading blackboard ({', '.join(a.agent_id for a in readers)}) produce breakthroughs; "
                    f"non-readers ({', '.join(a.agent_id for a in non_readers)}) do not",
        ))

    # ── Telemetry-driven insights ──────────────────────────────────────────

    if telemetry:
        # 7. Agent desires — surface what agents are asking for
        for desire in telemetry.desires:
            insights.append(Insight(
                kind="desire",
                message=desire,
                source="DESIRES.md",
            ))

        # 8. Key mistakes — extract lessons learned
        for mistake in telemetry.mistakes:
            lesson = mistake.get("lesson", "") or mistake.get("why", "")
            if lesson:
                insights.append(Insight(
                    kind="mistake_lesson",
                    message=f"{mistake.get('exp', '?')}: {lesson}",
                    source="MISTAKES.md",
                ))

        # 9. Key learnings — surface environment discoveries
        #    Highlight learnings that contain strong signals
        strong_signals = ["IMPORTANT", "massive win", "dominat", "never ", "always ", "confirmed"]
        for learning in telemetry.learnings:
            is_strong = any(sig.lower() in learning.lower() for sig in strong_signals)
            if is_strong:
                insights.append(Insight(
                    kind="key_learning",
                    message=learning,
                    source="LEARNINGS.md",
                ))

        # 10. Desire-experiment gap: are desires being addressed by experiments?
        if telemetry.desires and experiments:
            desire_keywords = set()
            for d in telemetry.desires:
                desire_keywords.update(w.lower() for w in d.split() if len(w) > 4)
            exp_keywords = set()
            for e in experiments:
                exp_keywords.update(w.lower() for w in e.description.split() if len(w) > 4)
            unaddressed = desire_keywords - exp_keywords
            if len(unaddressed) > len(desire_keywords) * 0.7 and telemetry.desires:
                insights.append(Insight(
                    kind="unaddressed_desires",
                    message=f"{len(telemetry.desires)} desires filed but mostly unaddressed — gardener should read DESIRES.md",
                    source="gap_analysis",
                ))

        # 11. Mistake pattern: same failure mode repeated
        if len(telemetry.mistakes) >= 2:
            lessons = [m.get("lesson", "").lower() for m in telemetry.mistakes if m.get("lesson")]
            # Check for repeated themes
            theme_counts: dict[str, int] = defaultdict(int)
            for lesson in lessons:
                for theme in ["throughput", "step count", "transfer", "vram", "batch", "learning rate"]:
                    if theme in lesson:
                        theme_counts[theme] += 1
            for theme, count in theme_counts.items():
                if count >= 2:
                    insights.append(Insight(
                        kind="recurring_mistake",
                        message=f"'{theme}' appears in {count} mistake lessons — agents keep hitting the same wall",
                        source="MISTAKES.md",
                    ))

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# Domain report
# ═══════════════════════════════════════════════════════════════════════════════

def score_domain(domain_dir: Path, with_traces: bool = False) -> DomainReport:
    """Build complete domain scoring report."""
    results_path = domain_dir / "results.tsv"
    if not results_path.exists():
        return DomainReport(domain=domain_dir.name)

    rows = parse_results_tsv(results_path)
    if not rows:
        return DomainReport(domain=domain_dir.name)

    # Detect score direction
    descriptions = [r.get("description", "") for r in rows]
    direction = detect_score_direction(domain_dir.name, descriptions)

    # Classify experiments
    experiments = classify_experiments(rows, direction)

    # Agent reports
    agent_reports = build_agent_reports(experiments, direction)

    # Trace enrichment
    if with_traces:
        logs_dir = domain_dir / "logs"
        if logs_dir.exists():
            enrich_from_traces(agent_reports, logs_dir)

    # Summary stats
    lower = direction == "lower"
    valid_scores = [e.score for e in experiments if e.score is not None]
    best_score = None
    best_exp = None
    if valid_scores:
        best_idx = valid_scores.index(min(valid_scores) if lower else max(valid_scores))
        valid_exps = [e for e in experiments if e.score is not None]
        best_score = valid_exps[best_idx].score
        best_exp = valid_exps[best_idx].exp_id

    breakthroughs = sum(1 for e in experiments if e.outcome_class == "BREAKTHROUGH")
    crashes = sum(1 for e in experiments if e.outcome_class == "CRASH")
    redundant = sum(1 for e in experiments if e.redundant_with is not None)

    # Stagnation: experiments since last breakthrough
    stagnation = 0
    for e in reversed(experiments):
        if e.outcome_class == "BREAKTHROUGH":
            break
        stagnation += 1

    # Efficiency composite: breakthroughs/total * (1 - waste) * novelty_avg
    total = len(experiments)
    if total > 0:
        bt_rate = breakthroughs / total
        waste = (crashes + redundant) / total
        avg_novelty = sum(e.novelty for e in experiments) / total
        efficiency = bt_rate * (1 - waste) * avg_novelty
    else:
        efficiency = 0.0

    # Telemetry
    telemetry = parse_telemetry(domain_dir)

    # Anomaly detection
    anomalies = detect_anomalies(experiments, agent_reports, direction)

    # Workflow validation
    workflow_checks = validate_workflow(domain_dir, agent_reports, experiments)

    # Insight generation
    insights = generate_insights(experiments, agent_reports, anomalies, direction, telemetry)

    return DomainReport(
        domain=domain_dir.name,
        total_experiments=total,
        best_score=best_score,
        best_exp=best_exp,
        score_direction=direction,
        breakthroughs=breakthroughs,
        crashes=crashes,
        redundant_pairs=redundant,
        active_agents=len(agent_reports),
        experiments=experiments,
        agents=agent_reports,
        stagnation_depth=stagnation,
        efficiency_score=efficiency,
        anomalies=anomalies,
        workflow_checks=workflow_checks,
        insights=insights,
        telemetry=telemetry,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════

OUTCOME_SYMBOLS = {
    "BREAKTHROUGH": "★",
    "INCREMENTAL": "↑",
    "PLATEAU": "→",
    "REGRESSION": "↓",
    "CRASH": "✗",
}


def format_report(report: DomainReport) -> str:
    """Human-readable report."""
    lines = []
    lines.append(f"# TrustLoop Scorer — {report.domain}")
    lines.append(f"")
    lines.append(f"**Experiments:** {report.total_experiments}  |  "
                 f"**Best:** {report.best_score} ({report.best_exp})  |  "
                 f"**Direction:** {report.score_direction}_is_better")
    lines.append(f"**Breakthroughs:** {report.breakthroughs}  |  "
                 f"**Crashes:** {report.crashes}  |  "
                 f"**Redundant:** {report.redundant_pairs}  |  "
                 f"**Stagnation:** {report.stagnation_depth}")
    lines.append(f"**Efficiency:** {report.efficiency_score:.3f}")
    lines.append("")

    # Experiment table
    lines.append("## Experiments")
    lines.append("")
    lines.append(f"{'ID':<8} {'Score':<12} {'Class':<14} {'Nov':<5} {'Agent':<8} {'Description'}")
    lines.append(f"{'─'*8} {'─'*12} {'─'*14} {'─'*5} {'─'*8} {'─'*40}")
    for e in report.experiments:
        sym = OUTCOME_SYMBOLS.get(e.outcome_class, "?")
        score_str = f"{e.score:.6f}" if e.score is not None else "crash"
        dup = f" [dup:{e.redundant_with}]" if e.redundant_with else ""
        lines.append(f"{e.exp_id:<8} {score_str:<12} {sym} {e.outcome_class:<12} "
                     f"{e.novelty:.2f}  {e.agent:<8} {e.description[:50]}{dup}")

    # Agent table
    lines.append("")
    lines.append("## Agents")
    lines.append("")
    lines.append(f"{'Agent':<10} {'Exps':<6} {'Keeps':<6} {'BT':<4} "
                 f"{'Crash':<6} {'Dup':<5} {'Rate':<6} {'Waste':<6} {'Best'}")
    lines.append(f"{'─'*10} {'─'*6} {'─'*6} {'─'*4} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*12}")
    for a in report.agents:
        best_str = f"{a.best_score:.6f}" if a.best_score is not None else "—"
        lines.append(f"{a.agent_id:<10} {a.total_experiments:<6} {a.keeps:<6} "
                     f"{a.breakthroughs:<4} {a.crashes:<6} {a.redundant_count:<5} "
                     f"{a.success_rate:.2f}  {a.waste_ratio:.2f}  {best_str}")

    # Trace stats (if available)
    has_traces = any(a.tool_calls > 0 for a in report.agents)
    if has_traces:
        lines.append("")
        lines.append("## Trace Analysis")
        lines.append("")
        lines.append(f"{'Agent':<10} {'Tools':<8} {'Think':<8} {'BB Read':<9} {'BB Write'}")
        lines.append(f"{'─'*10} {'─'*8} {'─'*8} {'─'*9} {'─'*9}")
        for a in report.agents:
            lines.append(f"{a.agent_id:<10} {a.tool_calls:<8} {a.thinking_blocks:<8} "
                         f"{a.blackboard_reads:<9} {a.blackboard_writes}")

    # Telemetry summary
    tel = report.telemetry
    if tel.desires or tel.mistakes or tel.learnings:
        lines.append("")
        lines.append("## Agent Telemetry")
        lines.append("")
        lines.append(f"**Desires:** {len(tel.desires)}  |  "
                     f"**Mistakes logged:** {len(tel.mistakes)}  |  "
                     f"**Learnings:** {len(tel.learnings)}")
        if tel.desires:
            lines.append("")
            lines.append("### What agents want:")
            for d in tel.desires:
                lines.append(f"  - {d}")
        if tel.mistakes:
            lines.append("")
            lines.append("### Logged failures:")
            for m in tel.mistakes:
                lesson = m.get("lesson", "") or m.get("why", "")
                lines.append(f"  - {m.get('exp', '?')}: {m.get('what', '?')}"
                             + (f" => {lesson}" if lesson else ""))

    # Anomalies
    if report.anomalies:
        lines.append("")
        lines.append("## Anomalies")
        lines.append("")
        for a in report.anomalies:
            icon = {"INFO": ".", "WARN": "!", "ALERT": "!!!"}[a.severity]
            target = f"[{a.exp_id or a.agent or 'global'}]"
            lines.append(f"  {icon} {a.severity} {a.category} {target}: {a.message}")

    # Workflow checks
    failed = [c for c in report.workflow_checks if not c.passed]
    if failed:
        lines.append("")
        lines.append("## Workflow Issues")
        lines.append("")
        for c in failed:
            lines.append(f"  FAIL {c.check}: {c.detail}")

    passed_count = sum(1 for c in report.workflow_checks if c.passed)
    total_checks = len(report.workflow_checks)
    if total_checks > 0:
        lines.append(f"")
        lines.append(f"Workflow: {passed_count}/{total_checks} checks passed")

    # Insights
    if report.insights:
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        for ins in report.insights:
            lines.append(f"  [{ins.kind}] {ins.message}")

    return "\n".join(lines)


def to_json(report: DomainReport) -> str:
    """JSON output for programmatic consumption."""
    d = asdict(report)
    return json.dumps(d, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TrustLoop Experiment Scorer")
    parser.add_argument("domain", type=Path, help="Path to domain directory")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--traces", action="store_true", help="Include trace analysis (reads JSONL logs)")
    args = parser.parse_args()

    report = score_domain(args.domain, with_traces=args.traces)

    if args.json:
        print(to_json(report))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()

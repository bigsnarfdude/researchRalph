#!/usr/bin/env python3
"""
TrustLoop Stoplight Report — 30 red/yellow/green telemetry signals.

Scan for red, investigate yellow, ignore green.

Usage:
    python3 tools/trustloop_stoplight.py domains/gpt2-tinystories-v44
    python3 tools/trustloop_stoplight.py domains/gpt2-tinystories-v44 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

Color = Literal["GREEN", "YELLOW", "RED"]
ANSI = {"GREEN": "\033[32m", "YELLOW": "\033[33m", "RED": "\033[31m", "RESET": "\033[0m"}


@dataclass
class Light:
    name: str
    color: Color
    summary: str
    value: str = ""
    section: str = ""


def L(name: str, color: Color, summary: str, value: str = "") -> Light:
    return Light(name=name, color=color, summary=summary, value=value)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    rows = []
    text = path.read_text().strip()
    if not text:
        return rows
    lines = text.split("\n")
    first = lines[0].split("\t")
    if any(h in first for h in ["exp_id", "score"]):
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


def load_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def load_lines(path: Path) -> list[str]:
    return load_text(path).splitlines() if path.exists() else []


def scores(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        try:
            out.append(float(r["score"]))
        except (ValueError, TypeError, KeyError):
            pass
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def breakthrough_indices(rows: list[dict]) -> list[int]:
    """Indices of rows that set a new best (lower-is-better)."""
    best = None
    idxs = []
    for i, r in enumerate(rows):
        try:
            s = float(r["score"])
        except (ValueError, TypeError, KeyError):
            continue
        if best is None or s < best:
            best = s
            idxs.append(i)
    return idxs


# ═══════════════════════════════════════════════════════════════════════════════
# 30 checks
# ═══════════════════════════════════════════════════════════════════════════════

# ── RUN HEALTH (1-6) ─────────────────────────────────────────────────────────

def ck_workflow_files(domain: Path, **_) -> Light:
    required = ["program.md", "blackboard.md", "results.tsv", "run.sh"]
    missing = [f for f in required if not (domain / f).exists()]
    if missing:
        return L("Workflow Files", "RED", f"Missing: {', '.join(missing)}")
    return L("Workflow Files", "GREEN", "All scaffold files present", "4/4")


def ck_experiment_count(rows: list[dict], **_) -> Light:
    n = len(rows)
    if n == 0:
        return L("Experiments", "RED", "No experiments found")
    if n < 5:
        return L("Experiments", "YELLOW", f"Only {n} — too early to judge")
    return L("Experiments", "GREEN", f"{n} experiments logged", str(n))


def ck_best_score(rows: list[dict], **_) -> Light:
    best, best_id = None, None
    for r in rows:
        try:
            s = float(r["score"])
        except (ValueError, TypeError, KeyError):
            continue
        if best is None or s < best:
            best, best_id = s, r.get("exp_id", "?")
    if best is None:
        return L("Best Score", "RED", "No valid scores recorded")
    return L("Best Score", "GREEN", f"{best:.6f} ({best_id})", f"{best:.6f}")


def ck_total_improvement(rows: list[dict], **_) -> Light:
    valid = scores(rows)
    if len(valid) < 2:
        return L("Total Improvement", "GREEN", "Not enough data yet")
    first, best = valid[0], min(valid)
    delta = first - best
    pct = delta / first * 100 if first else 0
    if delta <= 0:
        return L("Total Improvement", "RED", "No improvement over baseline")
    return L("Total Improvement", "GREEN", f"{delta:.4f} ({pct:.1f}%) from baseline {first:.4f}", f"-{delta:.4f}")


def ck_score_variance(rows: list[dict], **_) -> Light:
    recent = scores(rows)[-10:]
    if len(recent) < 3:
        return L("Recent Variance", "GREEN", "Not enough data")
    mean = sum(recent) / len(recent)
    var = sum((s - mean) ** 2 for s in recent) / len(recent)
    std = var ** 0.5
    cv = std / abs(mean) * 100 if mean else 0
    if cv > 5:
        return L("Recent Variance", "RED", f"CV={cv:.1f}% — results are noisy, configs may be unstable")
    if cv > 2:
        return L("Recent Variance", "YELLOW", f"CV={cv:.1f}% — moderate spread in recent scores")
    return L("Recent Variance", "GREEN", f"CV={cv:.1f}% — results are tight", f"{std:.4f}")


def ck_logs_present(domain: Path, **_) -> Light:
    logs = domain / "logs"
    if not logs.is_dir():
        return L("Agent Logs", "RED", "logs/ directory missing — no trace capture")
    n = len(list(logs.glob("*.jsonl")))
    if n == 0:
        return L("Agent Logs", "RED", "No JSONL logs found in logs/")
    return L("Agent Logs", "GREEN", f"{n} log files captured", str(n))


# ── CRASH ANALYSIS (7-10) ────────────────────────────────────────────────────

def ck_crash_rate(rows: list[dict], **_) -> Light:
    total = len(rows)
    if total == 0:
        return L("Crash Rate", "GREEN", "No experiments yet")
    crashes = sum(1 for r in rows if r.get("status") == "crash")
    rate = crashes / total
    val = f"{crashes}/{total} ({rate:.0%})"
    if rate > 0.20:
        return L("Crash Rate", "RED", f"{crashes} crashes — agents hitting broken configs", val)
    if rate > 0.10:
        return L("Crash Rate", "YELLOW", f"{crashes} crashes — some preventable failures", val)
    return L("Crash Rate", "GREEN", f"{crashes} crashes, within tolerance", val)


def ck_crash_streak(rows: list[dict], **_) -> Light:
    max_streak = 0
    streak = 0
    for r in rows:
        if r.get("status") == "crash":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    if max_streak >= 3:
        return L("Crash Streak", "RED", f"{max_streak} consecutive crashes — agent stuck on broken config")
    if max_streak >= 2:
        return L("Crash Streak", "YELLOW", f"{max_streak} back-to-back crashes")
    return L("Crash Streak", "GREEN", "No crash streaks", str(max_streak))


def ck_agent_crash_skew(rows: list[dict], **_) -> Light:
    agents = defaultdict(lambda: {"total": 0, "crashes": 0})
    for r in rows:
        a = r.get("agent", "?")
        agents[a]["total"] += 1
        if r.get("status") == "crash":
            agents[a]["crashes"] += 1
    if len(agents) < 2:
        return L("Crash Skew", "GREEN", "Single agent — no skew to measure")
    rates = {a: d["crashes"] / d["total"] if d["total"] else 0 for a, d in agents.items()}
    worst = max(rates, key=rates.get)
    best_a = min(rates, key=rates.get)
    if rates[worst] > 0.25 and rates[worst] > rates[best_a] * 3:
        return L("Crash Skew", "RED", f"{worst} crashes {rates[worst]:.0%} vs {best_a} {rates[best_a]:.0%}")
    if rates[worst] > rates[best_a] * 2 and rates[worst] > 0.10:
        return L("Crash Skew", "YELLOW", f"{worst} crashes more ({rates[worst]:.0%} vs {rates[best_a]:.0%})")
    return L("Crash Skew", "GREEN", "Crash rates balanced across agents")


def ck_recent_crashes(rows: list[dict], **_) -> Light:
    last10 = rows[-10:] if len(rows) >= 10 else rows
    crashes = sum(1 for r in last10 if r.get("status") == "crash")
    if crashes >= 4:
        return L("Recent Crashes", "RED", f"{crashes}/{len(last10)} recent experiments crashed")
    if crashes >= 2:
        return L("Recent Crashes", "YELLOW", f"{crashes}/{len(last10)} recent experiments crashed")
    return L("Recent Crashes", "GREEN", f"{crashes} crashes in last {len(last10)} experiments")


# ── PROGRESS (11-15) ─────────────────────────────────────────────────────────

def ck_stagnation(rows: list[dict], **_) -> Light:
    best, stag = None, 0
    for r in rows:
        try:
            s = float(r["score"])
        except (ValueError, TypeError, KeyError):
            continue
        if best is None or s < best:
            best, stag = s, 0
        else:
            stag += 1
    if stag >= 15:
        return L("Stagnation", "RED", f"{stag} experiments since last improvement — search exhausted")
    if stag >= 8:
        return L("Stagnation", "YELLOW", f"{stag} experiments since last improvement")
    return L("Stagnation", "GREEN", f"Last improvement {stag} experiments ago", str(stag))


def ck_breakthrough_rate(rows: list[dict], **_) -> Light:
    total = len(rows)
    if total == 0:
        return L("Breakthrough Rate", "GREEN", "No experiments yet")
    bts = len(breakthrough_indices(rows))
    rate = bts / total
    val = f"{bts}/{total} ({rate:.0%})"
    if rate < 0.05 and total >= 20:
        return L("Breakthrough Rate", "RED", f"Only {bts} in {total} — diminishing returns", val)
    if rate < 0.15 and total >= 10:
        return L("Breakthrough Rate", "YELLOW", f"{bts} breakthroughs — search narrowing", val)
    return L("Breakthrough Rate", "GREEN", f"{bts} breakthroughs in {total} experiments", val)


def ck_improvement_velocity(rows: list[dict], **_) -> Light:
    if len(rows) < 10:
        return L("Velocity", "GREEN", "Too early to measure")
    mid = len(rows) // 2
    def count_bt(sl):
        best, bt = None, 0
        for r in sl:
            try:
                s = float(r["score"])
            except (ValueError, TypeError, KeyError):
                continue
            if best is None or s < best:
                best = s
                bt += 1
        return bt
    h1, h2 = count_bt(rows[:mid]), count_bt(rows[mid:])
    val = f"{h1} → {h2}"
    if h2 == 0 and len(rows) >= 20:
        return L("Velocity", "RED", f"Zero breakthroughs in 2nd half ({len(rows)-mid} exp)", val)
    if h2 < h1 * 0.3 and len(rows) >= 20:
        return L("Velocity", "YELLOW", f"Slowing: {h1} → {h2} breakthroughs", val)
    return L("Velocity", "GREEN", f"Breakthroughs {h1} (1st) → {h2} (2nd)", val)


def ck_keep_rate(rows: list[dict], **_) -> Light:
    total = len(rows)
    if total == 0:
        return L("Keep Rate", "GREEN", "No experiments yet")
    keeps = sum(1 for r in rows if r.get("status") == "keep")
    rate = keeps / total
    val = f"{keeps}/{total} ({rate:.0%})"
    if rate < 0.15 and total >= 15:
        return L("Keep Rate", "RED", f"Only {rate:.0%} experiments kept — mostly waste")
    if rate < 0.30 and total >= 10:
        return L("Keep Rate", "YELLOW", f"{rate:.0%} keep rate — high discard ratio", val)
    return L("Keep Rate", "GREEN", f"{rate:.0%} of experiments kept", val)


def ck_regression_rate(rows: list[dict], **_) -> Light:
    valid = [r for r in rows if r.get("status") not in ("crash",)]
    if len(valid) < 5:
        return L("Regression Rate", "GREEN", "Not enough data")
    best_so_far = None
    regressions = 0
    for r in valid:
        try:
            s = float(r["score"])
        except (ValueError, TypeError, KeyError):
            continue
        if best_so_far is None:
            best_so_far = s
            continue
        if s > best_so_far * 1.01:
            regressions += 1
        if s < best_so_far:
            best_so_far = s
    rate = regressions / len(valid)
    if rate > 0.30:
        return L("Regression Rate", "RED", f"{regressions} regressions ({rate:.0%}) — agents going backwards")
    if rate > 0.15:
        return L("Regression Rate", "YELLOW", f"{regressions} regressions ({rate:.0%})")
    return L("Regression Rate", "GREEN", f"{regressions} regressions ({rate:.0%})", str(regressions))


# ── COORDINATION (16-21) ─────────────────────────────────────────────────────

def ck_agent_balance(rows: list[dict], **_) -> Light:
    agents = Counter(r.get("agent", "?") for r in rows)
    if len(agents) <= 1:
        return L("Agent Balance", "GREEN", "Single agent run")
    total = len(rows)
    max_share = max(agents.values()) / total
    counts = ", ".join(f"{a}={c}" for a, c in agents.most_common())
    if max_share > 0.80:
        dom = agents.most_common(1)[0][0]
        return L("Agent Balance", "RED", f"{dom} dominates ({max_share:.0%}) — other agent stuck?", counts)
    if max_share > 0.65:
        return L("Agent Balance", "YELLOW", f"Slight imbalance: {counts}")
    return L("Agent Balance", "GREEN", f"Balanced: {counts}")


def ck_duplicate_rate(rows: list[dict], **_) -> Light:
    descs = [r.get("description", "") for r in rows]
    dupes = 0
    for i, d in enumerate(descs):
        for j in range(i):
            if SequenceMatcher(None, d.lower(), descs[j].lower()).ratio() > 0.85:
                dupes += 1
                break
    total = len(rows)
    if total == 0:
        return L("Duplicates", "GREEN", "No experiments yet")
    rate = dupes / total
    val = f"{dupes}/{total}"
    if rate > 0.15:
        return L("Duplicates", "RED", f"{dupes} near-duplicates — agents not coordinating", val)
    if rate > 0.05:
        return L("Duplicates", "YELLOW", f"{dupes} near-duplicates — some redundant work", val)
    return L("Duplicates", "GREEN", f"Minimal redundancy ({dupes})", val)


def ck_same_idea_parallel(rows: list[dict], **_) -> Light:
    """Detect same hypothesis tested by both agents within 2 experiment IDs."""
    pairs = 0
    pair_ids = []
    for i in range(len(rows)):
        for j in range(i+1, min(i+3, len(rows))):
            if rows[i].get("agent") != rows[j].get("agent"):
                d1 = rows[i].get("description", "").lower()
                d2 = rows[j].get("description", "").lower()
                # Check for shared key phrases (not just string similarity)
                words1 = set(re.findall(r'[a-z_]+=[\d.]+|[a-z]{5,}', d1))
                words2 = set(re.findall(r'[a-z_]+=[\d.]+|[a-z]{5,}', d2))
                overlap = words1 & words2
                if len(overlap) >= 3 and len(overlap) / max(len(words1), len(words2), 1) > 0.5:
                    pairs += 1
                    pair_ids.append(f"{rows[i].get('exp_id','?')}+{rows[j].get('exp_id','?')}")
    if pairs >= 3:
        return L("Parallel Waste", "RED", f"{pairs} same-idea collisions: {', '.join(pair_ids[:3])}")
    if pairs >= 1:
        return L("Parallel Waste", "YELLOW", f"{pairs} same-idea collision(s): {', '.join(pair_ids[:2])}")
    return L("Parallel Waste", "GREEN", "No parallel idea collisions detected")


def ck_blackboard_health(domain: Path, **_) -> Light:
    bb = load_lines(domain / "blackboard.md")
    n = len(bb)
    if n == 0:
        return L("Blackboard", "RED", "Empty — agents have no shared state")
    if n < 10:
        return L("Blackboard", "YELLOW", f"Thin ({n} lines) — agents may not be sharing")
    return L("Blackboard", "GREEN", f"Active ({n} lines)", str(n))


def ck_race_conditions(learnings_text: str, mistakes_text: str, **_) -> Light:
    text = (learnings_text + mistakes_text).lower()
    hits = text.count("race condition") + text.count("corrupted") + text.count("contaminated")
    if hits >= 4:
        return L("Race Conditions", "RED", f"{hits} mentions — best/train.py integrity compromised")
    if hits >= 2:
        return L("Race Conditions", "YELLOW", f"{hits} mentions — occasional file conflicts")
    return L("Race Conditions", "GREEN", "No race condition issues detected", str(hits))


def ck_design_diversity(rows: list[dict], **_) -> Light:
    designs = [r.get("design", "unknown") for r in rows if r.get("design")]
    if not designs:
        return L("Design Diversity", "GREEN", "No design labels")
    counts = Counter(designs)
    n_cats = len(counts)
    top = counts.most_common(1)[0]
    top_share = top[1] / len(designs)
    if n_cats <= 1 and len(rows) >= 10:
        return L("Design Diversity", "RED", f"All experiments are '{top[0]}' — explore other axes")
    if top_share > 0.60 and len(rows) >= 10:
        return L("Design Diversity", "YELLOW", f"'{top[0]}' dominates ({top_share:.0%}) — {n_cats} categories total")
    return L("Design Diversity", "GREEN", f"{n_cats} design categories explored", ", ".join(f"{k}={v}" for k, v in counts.most_common()))


# ── CONSTRAINT MEMORY (22-25) ────────────────────────────────────────────────

def ck_constraint_violations(rows: list[dict], mistakes_text: str, **_) -> Light:
    bad_keywords = []
    for line in mistakes_text.splitlines():
        m = re.match(r"- \*\*Lesson\*\*:\s*(.*)", line)
        if m:
            lesson = m.group(1).lower()
            if any(w in lesson for w in ["don't", "never", "do not", "stop"]):
                bad_keywords.extend(re.findall(r'[a-z_]+=[\d.]+', lesson))

    violations = []
    for r in rows[-20:]:
        desc = r.get("description", "").lower()
        status = r.get("status", "")
        if status in ("crash", "discard"):
            for bk in bad_keywords:
                if bk in desc:
                    violations.append(r.get("exp_id", "?"))
                    break

    n = len(violations)
    if n >= 3:
        return L("Constraint Violations", "RED", f"{n} experiments retested known-bad ideas ({', '.join(violations[:3])})")
    if n >= 1:
        return L("Constraint Violations", "YELLOW", f"{n} retested known-bad ideas ({', '.join(violations)})")
    return L("Constraint Violations", "GREEN", "No known-bad retests in recent experiments")


def ck_mistakes_documented(domain: Path, rows: list[dict], **_) -> Light:
    mistakes = load_text(domain / "MISTAKES.md")
    crashes = sum(1 for r in rows if r.get("status") == "crash")
    discards = sum(1 for r in rows if r.get("status") == "discard")
    failures = crashes + discards
    sections = len(re.findall(r'^## ', mistakes, re.MULTILINE))
    if failures > 5 and sections < 3:
        return L("Mistakes Documented", "RED", f"{failures} failures but only {sections} writeups — not learning")
    if failures > 10 and sections < failures * 0.3:
        return L("Mistakes Documented", "YELLOW", f"{sections} writeups for {failures} failures — some undocumented")
    return L("Mistakes Documented", "GREEN", f"{sections} failure writeups for {failures} failures", f"{sections}/{failures}")


def ck_learnings_growth(domain: Path, rows: list[dict], **_) -> Light:
    learnings = load_lines(domain / "LEARNINGS.md")
    bullets = sum(1 for l in learnings if l.strip().startswith("- "))
    sections = sum(1 for l in learnings if l.strip().startswith("## "))
    total = len(rows)
    if total >= 20 and sections < 3:
        return L("Learnings Growth", "RED", f"Only {sections} learning sections after {total} experiments")
    if total >= 10 and sections < 2:
        return L("Learnings Growth", "YELLOW", f"{sections} sections — agents not synthesizing enough")
    return L("Learnings Growth", "GREEN", f"{sections} sections, {bullets} facts documented", f"{sections}s/{bullets}b")


def ck_dead_ends_tracked(mistakes_text: str, program_text: str, **_) -> Light:
    # Count "Lesson" lines with "don't/never" in mistakes
    constraints = 0
    for line in mistakes_text.splitlines():
        m = re.match(r"- \*\*Lesson\*\*:\s*(.*)", line)
        if m and any(w in m.group(1).lower() for w in ["don't", "never", "do not"]):
            constraints += 1
    # Check if program.md has "do not" / "avoid" rules
    program_lower = program_text.lower()
    bans = program_lower.count("do not") + program_lower.count("avoid") + program_lower.count("don't")
    if constraints >= 5 and bans < 2:
        return L("Dead Ends Tracked", "RED", f"{constraints} learned constraints but program.md has {bans} bans")
    if constraints >= 3 and bans < constraints * 0.5:
        return L("Dead Ends Tracked", "YELLOW", f"{constraints} constraints, only {bans} reflected in program.md")
    return L("Dead Ends Tracked", "GREEN", f"{constraints} constraints, {bans} bans in program.md")


# ── SELF-AWARENESS (26-30) ───────────────────────────────────────────────────

def ck_telemetry_files(domain: Path, **_) -> Light:
    files = ["DESIRES.md", "MISTAKES.md", "LEARNINGS.md"]
    missing = [f for f in files if not (domain / f).exists() or (domain / f).stat().st_size < 20]
    if len(missing) == 3:
        return L("Telemetry Files", "RED", "No telemetry — agents not reflecting")
    if missing:
        return L("Telemetry Files", "YELLOW", f"Missing: {', '.join(missing)}")
    return L("Telemetry Files", "GREEN", "All telemetry files present", "3/3")


def ck_desires_addressed(domain: Path, **_) -> Light:
    desires = [l.strip() for l in load_lines(domain / "DESIRES.md") if l.strip().startswith("- ")]
    if not desires:
        return L("Desires Addressed", "GREEN", "No desires filed")
    program = load_text(domain / "program.md").lower()
    addressed = 0
    for d in desires:
        words = re.findall(r'\b[a-z]{4,}\b', d.lower())
        if any(w in program for w in words[:3]):
            addressed += 1
    rate = addressed / len(desires)
    val = f"{addressed}/{len(desires)}"
    if rate < 0.2:
        return L("Desires Addressed", "RED", f"Only {addressed}/{len(desires)} desires in program.md", val)
    if rate < 0.5:
        return L("Desires Addressed", "YELLOW", f"{addressed}/{len(desires)} desires partially addressed", val)
    return L("Desires Addressed", "GREEN", f"{addressed}/{len(desires)} desires addressed", val)


def ck_desires_volume(domain: Path, **_) -> Light:
    desires = [l.strip() for l in load_lines(domain / "DESIRES.md") if l.strip().startswith("- ")]
    n = len(desires)
    if n > 15:
        return L("Desire Backlog", "RED", f"{n} unaddressed desires — agents frustrated, gardener not listening")
    if n > 8:
        return L("Desire Backlog", "YELLOW", f"{n} desires piling up")
    return L("Desire Backlog", "GREEN", f"{n} desires on file", str(n))


def ck_config_snapshot(domain: Path, **_) -> Light:
    best_dir = domain / "best"
    if not best_dir.is_dir():
        return L("Config Snapshot", "YELLOW", "No best/ directory — can't verify winning config")
    train = best_dir / "train.py"
    if not train.exists():
        return L("Config Snapshot", "RED", "best/train.py missing — no record of winning config")
    return L("Config Snapshot", "GREEN", "best/train.py exists")


def ck_results_integrity(rows: list[dict], **_) -> Light:
    """Check for gaps, duplicates, or malformed rows in results."""
    ids = [r.get("exp_id", "") for r in rows]
    issues = []
    # Duplicate IDs
    id_counts = Counter(ids)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    if dupes:
        issues.append(f"duplicate IDs: {', '.join(dupes.keys())}")
    # Missing scores on non-crash rows
    for r in rows:
        if r.get("status") != "crash":
            try:
                float(r.get("score", "nan"))
            except (ValueError, TypeError):
                issues.append(f"{r.get('exp_id','?')} has no score but isn't crash")
    if len(issues) >= 2:
        return L("Results Integrity", "RED", f"{len(issues)} issues: {'; '.join(issues[:2])}")
    if issues:
        return L("Results Integrity", "YELLOW", issues[0])
    return L("Results Integrity", "GREEN", f"{len(rows)} rows, all well-formed")


# ═══════════════════════════════════════════════════════════════════════════════
# Assemble
# ═══════════════════════════════════════════════════════════════════════════════

SECTIONS = [
    ("RUN HEALTH",        [ck_workflow_files, ck_experiment_count, ck_best_score,
                           ck_total_improvement, ck_score_variance, ck_logs_present]),
    ("CRASH ANALYSIS",    [ck_crash_rate, ck_crash_streak, ck_agent_crash_skew,
                           ck_recent_crashes]),
    ("PROGRESS",          [ck_stagnation, ck_breakthrough_rate, ck_improvement_velocity,
                           ck_keep_rate, ck_regression_rate]),
    ("COORDINATION",      [ck_agent_balance, ck_duplicate_rate, ck_same_idea_parallel,
                           ck_blackboard_health, ck_race_conditions, ck_design_diversity]),
    ("CONSTRAINT MEMORY", [ck_constraint_violations, ck_mistakes_documented,
                           ck_learnings_growth, ck_dead_ends_tracked]),
    ("SELF-AWARENESS",    [ck_telemetry_files, ck_desires_addressed, ck_desires_volume,
                           ck_config_snapshot, ck_results_integrity]),
]


def build_stoplight(domain: Path) -> list[tuple[str, list[Light]]]:
    results_path = domain / "results.tsv"
    if not results_path.exists():
        return [("ERROR", [L("Results", "RED", "results.tsv not found")])]

    rows = load_results(results_path)
    ctx = dict(
        domain=domain,
        rows=rows,
        learnings_text=load_text(domain / "LEARNINGS.md"),
        mistakes_text=load_text(domain / "MISTAKES.md"),
        program_text=load_text(domain / "program.md"),
    )

    result = []
    for section_name, checks in SECTIONS:
        lights = []
        for fn in checks:
            lights.append(fn(**ctx))
        result.append((section_name, lights))
    return result


def render_text(sections: list[tuple[str, list[Light]]], domain_name: str) -> str:
    lines = []
    all_lights = [l for _, ls in sections for l in ls]
    reds = sum(1 for l in all_lights if l.color == "RED")
    yellows = sum(1 for l in all_lights if l.color == "YELLOW")
    greens = sum(1 for l in all_lights if l.color == "GREEN")

    lines.append(f"{'─'*72}")
    lines.append(f"  STOPLIGHT — {domain_name}")
    lines.append(f"{'─'*72}")

    for section_name, lights in sections:
        lines.append(f"  {ANSI['RESET']}┄┄ {section_name} ┄┄")
        for l in lights:
            a = ANSI[l.color]
            r = ANSI["RESET"]
            pad = max(1, 24 - len(l.name))
            lines.append(f"  {a}●{r} {l.name}{' ' * pad}{l.summary}")

    lines.append(f"{'─'*72}")
    parts = []
    if reds:
        parts.append(f"{ANSI['RED']}{reds} RED{ANSI['RESET']}")
    if yellows:
        parts.append(f"{ANSI['YELLOW']}{yellows} YELLOW{ANSI['RESET']}")
    parts.append(f"{ANSI['GREEN']}{greens} GREEN{ANSI['RESET']}")
    lines.append(f"  {' | '.join(parts)}  ({len(all_lights)} checks)")
    lines.append(f"{'─'*72}")

    return "\n".join(lines)


def render_json(sections: list[tuple[str, list[Light]]]) -> str:
    out = {}
    for section_name, lights in sections:
        out[section_name] = [asdict(l) for l in lights]
    return json.dumps(out, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TrustLoop Stoplight Report")
    parser.add_argument("domain", help="Path to domain directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    domain = Path(args.domain)
    if not domain.is_dir():
        print(f"Error: {domain} is not a directory", file=sys.stderr)
        sys.exit(1)

    sections = build_stoplight(domain)

    if args.json:
        print(render_json(sections))
    else:
        print(render_text(sections, domain.name))


if __name__ == "__main__":
    main()

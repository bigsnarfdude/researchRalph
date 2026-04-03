#!/usr/bin/env python3
"""
trustloop_api.py — TrustLoop v4.9: single MCP server, all report types.

Replaces trustloop_mcp.py + rrma_mcp.py.

Report hierarchy:
  EXPERIMENT  — per-run tombstone: outcome class, novelty, agent
  SUMMARY     — intent vs outcome: agent efficiency, insights
  STATUS      — 5-line health: stop/continue decision
  DIAGNOSIS   — full report: action items, gardener checks, evidence

Trace forensics degrade gracefully when no traces file is present.
Domain scoring reads trustloop: section from config.yaml for column mapping.

Usage:
    python3 tools/trustloop_api.py
    RRMA_ROOT=/path/to/researchRalph python3 tools/trustloop_api.py
    TRUSTLOOP_TRACES=/path/to.jsonl python3 tools/trustloop_api.py

Requires:
    pip install "mcp[cli]"
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Tool imports (libraries, not inlined) ────────────────────────────────────

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from trustloop_scorer import (
    score_domain,
    format_report,
    load_domain_scorer_config,
    detect_score_direction,
    _run_status,
)

try:
    from trace_forensics import TraceStore
    _HAS_TRACE_FORENSICS = True
except ImportError:
    _HAS_TRACE_FORENSICS = False

# ── Config ───────────────────────────────────────────────────────────────────

RRMA_ROOT = Path(os.environ.get("RRMA_ROOT", Path(__file__).resolve().parent.parent))
DOMAINS_DIR = RRMA_ROOT / "domains"
TRACES_PATH = os.environ.get("TRUSTLOOP_TRACES", "/tmp/rrma_traces.jsonl")

if not DOMAINS_DIR.exists():
    print(f"[trustloop_api] ERROR: domains dir not found at {DOMAINS_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"[trustloop_api] root={RRMA_ROOT} traces={TRACES_PATH}", file=sys.stderr)

# ── Lazy trace store (no exit if missing) ────────────────────────────────────

_store: "TraceStore | None" = None


def _get_store() -> "TraceStore | None":
    global _store
    if _store is not None:
        return _store
    if not _HAS_TRACE_FORENSICS:
        return None
    if not Path(TRACES_PATH).exists():
        return None
    try:
        _store = TraceStore(TRACES_PATH)
        print(
            f"[trustloop_api] loaded {len(_store.traces)} traces "
            f"from {TRACES_PATH}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[trustloop_api] failed to load traces: {e}", file=sys.stderr)
    return _store


def _no_traces() -> dict:
    if not _HAS_TRACE_FORENSICS:
        return {"error": "trace_forensics not installed — pip install anthropic"}
    return {
        "error": f"No traces at {TRACES_PATH}",
        "hint": "Set TRUSTLOOP_TRACES or run: trustloop ingest <domain>",
    }


# ── Domain file helpers ───────────────────────────────────────────────────────

ARTIFACT_NAMES = {
    "blackboard": ["blackboard.md"],
    "meta_blackboard": ["meta-blackboard.md"],
    "program": ["program.md"],
    "results": ["results.tsv"],
    "experiments": ["experiments.jsonl"],
    "desires": ["DESIRES.md", "desires.md"],
    "learnings": ["LEARNINGS.md", "learnings.md"],
    "mistakes": ["MISTAKES.md", "mistakes.md"],
    "calibration": ["calibration.md"],
    "stoplight": ["stoplight.md"],
    "config": ["config.yaml", "prompt_config.yaml"],
}


def _find_artifact(domain_dir: Path, artifact_type: str) -> Path | None:
    for name in ARTIFACT_NAMES.get(artifact_type, [f"{artifact_type}.md"]):
        p = domain_dir / name
        if p.exists():
            return p
    return None


def _read_artifact(domain_dir: Path, artifact_type: str) -> dict:
    p = _find_artifact(domain_dir, artifact_type)
    if p is None:
        return {"error": f"Artifact '{artifact_type}' not found in {domain_dir.name}"}
    content = p.read_text(errors="replace")
    return {
        "artifact_type": artifact_type,
        "filename": p.name,
        "lines": content.count("\n"),
        "bytes": len(content.encode()),
        "content_hash": hashlib.md5(content.encode()).hexdigest()[:12],
        "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        "content": content,
    }


def _parse_results_tsv(domain_dir: Path) -> list[dict]:
    """Parse results.tsv respecting per-domain trustloop config."""
    p = _find_artifact(domain_dir, "results")
    if p is None:
        return []

    # Load scorer config for column remapping
    scorer_cfg = load_domain_scorer_config(domain_dir)
    score_col = scorer_cfg.get("score_column", "score")
    time_col = scorer_cfg.get("time_column", "train_min")
    direction = scorer_cfg.get("score_direction", None)

    text = p.read_text(errors="replace")
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return []

    # Detect header
    first = lines[0].split("\t")
    if any(h.strip().lower() in ("exp_id", "score", "residual") for h in first):
        header = [h.strip().lower().replace("-", "_") for h in first]
        data_lines = lines[1:]
    else:
        header = ["exp_id", "score", "sol_norm", "sol_mean", "status",
                  "description", "agent", "design", "elapsed_s", "sol_energy"]
        data_lines = lines

    rows = []
    for line in data_lines:
        fields = line.split("\t")
        row = {h: (fields[i].strip() if i < len(fields) else "") for i, h in enumerate(header)}
        # Remap columns
        if score_col != "score" and score_col in row and "score" not in row:
            row["score"] = row[score_col]
        if time_col != "train_min" and time_col in row and "train_min" not in row:
            row["train_min"] = row[time_col]
        try:
            row["_score_f"] = float(row.get("score", "nan"))
        except (ValueError, TypeError):
            row["_score_f"] = None
        rows.append(row)

    return rows


def _domain_summary(domain_dir: Path) -> dict:
    """Quick summary for a domain — direction-aware best score."""
    scorer_cfg = load_domain_scorer_config(domain_dir)
    direction = scorer_cfg.get("score_direction", None)

    # Config metadata
    config_path = _find_artifact(domain_dir, "config")
    metric = "unknown"
    if config_path:
        for line in config_path.read_text().splitlines():
            if line.strip().startswith("metric:"):
                metric = line.split(":", 1)[1].strip()
                break

    results = _parse_results_tsv(domain_dir)
    scores = [r["_score_f"] for r in results if r.get("_score_f") is not None]

    if direction is None:
        descs = [r.get("description", "") for r in results]
        direction = detect_score_direction(domain_dir.name, descs)

    best = (min(scores) if direction == "lower" else max(scores)) if scores else None

    # Last modified
    artifact_files = [
        f for f in domain_dir.iterdir()
        if f.suffix in (".md", ".tsv", ".jsonl") and f.name != "README.md"
    ]
    last_modified = None
    if artifact_files:
        newest = max(artifact_files, key=lambda f: f.stat().st_mtime)
        last_modified = datetime.fromtimestamp(newest.stat().st_mtime).isoformat()

    log_count = len(list((domain_dir / "logs").glob("*.jsonl"))) if (domain_dir / "logs").exists() else 0

    return {
        "domain": domain_dir.name,
        "metric": metric,
        "direction": direction,
        "experiment_count": len(results),
        "best_score": best,
        "log_files": log_count,
        "last_modified": last_modified,
    }


def _get_screen_sessions() -> list[str]:
    try:
        out = subprocess.run(["screen", "-ls"], capture_output=True, text=True, timeout=5)
        sessions = []
        for line in out.stdout.splitlines():
            m = re.search(r"\d+\.(\S+)", line)
            if m:
                sessions.append(m.group(1))
        return sessions
    except Exception:
        return []


def _require_domain(domain: str) -> tuple[Path | None, dict | None]:
    """Return (domain_dir, None) or (None, error_dict)."""
    domain_dir = DOMAINS_DIR / domain
    if not domain_dir.exists():
        available = [d.name for d in sorted(DOMAINS_DIR.iterdir()) if d.is_dir()
                     and not d.name.startswith(".") and d.name != "template"]
        return None, {"error": f"Domain '{domain}' not found", "available": available}
    return domain_dir, None


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("trustloop", json_response=True)


# ══════════════════════════════════════════════════════════════════════════════
# Domain discovery
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def domains() -> dict:
    """
    List all RRMA domains with: name, metric, direction, experiment count,
    best score, log files, last modified. Start here to see what's available.
    """
    result = []
    for d in sorted(DOMAINS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "template":
            continue
        if not (_find_artifact(d, "config") or _find_artifact(d, "program")):
            continue
        result.append(_domain_summary(d))
    return {"count": len(result), "domains": result}


@mcp.tool()
def run_status(domain: str | None = None) -> dict:
    """
    Check if an RRMA run is active. Detects screen sessions (agent0, agent1,
    meta-agent, gardener). If domain is specified, includes latest scores.

    Args:
        domain: Optional domain name for domain-specific status
    """
    sessions = _get_screen_sessions()
    agent_sessions = [s for s in sessions if re.match(r"(agent\d+|rrma-worker\d*)$", s)]
    meta_sessions = [s for s in sessions if "meta" in s.lower()]
    gardener_sessions = [s for s in sessions if "gardener" in s.lower()]

    out: dict = {
        "running": len(agent_sessions) > 0,
        "agent_sessions": sorted(agent_sessions),
        "meta_sessions": meta_sessions,
        "gardener_sessions": gardener_sessions,
        "all_screen_sessions": sorted(sessions),
    }

    if domain:
        domain_dir, err = _require_domain(domain)
        if err:
            out.update(err)
        else:
            results = _parse_results_tsv(domain_dir)
            scores = [r["_score_f"] for r in results if r.get("_score_f") is not None]
            out["domain"] = domain
            out["experiment_count"] = len(results)
            out["latest"] = [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in results[-5:]
            ]

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Report hierarchy
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def report_experiment(domain: str, exp_id: str) -> dict:
    """
    EXPERIMENT report — tombstone for a single experiment.
    Returns outcome class (BREAKTHROUGH/INCREMENTAL/PLATEAU/REGRESSION/CRASH),
    novelty score, redundancy, agent, description, and score.

    Args:
        domain: Domain name (e.g. "nirenberg-1d")
        exp_id: Experiment ID (e.g. "exp042")
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    r = score_domain(domain_dir, with_traces=False)
    for exp in r.experiments:
        if exp.exp_id == exp_id:
            return {
                "exp_id": exp.exp_id,
                "score": exp.score,
                "status": exp.status,
                "outcome_class": exp.outcome_class,
                "is_best_ever": exp.is_best_ever,
                "improvement_pct": exp.improvement_pct,
                "novelty": exp.novelty,
                "redundant_with": exp.redundant_with,
                "agent": exp.agent,
                "design": exp.design,
                "description": exp.description,
                "train_min": exp.train_min,
            }

    exp_ids = [e.exp_id for e in r.experiments]
    return {"error": f"Experiment '{exp_id}' not found", "available": exp_ids[-20:]}


@mcp.tool()
def report_summary(domain: str) -> dict:
    """
    SUMMARY report — intent vs outcome reconciliation.
    Agent efficiency table, top insights, best score, stagnation depth,
    and telemetry signals (desires, key learnings, mistake lessons).

    Args:
        domain: Domain name (e.g. "nirenberg-1d")
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    r = score_domain(domain_dir, with_traces=False)

    # Agent table
    agents = [
        {
            "agent_id": a.agent_id,
            "experiments": a.total_experiments,
            "keeps": a.keeps,
            "breakthroughs": a.breakthroughs,
            "crashes": a.crashes,
            "success_rate": round(a.success_rate, 3),
            "waste_ratio": round(a.waste_ratio, 3),
            "best_score": a.best_score,
            "best_exp": a.best_exp,
        }
        for a in r.agents
    ]

    # Key insights only (no evidence table)
    insights = [
        {"kind": i.kind, "message": i.message, "source": i.source}
        for i in r.insights
        if i.kind in ("winning_strategy", "dead_end", "key_learning",
                      "desire", "top_agent", "recurring_mistake")
    ]

    return {
        "domain": domain,
        "total_experiments": r.total_experiments,
        "best_score": r.best_score,
        "best_exp": r.best_exp,
        "score_direction": r.score_direction,
        "stagnation_depth": r.stagnation_depth,
        "efficiency_score": round(r.efficiency_score, 4),
        "breakthroughs": r.breakthroughs,
        "crashes": r.crashes,
        "redundant_pairs": r.redundant_pairs,
        "agents": agents,
        "insights": insights,
        "desires": r.telemetry.desires,
        "learnings": r.telemetry.learnings[:10],
    }


@mcp.tool()
def report_status(domain: str) -> dict:
    """
    STATUS report — 5-line health check. Stop or continue?
    Returns: status label, key numbers, anomaly alerts, workflow failures.
    Use this before deciding whether to launch another generation.

    Args:
        domain: Domain name (e.g. "nirenberg-1d")
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    r = score_domain(domain_dir, with_traces=False)
    status = _run_status(r)

    alerts = [
        {"severity": a.severity, "category": a.category, "message": a.message}
        for a in r.anomalies
        if a.severity in ("ALERT", "WARN")
    ]

    failed_checks = [
        {"check": c.check, "detail": c.detail}
        for c in r.workflow_checks
        if not c.passed
    ]

    return {
        "domain": domain,
        "status": status,
        "experiments": r.total_experiments,
        "best_score": r.best_score,
        "best_exp": r.best_exp,
        "breakthroughs": r.breakthroughs,
        "crashes": r.crashes,
        "stagnation_depth": r.stagnation_depth,
        "efficiency": round(r.efficiency_score, 4),
        "alerts": alerts,
        "failed_workflow_checks": failed_checks,
    }


@mcp.tool()
def report_diagnosis(domain: str, with_traces: bool = False) -> dict:
    """
    DIAGNOSIS report — full TrustLoop report as formatted text.
    Sections: Run Health, Diagnosis, Action Items, Gardener Report,
    Unresolved, Evidence. This is the most detailed report.

    Args:
        domain: Domain name (e.g. "nirenberg-1d")
        with_traces: If True, also reads agent JSONL logs for trace enrichment
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    r = score_domain(domain_dir, with_traces=with_traces)
    return {
        "domain": domain,
        "report": format_report(r),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Artifact access
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def artifact(domain: str, artifact_type: str) -> dict:
    """
    Read a domain artifact (coordination layer between agents).

    Args:
        domain: Domain name
        artifact_type: One of: blackboard, meta_blackboard, program, results,
                       experiments, desires, learnings, mistakes, calibration,
                       stoplight, config
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    art = _read_artifact(domain_dir, artifact_type)
    if "error" in art:
        art["available"] = [t for t in ARTIFACT_NAMES if _find_artifact(domain_dir, t)]
    return art


@mcp.tool()
def artifacts_list(domain: str) -> dict:
    """
    List all artifacts in a domain with sizes and modification times.
    Use before fetching individual artifacts.

    Args:
        domain: Domain name
    """
    domain_dir, err = _require_domain(domain)
    if err:
        return err

    result = []
    for art_type in ARTIFACT_NAMES:
        p = _find_artifact(domain_dir, art_type)
        if p is None:
            continue
        stat = p.stat()
        content = p.read_text(errors="replace")
        result.append({
            "artifact_type": art_type,
            "filename": p.name,
            "lines": content.count("\n"),
            "bytes": stat.st_size,
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:12],
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return {"domain": domain, "count": len(result), "artifacts": result}


# ══════════════════════════════════════════════════════════════════════════════
# Trace forensics (optional — degrade gracefully)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def traces_status() -> dict:
    """
    Overview of loaded trace data: domain, agent count, session count,
    total steps, thinking blocks, tool calls, experiments, best score.
    Returns error if no traces are loaded — see hint for how to load them.
    """
    store = _get_store()
    if store is None:
        return _no_traces()

    s = store.swarm
    agents = []
    for aid in sorted(store.traces_by_agent):
        traces = store.traces_by_agent[aid]
        steps = sum(t["metrics"]["total_steps"] for t in traces)
        thinking = sum(
            sum(1 for st in t["steps"] if st.get("reasoning_content"))
            for t in traces
        )
        agents.append({"agent_id": aid, "sessions": len(traces), "steps": steps, "thinking_blocks": thinking})

    return {
        "domain": s.get("domain", "unknown"),
        "trace_count": len(store.traces),
        "agent_count": len(store.traces_by_agent),
        "total_steps": sum(a["steps"] for a in agents),
        "total_thinking": sum(a["thinking_blocks"] for a in agents),
        "total_tool_calls": sum(
            sum(len(st.get("tool_calls", [])) for st in t["steps"])
            for t in store.traces
        ),
        "total_experiments": s.get("total_experiments", 0),
        "best_score": s.get("best_score"),
        "traces_path": TRACES_PATH,
        "agents": agents,
    }


@mcp.tool()
def traces_agent(
    agent_id: str,
    mode: str = "summary",
    max_results: int = 20,
) -> dict:
    """
    Investigate a specific agent's trace data.
    Modes:
    - "summary": sessions, steps, tool calls, artifact reads/writes
    - "thinking": last N thinking blocks with context
    - "timeline": key events — artifact writes, harness scores

    Args:
        agent_id: Agent to investigate (e.g. "agent0")
        mode: "summary", "thinking", or "timeline"
        max_results: Max thinking blocks or timeline events (default 20)
    """
    store = _get_store()
    if store is None:
        return _no_traces()

    if agent_id not in store.traces_by_agent:
        return {
            "error": f"Unknown agent: {agent_id}",
            "available": sorted(store.traces_by_agent.keys()),
        }

    if mode == "thinking":
        blocks = store.get_agent_thinking(agent_id, max_results)
        return {"agent_id": agent_id, "mode": "thinking", "count": len(blocks), "blocks": blocks}

    if mode == "timeline":
        events = store.get_agent_timeline(agent_id)
        return {"agent_id": agent_id, "mode": "timeline", "count": len(events), "events": events[:max_results]}

    # Summary mode
    traces = store.traces_by_agent[agent_id]
    total_steps = sum(t["metrics"]["total_steps"] for t in traces)
    thinking = sum(sum(1 for st in t["steps"] if st.get("reasoning_content")) for t in traces)
    tool_calls = sum(sum(len(st.get("tool_calls", [])) for st in t["steps"]) for t in traces)

    reads: Counter = Counter()
    writes: Counter = Counter()
    for t in traces:
        for st in t["steps"]:
            for r in st.get("artifact_reads", []):
                reads[r] += 1
            for w in st.get("artifact_writes", []):
                writes[w] += 1

    return {
        "agent_id": agent_id,
        "mode": "summary",
        "sessions": len(traces),
        "steps": total_steps,
        "thinking_blocks": thinking,
        "tool_calls": tool_calls,
        "artifact_reads": dict(reads.most_common()),
        "artifact_writes": dict(writes.most_common()),
    }


@mcp.tool()
def traces_search(query: str, max_results: int = 15) -> dict:
    """
    Full-text search across all agent traces: thinking, output, tool calls,
    artifact names. Returns matching steps with agent, step index, and snippet.

    Args:
        query: Search term (e.g. "fourier", "omega", "blackboard")
        max_results: Maximum results (default 15)
    """
    store = _get_store()
    if store is None:
        return _no_traces()

    results = store.search_traces(query, max_results)
    return {"query": query, "count": len(results), "results": results}


@mcp.tool()
def traces_step(trace_id: str, step_index: int) -> dict:
    """
    Raw step data for a specific trace and step: reasoning, tool calls,
    artifact interactions. Use after traces_agent or traces_search to drill in.

    Args:
        trace_id: Full or partial (first 8 chars) trace ID
        step_index: Step number within the trace
    """
    store = _get_store()
    if store is None:
        return _no_traces()
    return store.get_step(trace_id, step_index)


@mcp.tool()
def traces_index() -> dict:
    """
    Compact forensic index of the entire run: all agents, sessions, scores,
    artifacts, influences, and tool usage. Good starting point for forensic work.
    """
    store = _get_store()
    if store is None:
        return _no_traces()
    return {"index": store.build_index()}


@mcp.tool()
def traces_influences(agent_id: str | None = None) -> dict:
    """
    Cross-agent influence edges: when one agent reads what another wrote.
    Optionally filter to influences involving a specific agent.

    Args:
        agent_id: Optional filter (e.g. "agent0")
    """
    store = _get_store()
    if store is None:
        return _no_traces()

    infs = store.get_influences(agent_id)
    edges = []
    for i in infs:
        src = store._agent_for_trace(i["source_trace_id"])
        tgt = store._agent_for_trace(i["target_trace_id"])
        edges.append({
            "source_agent": src,
            "target_agent": tgt,
            "artifact_id": i["artifact_id"],
            "influence_type": i["influence_type"],
        })

    return {"filter": agent_id, "count": len(edges), "edges": edges}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()

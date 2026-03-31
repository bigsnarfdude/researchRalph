#!/usr/bin/env python3
"""
ResearchRalph MCP Server — RRMA run control as Claude Code tools.

Read-only tools for inspecting domains, artifacts, results, and run status.
Write tools (start/stop) to be added in v2.

Usage (standalone):
    RRMA_ROOT=/path/to/researchRalph python3 tools/rrma_mcp.py

Usage (via Claude Code MCP config):
    Configured in .rrma/mcp.json — Claude Code launches this automatically.

Requires:
    pip install "mcp[cli]"
"""

import csv
import hashlib
import io
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Paths ────────────────────────────────────────────────────────────────────

RRMA_ROOT = Path(os.environ.get("RRMA_ROOT", Path(__file__).resolve().parent.parent))
DOMAINS_DIR = RRMA_ROOT / "domains"

print(f"RRMA MCP: root={RRMA_ROOT}, domains={DOMAINS_DIR}", file=sys.stderr)

if not DOMAINS_DIR.exists():
    print(f"RRMA MCP: domains directory not found at {DOMAINS_DIR}", file=sys.stderr)
    sys.exit(1)

# ── Helpers ──────────────────────────────────────────────────────────────────

# Artifact filenames to search for (in priority order per artifact type)
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
    "config": ["config.yaml", "prompt_config.yaml"],
}


def find_artifact(domain_dir: Path, artifact_type: str) -> Path | None:
    """Find an artifact file, trying multiple name variants."""
    names = ARTIFACT_NAMES.get(artifact_type, [f"{artifact_type}.md"])
    for name in names:
        p = domain_dir / name
        if p.exists():
            return p
    return None


def read_artifact(domain_dir: Path, artifact_type: str) -> dict:
    """Read an artifact file and return content + metadata."""
    p = find_artifact(domain_dir, artifact_type)
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


def parse_results_tsv(domain_dir: Path) -> list[dict]:
    """Parse results.tsv into a list of experiment dicts."""
    p = find_artifact(domain_dir, "results")
    if p is None:
        return []
    text = p.read_text(errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = []
    for row in reader:
        # Normalize score to float
        if "score" in row:
            try:
                row["score"] = float(row["score"])
            except (ValueError, TypeError):
                pass
        rows.append(dict(row))
    return rows


def parse_config(domain_dir: Path) -> dict:
    """Read config.yaml (simple key: value parsing, no pyyaml dependency)."""
    p = find_artifact(domain_dir, "config")
    if p is None:
        return {"error": "No config.yaml or prompt_config.yaml found"}
    config = {"_file": p.name}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            config[key.strip()] = val.strip()
    return config


def get_screen_sessions() -> list[str]:
    """Get list of active screen session names."""
    try:
        out = subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True, timeout=5
        )
        sessions = []
        for line in out.stdout.splitlines():
            m = re.search(r"\d+\.(\S+)", line)
            if m:
                sessions.append(m.group(1))
        return sessions
    except Exception:
        return []


def domain_summary(domain_dir: Path) -> dict:
    """Build a quick summary for a domain."""
    config = parse_config(domain_dir)
    results = parse_results_tsv(domain_dir)

    scores = [r["score"] for r in results if isinstance(r.get("score"), float)]
    best = max(scores) if scores else None

    # Count log files
    logs_dir = domain_dir / "logs"
    log_count = len(list(logs_dir.glob("*.jsonl"))) if logs_dir.exists() else 0

    # Last modified artifact
    artifact_files = [
        f for f in domain_dir.iterdir()
        if f.suffix in (".md", ".tsv", ".jsonl") and f.name != "README.md"
    ]
    last_modified = None
    if artifact_files:
        newest = max(artifact_files, key=lambda f: f.stat().st_mtime)
        last_modified = datetime.fromtimestamp(newest.stat().st_mtime).isoformat()

    return {
        "domain": config.get("domain", domain_dir.name),
        "metric": config.get("metric", "unknown"),
        "direction": config.get("direction", "unknown"),
        "experiment_count": len(results),
        "best_score": best,
        "log_files": log_count,
        "last_modified": last_modified,
    }


# ── FastMCP Server ───────────────────────────────────────────────────────────

mcp = FastMCP("researchralph", json_response=True)


@mcp.tool()
def rrma_domains() -> dict:
    """
    List all available RRMA domains with basic metadata: name, metric,
    experiment count, best score, log file count, and last modified time.
    Use this to see what domains exist before drilling into one.
    """
    domains = []
    for d in sorted(DOMAINS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Skip template and hidden dirs
        if d.name.startswith(".") or d.name == "template":
            continue
        # Must have at least a config or program to be a real domain
        if not (find_artifact(d, "config") or find_artifact(d, "program")):
            continue
        domains.append(domain_summary(d))
    return {"count": len(domains), "domains": domains}


@mcp.tool()
def rrma_config(domain: str) -> dict:
    """
    Get the full configuration for a domain. Returns config.yaml contents
    plus available artifact files.

    Args:
        domain: Domain name (e.g. "rrma-lean", "trustloop-test")
    """
    domain_dir = DOMAINS_DIR / domain
    if not domain_dir.exists():
        return {
            "error": f"Domain '{domain}' not found",
            "available": [d.name for d in sorted(DOMAINS_DIR.iterdir()) if d.is_dir()],
        }

    config = parse_config(domain_dir)

    # List available artifacts
    available = []
    for art_type in ARTIFACT_NAMES:
        if find_artifact(domain_dir, art_type):
            available.append(art_type)

    # Check for run.sh
    has_harness = (domain_dir / "run.sh").exists()

    return {
        "domain": domain,
        "config": config,
        "available_artifacts": available,
        "has_harness": has_harness,
        "has_logs": (domain_dir / "logs").exists(),
    }


@mcp.tool()
def rrma_status(domain: str | None = None) -> dict:
    """
    Check if an RRMA run is active. Detects running screen sessions
    (agent0, agent1, ..., meta-agent, gardener). If a domain is specified,
    also returns its latest scores and experiment count.

    Args:
        domain: Optional domain name to include domain-specific status
    """
    sessions = get_screen_sessions()

    # Categorize sessions
    agent_sessions = [s for s in sessions if re.match(r"agent\d+$", s)]
    meta_sessions = [s for s in sessions if "meta" in s.lower()]
    gardener_sessions = [s for s in sessions if "gardener" in s.lower()]
    other_rrma = [s for s in sessions if s.startswith(("monitor", "diagnose", "score"))]

    status = {
        "running": len(agent_sessions) > 0,
        "agent_sessions": sorted(agent_sessions),
        "meta_sessions": meta_sessions,
        "gardener_sessions": gardener_sessions,
        "other_sessions": other_rrma,
        "all_screen_sessions": sorted(sessions),
    }

    if domain:
        domain_dir = DOMAINS_DIR / domain
        if domain_dir.exists():
            results = parse_results_tsv(domain_dir)
            scores = [r["score"] for r in results if isinstance(r.get("score"), float)]
            status["domain"] = domain
            status["experiment_count"] = len(results)
            status["best_score"] = max(scores) if scores else None
            status["latest_experiments"] = results[-5:] if results else []
        else:
            status["domain_error"] = f"Domain '{domain}' not found"

    return status


@mcp.tool()
def rrma_results(domain: str, last_n: int = 0) -> dict:
    """
    Get experiment results from results.tsv. Returns all experiments with
    scores, agents, descriptions, and status.

    Args:
        domain: Domain name (e.g. "rrma-lean")
        last_n: If > 0, return only the last N experiments (default: all)
    """
    domain_dir = DOMAINS_DIR / domain
    if not domain_dir.exists():
        return {"error": f"Domain '{domain}' not found"}

    results = parse_results_tsv(domain_dir)
    if not results:
        return {"domain": domain, "count": 0, "results": [], "note": "No results.tsv or empty"}

    scores = [r["score"] for r in results if isinstance(r.get("score"), float)]

    # Agent breakdown
    agents = {}
    for r in results:
        a = r.get("agent", "unknown")
        if a not in agents:
            agents[a] = {"count": 0, "scores": []}
        agents[a]["count"] += 1
        if isinstance(r.get("score"), float):
            agents[a]["scores"].append(r["score"])
    for a in agents:
        s = agents[a]["scores"]
        agents[a]["best"] = max(s) if s else None
        agents[a]["avg"] = round(sum(s) / len(s), 4) if s else None
        del agents[a]["scores"]

    output_results = results[-last_n:] if last_n > 0 else results

    return {
        "domain": domain,
        "count": len(results),
        "best_score": max(scores) if scores else None,
        "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
        "agents": agents,
        "results": output_results,
    }


@mcp.tool()
def rrma_artifact(domain: str, artifact_type: str) -> dict:
    """
    Read a shared artifact from a domain. Artifacts are the coordination
    layer between agents during a run.

    Args:
        domain: Domain name (e.g. "rrma-lean")
        artifact_type: One of: blackboard, meta_blackboard, program, results,
                       experiments, desires, learnings, mistakes, calibration, config
    """
    domain_dir = DOMAINS_DIR / domain
    if not domain_dir.exists():
        return {"error": f"Domain '{domain}' not found"}

    art = read_artifact(domain_dir, artifact_type)
    if "error" in art:
        # List what is available
        available = [at for at in ARTIFACT_NAMES if find_artifact(domain_dir, at)]
        art["available"] = available
    return art


@mcp.tool()
def rrma_artifacts_summary(domain: str) -> dict:
    """
    Get a summary of all artifacts in a domain without full content.
    Shows which artifacts exist, their sizes, line counts, and last
    modified times. Use this before fetching individual artifacts.

    Args:
        domain: Domain name (e.g. "rrma-lean")
    """
    domain_dir = DOMAINS_DIR / domain
    if not domain_dir.exists():
        return {"error": f"Domain '{domain}' not found"}

    artifacts = []
    for art_type in ARTIFACT_NAMES:
        p = find_artifact(domain_dir, art_type)
        if p is None:
            continue
        stat = p.stat()
        content = p.read_text(errors="replace")
        artifacts.append({
            "artifact_type": art_type,
            "filename": p.name,
            "lines": content.count("\n"),
            "bytes": stat.st_size,
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:12],
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return {
        "domain": domain,
        "count": len(artifacts),
        "artifacts": artifacts,
    }


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()

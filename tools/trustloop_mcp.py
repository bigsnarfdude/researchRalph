#!/usr/bin/env python3
"""
TrustLoop MCP Server — exposes TraceStore as Claude Code tools.

Wraps the existing TraceStore from trace_forensics.py as FastMCP tools.
Each tool maps to a CLI command in the trustloop script.

Usage (standalone):
    python3 tools/trustloop_mcp.py                    # uses /tmp/rrma_traces.jsonl
    TRUSTLOOP_TRACES=/path/to.jsonl python3 tools/trustloop_mcp.py

Usage (via Claude Code MCP config):
    Configured in .mcp.json — Claude Code launches this automatically.

Requires:
    pip install "mcp[cli]"
"""

import os
import sys
from collections import Counter
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Load TraceStore ───────────────────────────────────────────────────────────

# Add tools dir to path so we can import trace_forensics
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from trace_forensics import TraceStore

TRACES_PATH = os.environ.get("TRUSTLOOP_TRACES", "/tmp/rrma_traces.jsonl")

if not Path(TRACES_PATH).exists():
    print(
        f"TrustLoop: no traces at {TRACES_PATH}\n"
        f"Set TRUSTLOOP_TRACES or run: ./trustloop ingest <domain>",
        file=sys.stderr,
    )
    sys.exit(1)

store = TraceStore(TRACES_PATH)
print(
    f"TrustLoop: loaded {len(store.traces)} traces, "
    f"{len(store.traces_by_agent)} agents from {TRACES_PATH}",
    file=sys.stderr,
)

# ── FastMCP Server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "trustloop",
    json_response=True,
)


@mcp.tool()
def trustloop_status() -> dict:
    """
    Get an overview of the loaded RRMA run: domain, agent count, session count,
    total steps, thinking blocks, tool calls, experiments, best score, artifacts,
    and cross-agent influences. Start here to understand the run shape.
    """
    s = store.swarm
    agents = []
    for aid in sorted(store.traces_by_agent):
        traces = store.traces_by_agent[aid]
        steps = sum(t["metrics"]["total_steps"] for t in traces)
        thinking = sum(
            sum(1 for st in t["steps"] if st.get("reasoning_content"))
            for t in traces
        )
        agents.append({
            "agent_id": aid,
            "sessions": len(traces),
            "steps": steps,
            "thinking_blocks": thinking,
        })

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
        "artifact_count": len(s.get("artifacts", [])),
        "influence_count": len(s.get("influences", [])),
        "agents": agents,
    }


@mcp.tool()
def trustloop_agent(
    agent_id: str,
    mode: str = "summary",
    max_results: int = 20,
) -> dict:
    """
    Investigate a specific agent. Three modes:
    - "summary": session count, steps, tool calls, artifact reads/writes
    - "thinking": last N thinking blocks with artifact context and tool calls
    - "timeline": key events — artifact writes, harness scores, notable tool calls

    Args:
        agent_id: The agent to investigate (e.g. "agent0", "agent3")
        mode: One of "summary", "thinking", "timeline"
        max_results: Max thinking blocks or timeline events to return (default 20)
    """
    if agent_id not in store.traces_by_agent:
        return {
            "error": f"Unknown agent: {agent_id}",
            "available": sorted(store.traces_by_agent.keys()),
        }

    traces = store.traces_by_agent[agent_id]

    if mode == "thinking":
        blocks = store.get_agent_thinking(agent_id, max_results)
        return {
            "agent_id": agent_id,
            "mode": "thinking",
            "count": len(blocks),
            "blocks": blocks,
        }

    if mode == "timeline":
        events = store.get_agent_timeline(agent_id)
        return {
            "agent_id": agent_id,
            "mode": "timeline",
            "count": len(events),
            "events": events[:max_results],
        }

    # Summary mode
    total_steps = sum(t["metrics"]["total_steps"] for t in traces)
    thinking = sum(
        sum(1 for st in t["steps"] if st.get("reasoning_content"))
        for t in traces
    )
    tool_calls = sum(
        sum(len(st.get("tool_calls", [])) for st in t["steps"])
        for t in traces
    )

    reads = Counter()
    writes = Counter()
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
def trustloop_search(query: str, max_results: int = 15) -> dict:
    """
    Full-text search across all agent traces. Searches thinking blocks,
    visible output, tool call paths, and artifact names. Returns matching
    steps with agent ID, step index, which fields matched, and a snippet.

    Args:
        query: Search term (e.g. "omega", "run.sh", "blackboard")
        max_results: Maximum results to return (default 15)
    """
    results = store.search_traces(query, max_results)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@mcp.tool()
def trustloop_artifact(artifact_type: str) -> dict:
    """
    Get shared artifact content and metadata. Artifacts are the coordination
    layer between agents.

    Args:
        artifact_type: One of "blackboard", "program", "results", "desires",
                       "learnings", "mistakes"
    """
    art = store.get_artifact(artifact_type)
    if "error" in art:
        types = [a["artifact_type"] for a in store.swarm.get("artifacts", [])]
        art["available"] = types
    return art


@mcp.tool()
def trustloop_influences(agent_id: str | None = None) -> dict:
    """
    Get cross-agent influence edges. An influence is recorded when one agent
    reads an artifact that another agent wrote to, creating a causal
    dependency chain.

    Args:
        agent_id: Optional — filter to influences involving this agent
    """
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

    return {
        "filter": agent_id,
        "count": len(edges),
        "edges": edges,
    }


@mcp.tool()
def trustloop_step(trace_id: str, step_index: int) -> dict:
    """
    Get raw step data for a specific trace and step index. Use this when
    you need the exact reasoning content, tool calls, and artifact
    interactions for a particular moment in an agent's session.

    Args:
        trace_id: The trace ID (from trustloop_agent or trustloop_search results)
        step_index: The step number within the trace
    """
    step = store.get_step(trace_id, step_index)
    return step


@mcp.tool()
def trustloop_index() -> dict:
    """
    Get the compact forensic index of the entire run. This is a text summary
    of all agents, sessions, scores, artifacts, and influences. Useful for
    building an initial understanding before drilling into specifics.
    """
    return {"index": store.build_index()}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()

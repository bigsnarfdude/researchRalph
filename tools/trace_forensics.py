#!/usr/bin/env python3
"""
trace_forensics.py — Forensic analysis engine for RRMA trace data.

Loads SwarmRecord + TraceRecords from JSONL, builds a compact index,
and answers forensic questions via Claude tool-use loop.

Usage:
    # Interactive forensic chat
    python3 tools/trace_forensics.py /tmp/rrma_traces_test.jsonl

    # Single question
    python3 tools/trace_forensics.py /tmp/rrma_traces_test.jsonl \
        --ask "which agent first discovered that ring works for algebra?"

    # Build index only (dump to stdout)
    python3 tools/trace_forensics.py /tmp/rrma_traces_test.jsonl --index-only

Requires: pip install anthropic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ═══════════════════════════════════════════════════════════════════════════════
# Trace store — loads JSONL, builds index, answers queries
# ═══════════════════════════════════════════════════════════════════════════════

class TraceStore:
    """In-memory store for SwarmRecord + TraceRecords with query methods."""

    def __init__(self, jsonl_path: str | Path):
        self.swarm: dict = {}
        self.traces: list[dict] = []
        self.traces_by_id: dict[str, dict] = {}
        self.traces_by_agent: dict[str, list[dict]] = defaultdict(list)

        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("record_type") == "swarm":
                    self.swarm = rec
                else:
                    self.traces.append(rec)
                    self.traces_by_id[rec["trace_id"]] = rec
                    agent_id = rec.get("metadata", {}).get("agent_id", "unknown")
                    self.traces_by_agent[agent_id].append(rec)

    # ── Index builder ─────────────────────────────────────────────────────

    def build_index(self) -> str:
        """Build a compact text index (~3-4K tokens) for system prompt injection."""
        s = self.swarm
        lines = [
            f"# RRMA Swarm Forensic Index",
            f"Domain: {s.get('domain', '?')}",
            f"Schema: {s.get('schema_version', '?')}",
            f"Traces: {len(self.traces)} across {len(self.traces_by_agent)} agents",
            f"Experiments: {s.get('total_experiments', 0)}",
            f"Best score: {s.get('best_score', '?')} ({s.get('score_direction', '?')})",
            f"Influences: {len(s.get('influences', []))}",
            "",
            "## Agents",
        ]

        for agent_id in sorted(self.traces_by_agent):
            agent_traces = self.traces_by_agent[agent_id]
            total_steps = sum(t["metrics"]["total_steps"] for t in agent_traces)
            total_thinking = sum(
                sum(1 for st in t["steps"] if st.get("reasoning_content"))
                for t in agent_traces
            )
            total_tools = sum(
                sum(len(st.get("tool_calls", [])) for st in t["steps"])
                for t in agent_traces
            )
            # Artifact summary
            reads = Counter()
            writes = Counter()
            for t in agent_traces:
                for st in t["steps"]:
                    for r in st.get("artifact_reads", []):
                        reads[r] += 1
                    for w in st.get("artifact_writes", []):
                        writes[w] += 1

            trace_ids = [t["trace_id"][:8] for t in agent_traces]
            lines.append(
                f"- **{agent_id}**: {len(agent_traces)} sessions, "
                f"{total_steps} steps, {total_thinking} thinking, {total_tools} tool calls"
            )
            if reads:
                lines.append(f"  reads: {dict(reads.most_common(5))}")
            if writes:
                lines.append(f"  writes: {dict(writes.most_common(5))}")
            lines.append(f"  trace_ids: {trace_ids}")

        # Artifacts
        artifacts = s.get("artifacts", [])
        if artifacts:
            lines.append("")
            lines.append("## Shared Artifacts")
            for a in artifacts:
                snap = a["snapshots"][0] if a.get("snapshots") else {}
                lines.append(
                    f"- {a['artifact_type']}: {snap.get('line_count', 0)} lines, "
                    f"path={a.get('path', '?')}"
                )

        # Influence summary
        influences = s.get("influences", [])
        if influences:
            lines.append("")
            lines.append("## Cross-Agent Influences")
            # Group by source→target agent
            pairs = Counter()
            for inf in influences:
                src = self._agent_for_trace(inf["source_trace_id"])
                tgt = self._agent_for_trace(inf["target_trace_id"])
                pairs[(src, tgt)] += 1
            for (src, tgt), count in pairs.most_common(20):
                lines.append(f"- {src} → {tgt}: {count} ({influences[0].get('influence_type', '?')})")

        # Tool usage summary across all traces
        tool_counts = Counter()
        for t in self.traces:
            for st in t["steps"]:
                for tc in st.get("tool_calls", []):
                    tool_counts[tc["tool_name"]] += 1
        if tool_counts:
            lines.append("")
            lines.append("## Tool Usage (top 10)")
            for tool, count in tool_counts.most_common(10):
                lines.append(f"- {tool}: {count}")

        lines.append("")
        lines.append("## Available Query Tools")
        lines.append("Use these tools to drill into specific data:")
        lines.append("- get_step(trace_id, step_index) — full step detail")
        lines.append("- get_agent_thinking(agent_id) — all thinking blocks for an agent")
        lines.append("- get_artifact(artifact_type) — shared artifact content")
        lines.append("- search_traces(query) — search across all text/thinking content")
        lines.append("- get_agent_timeline(agent_id) — chronological summary of actions")
        lines.append("- get_influences(agent_id) — cross-agent influence chains")

        return "\n".join(lines)

    def _agent_for_trace(self, trace_id: str) -> str:
        t = self.traces_by_id.get(trace_id)
        if t:
            return t.get("metadata", {}).get("agent_id", trace_id[:8])
        # Try partial match
        for tid, t in self.traces_by_id.items():
            if tid.startswith(trace_id[:8]):
                return t.get("metadata", {}).get("agent_id", trace_id[:8])
        return trace_id[:8]

    # ── Query methods (exposed as Claude tools) ───────────────────────────

    def get_step(self, trace_id: str, step_index: int) -> dict:
        """Get a specific step from a trace."""
        trace = self._find_trace(trace_id)
        if not trace:
            return {"error": f"trace {trace_id} not found"}
        for step in trace["steps"]:
            if step["step_index"] == step_index:
                # Truncate large fields
                result = dict(step)
                if result.get("reasoning_content") and len(result["reasoning_content"]) > 4000:
                    result["reasoning_content"] = result["reasoning_content"][:4000] + "...[truncated]"
                for obs in result.get("observations", []):
                    if len(obs.get("content", "")) > 2000:
                        obs["content"] = obs["content"][:2000] + "...[truncated]"
                return result
        return {"error": f"step {step_index} not found in trace {trace_id}"}

    def get_agent_thinking(self, agent_id: str, max_blocks: int = 20) -> list[dict]:
        """Get thinking blocks for an agent, sorted by step index."""
        blocks = []
        for trace in self.traces_by_agent.get(agent_id, []):
            tid = trace["trace_id"][:8]
            for step in trace["steps"]:
                if step.get("reasoning_content"):
                    text = step["reasoning_content"]
                    if len(text) > 1500:
                        text = text[:1500] + "...[truncated]"
                    blocks.append({
                        "trace_id": tid,
                        "step_index": step["step_index"],
                        "thinking": text,
                        "tool_calls": [tc["tool_name"] for tc in step.get("tool_calls", [])],
                        "artifact_reads": step.get("artifact_reads", []),
                        "artifact_writes": step.get("artifact_writes", []),
                    })
        # Return most recent blocks (usually most interesting)
        return blocks[-max_blocks:]

    def get_artifact(self, artifact_type: str) -> dict:
        """Get a shared artifact's content."""
        for a in self.swarm.get("artifacts", []):
            if a["artifact_type"] == artifact_type:
                snap = a["snapshots"][0] if a.get("snapshots") else {}
                content = snap.get("content", "")
                if content and len(content) > 8000:
                    content = content[:8000] + "\n...[truncated]"
                return {
                    "artifact_type": a["artifact_type"],
                    "path": a.get("path", ""),
                    "line_count": snap.get("line_count", 0),
                    "content_hash": snap.get("content_hash", ""),
                    "content": content or "(content not stored inline — too large)",
                }
        return {"error": f"artifact {artifact_type} not found"}

    def search_traces(self, query: str, max_results: int = 15) -> list[dict]:
        """Search across all thinking + text content."""
        query_lower = query.lower()
        results = []
        for trace in self.traces:
            agent_id = trace.get("metadata", {}).get("agent_id", "?")
            tid = trace["trace_id"][:8]
            for step in trace["steps"]:
                found_in = []
                thinking = step.get("reasoning_content", "") or ""
                text = step.get("content", "") or ""

                if query_lower in thinking.lower():
                    found_in.append("thinking")
                if query_lower in text.lower():
                    found_in.append("content")
                # Search tool call inputs
                for tc in step.get("tool_calls", []):
                    inp_str = json.dumps(tc.get("input", {}))
                    if query_lower in inp_str.lower():
                        found_in.append(f"tool:{tc['tool_name']}")
                # Search observations
                for obs in step.get("observations", []):
                    if query_lower in (obs.get("content", "") or "").lower():
                        found_in.append("observation")

                if found_in:
                    # Extract context snippet from all matched sources
                    obs_text = "\n".join(
                        obs.get("content", "") or ""
                        for obs in step.get("observations", [])
                    )
                    full_text = thinking + "\n" + text + "\n" + obs_text
                    snippet = ""
                    idx = full_text.lower().find(query_lower)
                    if idx >= 0:
                        start = max(0, idx - 100)
                        end = min(len(full_text), idx + len(query) + 200)
                        snippet = full_text[start:end]

                    results.append({
                        "agent_id": agent_id,
                        "trace_id": tid,
                        "step_index": step["step_index"],
                        "found_in": found_in,
                        "snippet": snippet,
                    })
                    if len(results) >= max_results:
                        return results
        return results

    def get_agent_timeline(self, agent_id: str) -> list[dict]:
        """Get a chronological summary of an agent's key actions."""
        events = []
        for trace in self.traces_by_agent.get(agent_id, []):
            tid = trace["trace_id"][:8]
            for step in trace["steps"]:
                # Only include "interesting" steps
                has_thinking = bool(step.get("reasoning_content"))
                has_writes = bool(step.get("artifact_writes"))
                tool_names = [tc["tool_name"] for tc in step.get("tool_calls", [])]
                has_bash = "Bash" in tool_names
                has_write_tool = "Write" in tool_names

                if not (has_thinking or has_writes or has_write_tool):
                    continue

                summary = ""
                if has_thinking:
                    text = step["reasoning_content"][:200]
                    summary = text.replace("\n", " ")
                elif step.get("content"):
                    summary = step["content"][:200].replace("\n", " ")

                events.append({
                    "trace_id": tid,
                    "step_index": step["step_index"],
                    "tools": tool_names[:5],
                    "artifact_reads": step.get("artifact_reads", []),
                    "artifact_writes": step.get("artifact_writes", []),
                    "summary": summary,
                })

        # Limit to 50 most important events
        if len(events) > 50:
            # Prioritize: writes > thinking > reads
            events.sort(key=lambda e: (
                len(e["artifact_writes"]) * 3 +
                (1 if e["summary"] else 0) * 2 +
                len(e["artifact_reads"])
            ), reverse=True)
            events = events[:50]
            events.sort(key=lambda e: (e["trace_id"], e["step_index"]))

        return events

    def get_influences(self, agent_id: str | None = None) -> list[dict]:
        """Get cross-agent influence chains, optionally filtered by agent."""
        influences = self.swarm.get("influences", [])
        if not agent_id:
            return influences[:30]

        result = []
        for inf in influences:
            src = self._agent_for_trace(inf["source_trace_id"])
            tgt = self._agent_for_trace(inf["target_trace_id"])
            if agent_id in (src, tgt):
                result.append({
                    **inf,
                    "source_agent": src,
                    "target_agent": tgt,
                })
        return result[:30]

    def _find_trace(self, trace_id: str) -> dict | None:
        """Find trace by full or partial ID."""
        if trace_id in self.traces_by_id:
            return self.traces_by_id[trace_id]
        for tid, t in self.traces_by_id.items():
            if tid.startswith(trace_id):
                return t
        # Try by agent_id
        if trace_id in self.traces_by_agent:
            return self.traces_by_agent[trace_id][0] if self.traces_by_agent[trace_id] else None
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Claude tool-use forensic loop
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "get_step",
        "description": "Get the full detail of a specific step from a trace. Use this to inspect what an agent did at a particular moment — tool calls, thinking, observations, artifact access.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string", "description": "Full or partial (first 8 chars) trace ID"},
                "step_index": {"type": "integer", "description": "Step index within the trace"},
            },
            "required": ["trace_id", "step_index"],
        },
    },
    {
        "name": "get_agent_thinking",
        "description": "Get all thinking/reasoning blocks for a specific agent. Returns the agent's internal reasoning chain across all sessions. Use this to understand WHY an agent made decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent identifier (e.g. agent0, agent3, meta)"},
                "max_blocks": {"type": "integer", "description": "Max thinking blocks to return (default 20)", "default": 20},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "get_artifact",
        "description": "Get the content of a shared artifact (blackboard, meta_blackboard, results, desires, learnings, mistakes, program). These are the coordination surfaces agents read and write.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "enum": ["blackboard", "meta_blackboard", "results", "desires", "learnings", "mistakes", "program"],
                },
            },
            "required": ["artifact_type"],
        },
    },
    {
        "name": "search_traces",
        "description": "Full-text search across all agent thinking, text, tool calls, and observations. Use this to find when/where a concept, tactic, or problem was first mentioned or discussed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (case-insensitive substring match)"},
                "max_results": {"type": "integer", "description": "Max results (default 15)", "default": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_agent_timeline",
        "description": "Get a chronological summary of an agent's key actions: thinking moments, artifact writes, file creates. Use this for an overview of what an agent did across its sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent identifier"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "get_influences",
        "description": "Get cross-agent influence chains — how information flowed between agents through shared artifacts (blackboard writes → reads). Optionally filter by agent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Optional: filter influences involving this agent"},
            },
        },
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are a forensic analyst for multi-agent AI research systems.

You have access to structured trace data from an RRMA (Recursive Research Multi-Agent) run.
The traces capture everything: agent thinking, tool calls, shared artifact access, and cross-agent influence.

Your job: answer questions about agent behavior with specificity. Cite trace IDs, step indices, and exact quotes from thinking blocks. When you find something interesting, dig deeper with follow-up tool calls.

Think like a detective: follow the evidence chain. If agent3 wrote something to the blackboard, find who read it and what they did with it.

{index}

When answering:
- Be specific: cite agent IDs, step numbers, exact quotes
- Follow influence chains: who wrote what, who read it, what changed
- Flag anomalies: unexpected behavior, stagnation, gaming, circular reasoning
- Distinguish correlation from causation: just because agent5 read the blackboard before succeeding doesn't mean the blackboard caused the success
"""


def forensic_query(store: TraceStore, question: str, verbose: bool = False) -> str:
    """Run a single forensic query through Claude tool-use loop."""
    if not HAS_ANTHROPIC:
        return "Error: pip install anthropic"

    client = anthropic.Anthropic()
    index = store.build_index()
    system = SYSTEM_PROMPT_TEMPLATE.format(index=index)

    messages = [{"role": "user", "content": question}]

    # Tool-use loop (max 10 rounds)
    for round_num in range(10):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Process response
        tool_results = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                if verbose:
                    print(f"  [tool] {block.name}({json.dumps(block.input)[:100]})", file=sys.stderr)

                # Dispatch tool call
                result = _dispatch_tool(store, block.name, block.input)
                result_str = json.dumps(result, indent=2, default=str)

                # Truncate very large results
                if len(result_str) > 12000:
                    result_str = result_str[:12000] + "\n...[truncated]"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        # If no tool calls, we're done
        if response.stop_reason == "end_turn":
            return "\n".join(text_parts)

        # Continue loop with tool results
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "\n".join(text_parts) if text_parts else "(max rounds reached)"


def _dispatch_tool(store: TraceStore, name: str, input_data: dict) -> Any:
    """Route a tool call to the appropriate TraceStore method."""
    if name == "get_step":
        return store.get_step(input_data["trace_id"], input_data["step_index"])
    elif name == "get_agent_thinking":
        return store.get_agent_thinking(
            input_data["agent_id"],
            input_data.get("max_blocks", 20),
        )
    elif name == "get_artifact":
        return store.get_artifact(input_data["artifact_type"])
    elif name == "search_traces":
        return store.search_traces(
            input_data["query"],
            input_data.get("max_results", 15),
        )
    elif name == "get_agent_timeline":
        return store.get_agent_timeline(input_data["agent_id"])
    elif name == "get_influences":
        return store.get_influences(input_data.get("agent_id"))
    else:
        return {"error": f"unknown tool: {name}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Web API (used by trustloop_server.py)
# ═══════════════════════════════════════════════════════════════════════════════

def handle_api_request(store: TraceStore, path: str, params: dict) -> dict:
    """Handle REST API requests from the web UI.

    Routes:
        /api/traces/index          — compact index
        /api/traces/agents         — agent list with stats
        /api/traces/step           — ?trace_id=X&step=N
        /api/traces/thinking       — ?agent_id=X
        /api/traces/artifact       — ?type=blackboard
        /api/traces/search         — ?q=ring+tactic
        /api/traces/timeline       — ?agent_id=X
        /api/traces/influences     — ?agent_id=X (optional)
        /api/traces/forensic       — POST {question: "..."}
    """
    if path == "/api/traces/index":
        return {"index": store.build_index()}

    elif path == "/api/traces/agents":
        agents = {}
        for agent_id, traces in store.traces_by_agent.items():
            agents[agent_id] = {
                "sessions": len(traces),
                "total_steps": sum(t["metrics"]["total_steps"] for t in traces),
                "trace_ids": [t["trace_id"][:8] for t in traces],
            }
        return {"agents": agents, "total_traces": len(store.traces)}

    elif path == "/api/traces/step":
        return store.get_step(
            params.get("trace_id", ""),
            int(params.get("step", "0")),
        )

    elif path == "/api/traces/thinking":
        return {"blocks": store.get_agent_thinking(
            params.get("agent_id", ""),
            int(params.get("max", "20")),
        )}

    elif path == "/api/traces/artifact":
        return store.get_artifact(params.get("type", ""))

    elif path == "/api/traces/search":
        return {"results": store.search_traces(
            params.get("q", ""),
            int(params.get("max", "15")),
        )}

    elif path == "/api/traces/timeline":
        return {"events": store.get_agent_timeline(params.get("agent_id", ""))}

    elif path == "/api/traces/influences":
        return {"influences": store.get_influences(params.get("agent_id"))}

    return {"error": f"unknown path: {path}"}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop(store: TraceStore):
    """Interactive forensic chat."""
    print("\n=== RRMA Trace Forensics ===")
    print(f"Loaded: {len(store.traces)} traces, {len(store.traces_by_agent)} agents")
    print(f"Type a question, or 'quit' to exit.\n")

    # Print compact overview
    for agent_id in sorted(store.traces_by_agent):
        traces = store.traces_by_agent[agent_id]
        steps = sum(t["metrics"]["total_steps"] for t in traces)
        print(f"  {agent_id}: {len(traces)} sessions, {steps} steps")
    print()

    while True:
        try:
            question = input("forensic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "index":
            print(store.build_index())
            continue

        print("  Analyzing...", file=sys.stderr)
        answer = forensic_query(store, question, verbose=True)
        print()
        print(answer)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Forensic analysis of RRMA trace data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("traces_jsonl", help="Path to traces JSONL (from rrma_traces.py)")
    parser.add_argument("--ask", default=None, help="Single question (non-interactive)")
    parser.add_argument("--index-only", action="store_true", help="Print index and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    store = TraceStore(args.traces_jsonl)

    if args.index_only:
        print(store.build_index())
        return

    if args.ask:
        answer = forensic_query(store, args.ask, verbose=args.verbose)
        print(answer)
        return

    interactive_loop(store)


if __name__ == "__main__":
    main()

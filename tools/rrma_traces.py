#!/usr/bin/env python3
"""
rrma_traces.py — OpenTraces schema + RRMA multi-agent extensions

Adopts the opentraces v0.1.0 base schema, adds 6 models and 4 Step fields
for multi-agent swarm coordination. Zero breaking changes to base schema.

Usage:
    # Convert a single RRMA domain run to traces
    python3 tools/rrma_traces.py domains/rrma-lean --logs data/rrma_lean_logs

    # Convert with all artifacts
    python3 tools/rrma_traces.py domains/rrma-r1 --logs domains/rrma-r1/logs

    # Dump as JSONL
    python3 tools/rrma_traces.py domains/rrma-lean --logs data/rrma_lean_logs --out traces.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "0.2.0-rrma"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenTraces v0.1.0 base schema (adopted as-is)
# ═══════════════════════════════════════════════════════════════════════════════

class Task(BaseModel):
    description: str = ""
    source: str = ""
    repository: str | None = None
    base_commit: str | None = None


class Agent(BaseModel):
    name: str = "claude-code"
    version: str | None = None
    model: str | None = None


class VCS(BaseModel):
    type: str = "git"
    base_commit: str | None = None
    branch: str | None = None
    diff: str | None = None


class Environment(BaseModel):
    os: str | None = None
    shell: str = "bash"
    vcs: VCS = Field(default_factory=VCS)
    language_ecosystem: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    tool_call_id: str = ""
    tool_name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    input_text: str | None = None


class Observation(BaseModel):
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


class Snippet(BaseModel):
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    content: str = ""


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class Outcome(BaseModel):
    success: bool | None = None
    signal_source: str | None = None
    signal_confidence: str | None = None
    description: str = ""
    patch: str | None = None
    committed: bool | None = None
    commit_sha: str | None = None


class Metrics(BaseModel):
    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_s: float | None = None
    cache_hit_rate: float | None = None
    estimated_cost_usd: float | None = None


class SecurityMetadata(BaseModel):
    scanned: bool = False
    patterns_matched: list[str] = Field(default_factory=list)
    entropy_flags: list[str] = Field(default_factory=list)
    paths_anonymized: bool = False


class Attribution(BaseModel):
    authors: list[str] = Field(default_factory=list)
    license: str | None = None


class Step(BaseModel):
    """One API call/response in a trace. Base opentraces + RRMA extensions."""

    # ── opentraces base ──
    step_index: int = 0
    role: Literal["system", "user", "agent"] = "agent"
    content: str | None = None
    reasoning_content: str | None = None
    model: str | None = None
    system_prompt_hash: str | None = None
    agent_role: str | None = None
    parent_step: int | None = None
    call_type: str | None = None
    subagent_trajectory_ref: str | None = None
    tools_available: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    snippets: list[Snippet] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    timestamp: str | None = None

    # ── RRMA multi-agent extensions (all optional, backward-compatible) ──
    swarm_agent_id: str | None = Field(
        None, description="Agent's local identity: agent0, agent1, meta, gardener"
    )
    artifact_reads: list[str] = Field(
        default_factory=list, description="artifact_ids read during this step"
    )
    artifact_writes: list[str] = Field(
        default_factory=list, description="artifact_ids written during this step"
    )
    experiment_result: ExperimentResult | None = Field(
        None, description="Scored experiment produced by this step"
    )


class TraceRecord(BaseModel):
    """One agent session. Opentraces v0.1.0 compatible."""

    schema_version: str = SCHEMA_VERSION
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    content_hash: str | None = None
    timestamp_start: str | None = None
    timestamp_end: str | None = None

    task: Task = Field(default_factory=Task)
    agent: Agent = Field(default_factory=Agent)
    environment: Environment = Field(default_factory=Environment)
    system_prompts: dict[str, str] = Field(default_factory=dict)
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    dependencies: list[str] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    security: SecurityMetadata = Field(default_factory=SecurityMetadata)
    attribution: Attribution = Field(default_factory=Attribution)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# RRMA multi-agent extensions (6 new models)
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentResult(BaseModel):
    """A single scored experiment within a session.
    RRMA agents produce many experiments per session."""

    experiment_id: str
    score: float
    status: Literal["keep", "discard", "crash", "retest"] = "keep"
    description: str | None = None
    design: str | None = None
    agent_id: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    memory_gb: float | None = None


class ArtifactSnapshot(BaseModel):
    """Point-in-time snapshot of a shared artifact."""

    timestamp: str
    content_hash: str
    line_count: int = 0
    trigger: str = ""  # meta_compression, generation_end, nudge_append, etc.
    content: str | None = None
    delta: str | None = None


class SharedArtifact(BaseModel):
    """Mutable artifact shared across agents in a swarm."""

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: Literal[
        "blackboard", "meta_blackboard", "results",
        "desires", "learnings", "mistakes",
        "program", "calibration",
    ]
    path: str = ""
    snapshots: list[ArtifactSnapshot] = Field(default_factory=list)


class SupervisoryDecision(BaseModel):
    """Control decision from outer-loop supervisor (gardener)."""

    timestamp: str
    decision: Literal[
        "CONTINUE", "TOO_EARLY", "NUDGE",
        "STOP_HACKING", "STOP_DONE", "REDESIGN",
    ]
    process_quality: int = Field(ge=0, le=30)
    generation: int = 0
    total_experiments: int = 0
    best_score: float | None = None
    stagnation_depth: int = 0
    flat: bool = False
    blind_spots: int = 0
    scaffold_desires: int = 0
    pq_signals: dict[str, int] = Field(default_factory=dict)
    action: str | None = None
    action_content: str | None = None


class CrossAgentInfluence(BaseModel):
    """Causal influence between agents via shared artifacts.
    Not parent-child — broadcast influence through shared state."""

    source_trace_id: str
    source_step_index: int
    target_trace_id: str
    target_step_index: int
    artifact_id: str
    influence_type: Literal[
        "finding", "request", "nudge",
        "compression", "scaffold_rewrite",
    ]
    confidence: Literal["high", "medium", "low"] = "medium"


class ScaffoldVersion(BaseModel):
    """A version of program.md (agent instructions)."""

    generation: int
    timestamp: str
    content_hash: str
    trigger: str = ""  # initial, gardener_rewrite, stop_hacking
    content: str | None = None


class SwarmRecord(BaseModel):
    """Wraps multiple TraceRecords from a coordinated multi-agent run.
    This is the top-level RRMA extension — one per generation."""

    schema_version: str = SCHEMA_VERSION
    swarm_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    swarm_type: str = "rrma_v4"
    domain: str = ""
    generation: int = 1
    generation_total: int | None = None

    timestamp_start: str | None = None
    timestamp_end: str | None = None

    # Agent traces (references, not embedded)
    traces: list[str] = Field(
        default_factory=list,
        description="trace_ids of member TraceRecords"
    )
    roles: dict[str, str] = Field(
        default_factory=dict,
        description="trace_id -> role: worker|meta|gardener|conductor"
    )

    # Shared artifacts
    artifacts: list[SharedArtifact] = Field(default_factory=list)

    # Supervisory decisions
    decisions: list[SupervisoryDecision] = Field(default_factory=list)

    # Cross-agent influence
    influences: list[CrossAgentInfluence] = Field(default_factory=list)

    # Aggregate metrics
    total_experiments: int = 0
    best_score: float | None = None
    score_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    process_quality: int | None = None

    # Scaffold versioning
    scaffold_versions: list[ScaffoldVersion] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Converter: RRMA logs → TraceRecord + SwarmRecord
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_artifact_access(tool_name: str, tool_input: dict) -> tuple[list[str], list[str]]:
    """Detect which shared artifacts a tool call reads or writes."""
    reads, writes = [], []
    path = tool_input.get("file_path", "") or tool_input.get("command", "")

    artifact_patterns = {
        "blackboard.md": "blackboard",
        "meta-blackboard.md": "meta_blackboard",
        "results.tsv": "results",
        "DESIRES.md": "desires",
        "LEARNINGS.md": "learnings",
        "MISTAKES.md": "mistakes",
        "program.md": "program",
    }

    for pattern, artifact_type in artifact_patterns.items():
        if pattern in path:
            if tool_name in ("Read", "Glob", "Grep") or "cat " in path or "grep " in path:
                reads.append(artifact_type)
            elif tool_name in ("Write", "Edit") or "echo " in path or "cat >" in path or "tee " in path:
                writes.append(artifact_type)
            elif tool_name == "Bash":
                # Bash could be either — check for write indicators
                if any(w in path for w in [">", "tee ", "echo ", "cat <<", ">>", "write"]):
                    writes.append(artifact_type)
                else:
                    reads.append(artifact_type)

    return reads, writes


def _parse_experiment_from_results_line(line: str) -> ExperimentResult | None:
    """Parse a results.tsv line into an ExperimentResult."""
    parts = line.strip().split("\t")
    if len(parts) < 4 or parts[0].startswith("EXP-ID") or parts[0].startswith("#"):
        return None
    try:
        return ExperimentResult(
            experiment_id=parts[0].strip(),
            score=float(parts[1].strip()),
            status=parts[3].strip() if len(parts) > 3 else "keep",
            description=parts[4].strip() if len(parts) > 4 else None,
            agent_id=parts[5].strip() if len(parts) > 5 else None,
            design=parts[6].strip() if len(parts) > 6 else None,
        )
    except (ValueError, IndexError):
        return None


def parse_jsonl_to_trace(
    jsonl_path: Path,
    agent_id: str | None = None,
    domain: str | None = None,
) -> TraceRecord:
    """Convert one Claude Code JSONL log into a TraceRecord."""

    trace = TraceRecord(
        metadata={"source_file": str(jsonl_path)},
    )

    if agent_id:
        trace.metadata["agent_id"] = agent_id

    step_idx = 0
    all_text = []

    with open(jsonl_path) as f:
        for line_no, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")

            # System init
            if t == "system" and obj.get("subtype") == "init":
                trace.session_id = obj.get("session_id")
                trace.agent.model = obj.get("model")
                trace.environment.os = "linux"  # RRMA runs on nigel
                if obj.get("tools"):
                    trace.steps  # tools available captured on first step
                continue

            # Assistant message
            if t == "assistant":
                msg = obj.get("message", {})
                content_blocks = msg.get("content", [])

                thinking_parts = []
                text_parts = []
                tool_calls = []

                for block in content_blocks:
                    btype = block.get("type")
                    if btype == "thinking":
                        thinking_parts.append(block.get("thinking", ""))
                    elif btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        reads, writes = _detect_artifact_access(
                            block.get("name", ""),
                            block.get("input", {}),
                        )
                        tool_calls.append(ToolCall(
                            tool_call_id=block.get("id", ""),
                            tool_name=block.get("name", ""),
                            input=block.get("input", {}),
                        ))

                # Token usage
                usage = msg.get("usage", {})
                token_usage = TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                    cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                )

                # Collect all artifact reads/writes for this step
                all_reads, all_writes = [], []
                for tc in tool_calls:
                    r, w = _detect_artifact_access(tc.tool_name, tc.input)
                    all_reads.extend(r)
                    all_writes.extend(w)

                step = Step(
                    step_index=step_idx,
                    role="agent",
                    content="\n".join(text_parts) if text_parts else None,
                    reasoning_content="\n\n".join(thinking_parts) if thinking_parts else None,
                    model=msg.get("model"),
                    tool_calls=tool_calls,
                    token_usage=token_usage,
                    swarm_agent_id=agent_id,
                    artifact_reads=list(set(all_reads)),
                    artifact_writes=list(set(all_writes)),
                )

                trace.steps.append(step)
                if text_parts:
                    all_text.extend(text_parts)
                step_idx += 1

            # User message (tool results)
            elif t == "user":
                msg = obj.get("message", {})
                content_blocks = msg.get("content", [])

                obs = []
                for block in content_blocks if isinstance(content_blocks, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        text = block.get("content", "")
                        if isinstance(text, list):
                            text = "\n".join(
                                c.get("text", "") for c in text if isinstance(c, dict)
                            )
                        obs.append(Observation(
                            tool_call_id=block.get("tool_use_id", ""),
                            content=str(text)[:2000],  # truncate large results
                            is_error=block.get("is_error", False),
                        ))

                if obs:
                    step = Step(
                        step_index=step_idx,
                        role="user",
                        observations=obs,
                        swarm_agent_id=agent_id,
                    )
                    trace.steps.append(step)
                    step_idx += 1

    # Compute metrics
    total_in = sum(s.token_usage.input_tokens for s in trace.steps if s.token_usage)
    total_out = sum(s.token_usage.output_tokens for s in trace.steps if s.token_usage)
    cache_create = sum(s.token_usage.cache_creation_tokens for s in trace.steps if s.token_usage)
    cache_read = sum(s.token_usage.cache_read_tokens for s in trace.steps if s.token_usage)

    trace.metrics = Metrics(
        total_steps=len(trace.steps),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        cache_hit_rate=cache_read / max(cache_read + cache_create, 1),
    )

    # Content hash
    full_text = "\n".join(all_text)
    trace.content_hash = _sha256(full_text)

    if domain:
        trace.task.repository = domain
        trace.task.source = "rrma_v4_scaffold"

    return trace


def load_artifact(path: Path, artifact_type: str) -> SharedArtifact | None:
    """Load a shared artifact file as a SharedArtifact with one snapshot."""
    if not path.exists():
        return None

    content = path.read_text(errors="replace")
    if not content.strip():
        return None

    return SharedArtifact(
        artifact_type=artifact_type,
        path=str(path),
        snapshots=[ArtifactSnapshot(
            timestamp=_now_iso(),
            content_hash=_sha256(content),
            line_count=content.count("\n") + 1,
            trigger="snapshot",
            content=content if len(content) < 50_000 else None,
        )],
    )


def load_results_tsv(path: Path) -> list[ExperimentResult]:
    """Parse results.tsv into ExperimentResults."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text(errors="replace").splitlines():
        exp = _parse_experiment_from_results_line(line)
        if exp:
            results.append(exp)
    return results


def load_experiments_jsonl(path: Path) -> list[ExperimentResult]:
    """Load structured experiments from experiments.jsonl (preferred source)."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            results.append(ExperimentResult(
                experiment_id=obj["exp_id"],
                score=obj["score"],
                status=obj.get("status", "keep"),
                description=obj.get("description"),
                agent_id=obj.get("agent"),
                design=obj.get("design"),
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def extract_scores_from_trace(trace: TraceRecord) -> list[ExperimentResult]:
    """Fallback: extract SCORE= from tool call outputs in a trace.
    Used for existing runs that lack experiments.jsonl."""
    results = []
    agent_id = trace.metadata.get("agent_id")
    score_re = re.compile(r"SCORE=([0-9]+\.[0-9]+)")
    exp_id_re = re.compile(r"Logged:\s+(exp\d+|lam\d+)")

    for step in trace.steps:
        for obs in step.observations:
            m = score_re.search(obs.content)
            if m:
                score = float(m.group(1))
                # Try to find the experiment ID in the same output
                eid_match = exp_id_re.search(obs.content)
                exp_id = eid_match.group(1) if eid_match else f"extracted_{step.step_index}"
                results.append(ExperimentResult(
                    experiment_id=exp_id,
                    score=score,
                    status="keep" if score > 0 else "discard",
                    agent_id=agent_id,
                ))
    return results


def build_swarm_record(
    domain_dir: Path,
    logs_dir: Path | None = None,
) -> tuple[SwarmRecord, list[TraceRecord]]:
    """Build a SwarmRecord + TraceRecords from an RRMA domain directory."""

    domain_name = domain_dir.name
    swarm = SwarmRecord(domain=domain_name)

    # ── Load traces from JSONL logs ──
    traces = []
    if logs_dir is None:
        logs_dir = domain_dir / "logs"

    if logs_dir.exists():
        for jf in sorted(logs_dir.glob("*.jsonl")):
            # Extract agent_id from filename: agent1_20260328_095119.jsonl
            # or agent0_20260328_040045_20260328_051356.jsonl (concatenated sessions)
            name = jf.stem
            m = re.match(r"(agent\d+|meta|gardener)", name)
            agent_id = m.group(1) if m else name.split("_")[0]

            role = "worker"
            if "meta" in agent_id:
                role = "meta"
            elif "gardener" in agent_id:
                role = "gardener"

            trace = parse_jsonl_to_trace(jf, agent_id=agent_id, domain=domain_name)
            traces.append(trace)
            swarm.traces.append(trace.trace_id)
            swarm.roles[trace.trace_id] = role

    # ── Load shared artifacts ──
    artifact_files = {
        "blackboard.md": "blackboard",
        "meta-blackboard.md": "meta_blackboard",
        "results.tsv": "results",
        "DESIRES.md": "desires",
        "LEARNINGS.md": "learnings",
        "MISTAKES.md": "mistakes",
        "program.md": "program",
    }

    for filename, atype in artifact_files.items():
        artifact = load_artifact(domain_dir / filename, atype)
        if artifact:
            swarm.artifacts.append(artifact)

    # ── Load experiment results (prefer experiments.jsonl > results.tsv > trace extraction) ──
    experiments_jsonl = domain_dir / "experiments.jsonl"
    if experiments_jsonl.exists():
        experiments = load_experiments_jsonl(experiments_jsonl)
    else:
        experiments = load_results_tsv(domain_dir / "results.tsv")

    # Fallback: extract SCORE= from tool outputs if few experiments found
    if len(experiments) < 5:
        extracted = []
        seen_scores: set[tuple[str, float]] = {(e.agent_id or "", e.score) for e in experiments}
        for trace in traces:
            for exp in extract_scores_from_trace(trace):
                key = (exp.agent_id or "", exp.score)
                if key not in seen_scores:
                    extracted.append(exp)
                    seen_scores.add(key)
        if extracted:
            experiments.extend(extracted)

    # Link experiments to trace steps that produced them
    for trace in traces:
        agent_id = trace.metadata.get("agent_id")
        agent_exps = [e for e in experiments if e.agent_id == agent_id]
        # Attach experiments to steps that contain SCORE= in observations
        score_re = re.compile(r"SCORE=([0-9]+\.[0-9]+)")
        for step in trace.steps:
            for obs in step.observations:
                m = score_re.search(obs.content)
                if m:
                    score_val = float(m.group(1))
                    matching = [e for e in agent_exps if abs(e.score - score_val) < 0.0001]
                    if matching and step.experiment_result is None:
                        step.experiment_result = matching[0]
                        agent_exps.remove(matching[0])

    swarm.total_experiments = len(experiments)
    if experiments:
        scores = [e.score for e in experiments]
        if scores:
            # Read config.yaml for direction
            config_path = domain_dir / "config.yaml"
            if config_path.exists():
                config_text = config_path.read_text(errors="replace")
                if "lower_is_better" in config_text:
                    swarm.score_direction = "lower_is_better"
                    swarm.best_score = min(scores)
                else:
                    swarm.best_score = max(scores)
            else:
                swarm.best_score = max(scores)

    # ── Detect cross-agent influences (blackboard write → read) ──
    # Simple heuristic: if agent A writes to blackboard at step X,
    # and agent B reads blackboard at step Y (Y > X), that's influence.
    bb_writes = []  # (trace_id, step_index, agent_id)
    bb_reads = []

    for trace in traces:
        for step in trace.steps:
            if "blackboard" in step.artifact_writes:
                bb_writes.append((trace.trace_id, step.step_index, step.swarm_agent_id))
            if "blackboard" in step.artifact_reads:
                bb_reads.append((trace.trace_id, step.step_index, step.swarm_agent_id))

    # Deduplicate: one influence per (writer_agent, reader_agent) pair
    # Keep earliest write → earliest read
    seen_pairs: set[tuple[str, str]] = set()
    for w_trace, w_step, w_agent in bb_writes:
        for r_trace, r_step, r_agent in bb_reads:
            if w_trace == r_trace or w_agent == r_agent:
                continue
            pair = (w_agent or w_trace, r_agent or r_trace)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            swarm.influences.append(CrossAgentInfluence(
                source_trace_id=w_trace,
                source_step_index=w_step,
                target_trace_id=r_trace,
                target_step_index=r_step,
                artifact_id="blackboard",
                influence_type="finding",
                confidence="low",  # heuristic, not proven causal
            ))

    # ── Scaffold versioning ──
    program_path = domain_dir / "program.md"
    if program_path.exists():
        content = program_path.read_text(errors="replace")
        swarm.scaffold_versions.append(ScaffoldVersion(
            generation=1,
            timestamp=_now_iso(),
            content_hash=_sha256(content),
            trigger="current",
        ))

    return swarm, traces


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(swarm: SwarmRecord, traces: list[TraceRecord]):
    """Print a human-readable summary of the converted data."""

    print(f"\n{'='*60}")
    print(f"SwarmRecord: {swarm.swarm_id[:8]}...")
    print(f"{'='*60}")
    print(f"  Domain:       {swarm.domain}")
    print(f"  Schema:       {swarm.schema_version}")
    print(f"  Swarm type:   {swarm.swarm_type}")
    print(f"  Traces:       {len(traces)}")
    print(f"  Experiments:  {swarm.total_experiments}")
    print(f"  Best score:   {swarm.best_score}")
    print(f"  Direction:    {swarm.score_direction}")
    print(f"  Artifacts:    {len(swarm.artifacts)}")
    print(f"  Influences:   {len(swarm.influences)}")
    print(f"  Scaffolds:    {len(swarm.scaffold_versions)}")

    print(f"\n  Roles:")
    for tid, role in swarm.roles.items():
        print(f"    {tid[:8]}... → {role}")

    print(f"\n  Artifacts:")
    for a in swarm.artifacts:
        snap = a.snapshots[0] if a.snapshots else None
        lines = snap.line_count if snap else 0
        print(f"    {a.artifact_type:20s}  {lines:>5} lines  {a.path}")

    for trace in traces:
        agent_id = trace.metadata.get("agent_id", "?")
        n_steps = trace.metrics.total_steps
        n_thinking = sum(1 for s in trace.steps if s.reasoning_content)
        n_tool_calls = sum(len(s.tool_calls) for s in trace.steps)
        n_reads = sum(len(s.artifact_reads) for s in trace.steps)
        n_writes = sum(len(s.artifact_writes) for s in trace.steps)
        tokens_in = trace.metrics.total_input_tokens
        tokens_out = trace.metrics.total_output_tokens

        print(f"\n  TraceRecord: {trace.trace_id[:8]}... ({agent_id})")
        print(f"    Steps:          {n_steps}")
        print(f"    Thinking blocks:{n_thinking}")
        print(f"    Tool calls:     {n_tool_calls}")
        print(f"    Artifact reads: {n_reads}")
        print(f"    Artifact writes:{n_writes}")
        print(f"    Tokens:         {tokens_in:,} in / {tokens_out:,} out")
        print(f"    Cache hit rate: {trace.metrics.cache_hit_rate:.1%}" if trace.metrics.cache_hit_rate else "")
        print(f"    Model:          {trace.agent.model}")

    if swarm.influences:
        print(f"\n  Cross-Agent Influences ({len(swarm.influences)}):")
        # Summarize by type
        from collections import Counter
        types = Counter(i.influence_type for i in swarm.influences)
        for itype, count in types.most_common():
            print(f"    {itype}: {count}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert RRMA domain run to OpenTraces + RRMA extensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("domain_dir", help="Path to domain directory")
    parser.add_argument("--logs", default=None, help="Path to logs directory (default: domain_dir/logs)")
    parser.add_argument("--out", default=None, help="Output JSONL file (default: print summary)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)
    if not domain_dir.exists():
        print(f"Domain directory not found: {domain_dir}", file=sys.stderr)
        sys.exit(1)

    logs_dir = Path(args.logs) if args.logs else None

    swarm, traces = build_swarm_record(domain_dir, logs_dir)

    if args.out:
        out_path = Path(args.out)
        indent = 2 if args.pretty else None
        with open(out_path, "w") as f:
            # Write swarm record first
            f.write(json.dumps({"record_type": "swarm", **swarm.model_dump()}, default=str, indent=indent) + "\n")
            # Then each trace
            for trace in traces:
                f.write(json.dumps({"record_type": "trace", **trace.model_dump()}, default=str, indent=indent) + "\n")
        print(f"Wrote {1 + len(traces)} records to {out_path}")
        print(f"  1 SwarmRecord + {len(traces)} TraceRecords")
    else:
        print_summary(swarm, traces)


if __name__ == "__main__":
    main()

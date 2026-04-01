#!/usr/bin/env python3
"""
Agent Memory System — Scanner, Retriever, and Staleness Checker.

Three components for intelligent memory retrieval:
  1. Scanner: Parse frontmatter + mtime from topic files, produce a manifest
  2. Retriever: LLM side-query to pick top-N relevant files from manifest
  3. Staleness: Age-check memories, wrap old ones with verification warnings

Design: filesystem-native, no vector DB, no embeddings. Uses LLM reasoning
for semantic retrieval (better than cosine similarity on short descriptions).

Usage:
    # Scan memory directory, print manifest
    python3 tools/memory_system.py scan ~/.claude/projects/FOO/memory/

    # Retrieve top 5 relevant memories for a query
    python3 tools/memory_system.py retrieve ~/.claude/projects/FOO/memory/ "what's the cadenza proposal status"

    # Check staleness of all memories
    python3 tools/memory_system.py staleness ~/.claude/projects/FOO/memory/

    # Full pipeline: scan → retrieve → load with staleness warnings
    python3 tools/memory_system.py recall ~/.claude/projects/FOO/memory/ "query here"

    # JSON output for programmatic use
    python3 tools/memory_system.py recall ~/.claude/projects/FOO/memory/ "query" --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """A single memory file's metadata, extracted without reading full content."""
    path: Path
    name: str = ""
    description: str = ""
    type: str = ""  # user | feedback | project | reference
    mtime: float = 0.0
    age_days: float = 0.0
    size_bytes: int = 0
    line_count: int = 0

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def is_stale(self) -> bool:
        return self.age_days > 1.0

    @property
    def staleness_level(self) -> str:
        if self.age_days <= 1:
            return "fresh"
        elif self.age_days <= 7:
            return "recent"
        elif self.age_days <= 30:
            return "aging"
        else:
            return "stale"


@dataclass
class Manifest:
    """The scanner's output: a compact list of all memories with metadata."""
    entries: list[MemoryEntry] = field(default_factory=list)
    scanned_at: str = ""
    memory_dir: str = ""

    @property
    def count(self) -> int:
        return len(self.entries)

    def to_prompt(self) -> str:
        """Format manifest for LLM consumption (compact, one line per file)."""
        lines = []
        for e in self.entries:
            age_tag = f"[{e.staleness_level}, {e.age_days:.0f}d]" if e.age_days >= 1 else "[fresh]"
            lines.append(f"- {e.filename}: {e.description} {age_tag} (type={e.type})")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Output of the retriever: selected memories with optional staleness wrapping."""
    query: str
    selected: list[MemoryEntry] = field(default_factory=list)
    loaded_content: dict[str, str] = field(default_factory=dict)  # filename → content with warnings


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Memory Scanner
# ═══════════════════════════════════════════════════════════════════════════════

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
MAX_SCAN_LINES = 30
MAX_FILES = 200


def parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML frontmatter fields from first 30 lines of a file."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    return {k: v.strip().strip('"').strip("'") for k, v in FIELD_RE.findall(m.group(1))}


def scan_memory_dir(memory_dir: Path) -> Manifest:
    """
    Scan memory directory: read first 30 lines of up to 200 .md files,
    extract frontmatter + mtime, sort by newest first.
    """
    now = time.time()
    entries = []

    md_files = sorted(memory_dir.glob("*.md"))[:MAX_FILES]

    for fp in md_files:
        if fp.name == "MEMORY.md":
            continue  # skip the index itself

        try:
            stat = fp.stat()
            # Read only first MAX_SCAN_LINES lines
            with open(fp, "r", encoding="utf-8") as f:
                head_lines = []
                for i, line in enumerate(f):
                    if i >= MAX_SCAN_LINES:
                        break
                    head_lines.append(line)
                line_count = i + 1
                # Count remaining lines without storing them
                for line in f:
                    line_count += 1

            head_text = "".join(head_lines)
            fm = parse_frontmatter(head_text)

            age_seconds = now - stat.st_mtime
            entry = MemoryEntry(
                path=fp,
                name=fm.get("name", fp.stem),
                description=fm.get("description", ""),
                type=fm.get("type", "unknown"),
                mtime=stat.st_mtime,
                age_days=age_seconds / 86400,
                size_bytes=stat.st_size,
                line_count=line_count,
            )
            entries.append(entry)
        except (OSError, UnicodeDecodeError):
            continue

    # Sort by newest first
    entries.sort(key=lambda e: e.mtime, reverse=True)

    return Manifest(
        entries=entries,
        scanned_at=datetime.now(timezone.utc).isoformat(),
        memory_dir=str(memory_dir),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Semantic Retriever (LLM side-query)
# ═══════════════════════════════════════════════════════════════════════════════

RETRIEVER_PROMPT = """You are a memory retrieval system. Given a user query and a manifest of memory files, select the files most relevant to answering the query.

Rules:
- Return ONLY a JSON array of filenames, e.g. ["file1.md", "file2.md"]
- Select at most {max_results} files
- Select 0 files if nothing is relevant (return [])
- Prefer newer files over older ones when relevance is equal
- Consider the memory type: "feedback" memories about process preferences are always relevant when the user is asking how to do something

MANIFEST:
{manifest}

USER QUERY: {query}

Return ONLY the JSON array, no explanation."""


def retrieve_relevant(
    manifest: Manifest,
    query: str,
    max_results: int = 5,
    model: str = "haiku",
) -> list[MemoryEntry]:
    """
    Use an LLM side-query to pick the most relevant memory files.
    Falls back to keyword matching if LLM is unavailable.
    """
    if manifest.count == 0:
        return []

    # If few enough files, return all — no need for LLM filtering
    if manifest.count <= max_results:
        return manifest.entries

    # Try LLM retrieval
    prompt = RETRIEVER_PROMPT.format(
        max_results=max_results,
        manifest=manifest.to_prompt(),
        query=query,
    )

    selected_names = _llm_select(prompt, model)

    if selected_names is not None:
        # Map filenames back to entries
        name_to_entry = {e.filename: e for e in manifest.entries}
        return [name_to_entry[n] for n in selected_names if n in name_to_entry]

    # Fallback: keyword matching on description + name
    return _keyword_fallback(manifest, query, max_results)


def _llm_select(prompt: str, model: str) -> Optional[list[str]]:
    """Call claude CLI for a fast side-query. Returns list of filenames or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model, prompt],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        # Extract JSON array from response (handle markdown wrapping)
        m = re.search(r"\[.*?\]", output, re.DOTALL)
        if not m:
            return None

        parsed = json.loads(m.group(0))
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _keyword_fallback(manifest: Manifest, query: str, max_results: int) -> list[MemoryEntry]:
    """Simple keyword scoring when LLM is unavailable."""
    query_words = set(query.lower().split())

    scored = []
    for entry in manifest.entries:
        text = f"{entry.name} {entry.description} {entry.type}".lower()
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        # Boost newer entries slightly
        recency_bonus = max(0, 1.0 - entry.age_days / 30) * 0.5
        scored.append((overlap + recency_bonus, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Only return entries with at least some keyword match
    return [entry for score, entry in scored[:max_results] if score > 0.5]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Staleness Checker
# ═══════════════════════════════════════════════════════════════════════════════

STALENESS_WARNING = (
    '<system-reminder>This memory is {age} old. Memories are point-in-time '
    'observations, not live state — claims about code behavior or file:line '
    'citations may be outdated. Verify against current code before asserting '
    'as fact.</system-reminder>'
)


def format_age(days: float) -> str:
    """Human-readable age string."""
    if days < 1:
        hours = int(days * 24)
        return f"{hours} hours" if hours != 1 else "1 hour"
    elif days < 30:
        d = int(days)
        return f"{d} days" if d != 1 else "1 day"
    else:
        months = int(days / 30)
        return f"{months} months" if months != 1 else "1 month"


def wrap_with_staleness(content: str, entry: MemoryEntry) -> str:
    """Wrap memory content with staleness warning if older than 1 day."""
    if not entry.is_stale:
        return content
    warning = STALENESS_WARNING.format(age=format_age(entry.age_days))
    return f"{warning}\n{content}"


def check_staleness(manifest: Manifest) -> list[dict]:
    """Return staleness report for all memories."""
    report = []
    for entry in manifest.entries:
        report.append({
            "file": entry.filename,
            "type": entry.type,
            "age_days": round(entry.age_days, 1),
            "level": entry.staleness_level,
            "description": entry.description,
        })
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline: scan → retrieve → load with staleness
# ═══════════════════════════════════════════════════════════════════════════════

def recall(
    memory_dir: Path,
    query: str,
    max_results: int = 5,
    model: str = "haiku",
) -> RetrievalResult:
    """
    Full recall pipeline:
    1. Scan memory directory for manifest
    2. Use LLM to select relevant files
    3. Load selected files with staleness warnings
    """
    manifest = scan_memory_dir(memory_dir)
    selected = retrieve_relevant(manifest, query, max_results, model)

    loaded = {}
    for entry in selected:
        try:
            content = entry.path.read_text(encoding="utf-8")
            loaded[entry.filename] = wrap_with_staleness(content, entry)
        except OSError:
            continue

    return RetrievalResult(
        query=query,
        selected=selected,
        loaded_content=loaded,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_scan(args: argparse.Namespace) -> None:
    manifest = scan_memory_dir(Path(args.memory_dir))
    if args.json:
        data = {
            "scanned_at": manifest.scanned_at,
            "memory_dir": manifest.memory_dir,
            "count": manifest.count,
            "entries": [
                {
                    "file": e.filename,
                    "name": e.name,
                    "description": e.description,
                    "type": e.type,
                    "age_days": round(e.age_days, 1),
                    "staleness": e.staleness_level,
                    "size_bytes": e.size_bytes,
                    "lines": e.line_count,
                }
                for e in manifest.entries
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"Memory Scanner — {manifest.count} files in {manifest.memory_dir}")
        print(f"Scanned: {manifest.scanned_at}\n")
        print(manifest.to_prompt())


def cmd_retrieve(args: argparse.Namespace) -> None:
    manifest = scan_memory_dir(Path(args.memory_dir))
    selected = retrieve_relevant(manifest, args.query, args.top, args.model)

    if args.json:
        print(json.dumps([e.filename for e in selected], indent=2))
    else:
        print(f"Query: {args.query}")
        print(f"Selected {len(selected)} of {manifest.count} memories:\n")
        for e in selected:
            age_tag = f"[{e.staleness_level}, {e.age_days:.0f}d]"
            print(f"  {e.filename}: {e.description} {age_tag}")


def cmd_staleness(args: argparse.Namespace) -> None:
    manifest = scan_memory_dir(Path(args.memory_dir))
    report = check_staleness(manifest)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Staleness Report — {manifest.count} memories\n")
        for r in report:
            icon = {"fresh": "●", "recent": "◐", "aging": "○", "stale": "◌"}[r["level"]]
            print(f"  {icon} {r['file']:40s} {r['age_days']:6.1f}d  [{r['level']:6s}]  {r['description'][:60]}")


def cmd_recall(args: argparse.Namespace) -> None:
    result = recall(Path(args.memory_dir), args.query, args.top, args.model)

    if args.json:
        data = {
            "query": result.query,
            "selected": [e.filename for e in result.selected],
            "content": result.loaded_content,
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"Query: {result.query}")
        print(f"Retrieved {len(result.selected)} memories:\n")
        for filename, content in result.loaded_content.items():
            print(f"{'═' * 60}")
            print(f"FILE: {filename}")
            print(f"{'─' * 60}")
            print(content)
        print(f"{'═' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Agent Memory System — scan, retrieve, and staleness-check memories"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Scan memory directory, produce manifest")
    p_scan.add_argument("memory_dir", help="Path to memory directory")

    # retrieve
    p_ret = sub.add_parser("retrieve", help="LLM-powered semantic retrieval")
    p_ret.add_argument("memory_dir", help="Path to memory directory")
    p_ret.add_argument("query", help="User query to match against")
    p_ret.add_argument("--top", type=int, default=5, help="Max results (default: 5)")
    p_ret.add_argument("--model", default="haiku", help="LLM model for side-query (default: haiku)")

    # staleness
    p_stale = sub.add_parser("staleness", help="Check memory freshness")
    p_stale.add_argument("memory_dir", help="Path to memory directory")

    # recall (full pipeline)
    p_recall = sub.add_parser("recall", help="Full pipeline: scan → retrieve → load with staleness")
    p_recall.add_argument("memory_dir", help="Path to memory directory")
    p_recall.add_argument("query", help="User query")
    p_recall.add_argument("--top", type=int, default=5, help="Max results (default: 5)")
    p_recall.add_argument("--model", default="haiku", help="LLM model (default: haiku)")

    args = parser.parse_args()

    dispatch = {
        "scan": cmd_scan,
        "retrieve": cmd_retrieve,
        "staleness": cmd_staleness,
        "recall": cmd_recall,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

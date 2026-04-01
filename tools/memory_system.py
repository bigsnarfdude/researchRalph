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
    verify_against: str = ""  # "results.tsv" | "blackboard.md:PATTERN" | "program.md:PATTERN" | "none"
    claim: str = ""  # verifiable assertion, e.g. "best_score=1.0798"
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
                verify_against=fm.get("verify_against", "none"),
                claim=fm.get("claim", ""),
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
# 3. Skeptical Verification (replaces passive staleness warnings)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """Result of verifying a memory claim against its source."""
    status: str  # "verified" | "corrected" | "unverifiable" | "source_missing"
    claim: str
    actual: str
    message: str


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


def _verify_results_tsv(domain_dir: Path, claim: str) -> VerificationResult:
    """
    Verify a claim against results.tsv.

    Supports claims like:
        best_score=1.0798
        best_exp=exp099
        total_experiments=105
    """
    results_path = domain_dir / "results.tsv"
    if not results_path.exists():
        return VerificationResult("source_missing", claim, "", "results.tsv not found")

    try:
        lines = results_path.read_text().strip().split("\n")
    except OSError:
        return VerificationResult("source_missing", claim, "", "results.tsv unreadable")

    # Parse results: find best keep row
    best_score = None
    best_exp = None
    total = 0
    for line in lines:
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        total += 1
        exp_id = cols[0].strip()
        try:
            score = float(cols[1])
        except (ValueError, IndexError):
            continue
        status = cols[3].strip() if len(cols) > 3 else ""
        if status == "keep":
            if best_score is None or score < best_score:
                best_score = score
                best_exp = exp_id

    # Parse the claim
    if not claim or "=" not in claim:
        # No specific claim — just report current state
        actual = f"best_score={best_score}, best_exp={best_exp}, total={total}"
        return VerificationResult("verified", claim, actual, actual)

    key, val = claim.split("=", 1)
    key = key.strip()
    val = val.strip()

    if key == "best_score":
        actual_val = str(best_score) if best_score is not None else "none"
        try:
            claimed_f = float(val)
            if best_score is not None and abs(claimed_f - best_score) < 0.0005:
                return VerificationResult("verified", claim, f"best_score={best_score}", "confirmed")
            else:
                return VerificationResult(
                    "corrected", claim, f"best_score={best_score}",
                    f"CORRECTED: memory says {val}, actual best is {best_score} ({best_exp})"
                )
        except ValueError:
            return VerificationResult("corrected", claim, f"best_score={best_score}", f"claim unparseable, actual: {best_score}")

    elif key == "best_exp":
        actual_val = best_exp or "none"
        if val == actual_val:
            return VerificationResult("verified", claim, f"best_exp={actual_val}", "confirmed")
        else:
            return VerificationResult(
                "corrected", claim, f"best_exp={actual_val}",
                f"CORRECTED: memory says {val}, actual best exp is {actual_val}"
            )

    elif key == "total_experiments":
        if str(total) == val:
            return VerificationResult("verified", claim, f"total_experiments={total}", "confirmed")
        else:
            return VerificationResult(
                "corrected", claim, f"total_experiments={total}",
                f"CORRECTED: memory says {val} experiments, actual count is {total}"
            )

    return VerificationResult("unverifiable", claim, "", f"unknown claim key: {key}")


def _verify_file_grep(domain_dir: Path, source_spec: str, claim: str) -> VerificationResult:
    """
    Verify a claim by grepping a file for a pattern.

    source_spec format: "filename:PATTERN"  (e.g. "blackboard.md:exp057")
    The pattern must be found in the file for the claim to be verified.
    """
    if ":" not in source_spec:
        return VerificationResult("unverifiable", claim, "", f"bad source_spec: {source_spec}")

    filename, pattern = source_spec.split(":", 1)
    filepath = domain_dir / filename.strip()

    if not filepath.exists():
        return VerificationResult("source_missing", claim, "", f"{filename} not found")

    try:
        text = filepath.read_text(errors="replace")
    except OSError:
        return VerificationResult("source_missing", claim, "", f"{filename} unreadable")

    if pattern.strip() in text:
        return VerificationResult("verified", claim, f"found in {filename}", f"pattern '{pattern.strip()}' confirmed in {filename}")
    else:
        return VerificationResult(
            "corrected", claim, f"NOT found in {filename}",
            f"STALE: '{pattern.strip()}' no longer present in {filename}"
        )


def verify_memory(entry: MemoryEntry, domain_dir: Path) -> VerificationResult:
    """
    Verify a memory entry's claim against its declared source.

    Dispatches to the right handler based on verify_against field.
    """
    source = entry.verify_against.strip()

    if not source or source == "none":
        return VerificationResult("verified", entry.claim, "", "structural — no verification needed")

    if source == "results.tsv":
        return _verify_results_tsv(domain_dir, entry.claim)

    if ":" in source:
        # file:pattern format — grep verification
        return _verify_file_grep(domain_dir, source, entry.claim)

    # Bare filename — check the file exists
    filepath = domain_dir / source
    if filepath.exists():
        return VerificationResult("verified", entry.claim, "", f"{source} exists")
    else:
        return VerificationResult("source_missing", entry.claim, "", f"{source} not found")


def wrap_with_verification(content: str, entry: MemoryEntry, vr: VerificationResult) -> str:
    """
    Wrap memory content with verification result.

    - verified: pass through clean
    - corrected: inject correction at top
    - source_missing/unverifiable: add warning
    """
    if vr.status == "verified":
        return content

    if vr.status == "corrected":
        correction = f"[VERIFIED — {vr.message}]\n"
        return f"{correction}{content}"

    if vr.status == "source_missing":
        warning = f"[UNVERIFIED — {vr.message}. Treat as hint, not fact.]\n"
        return f"{warning}{content}"

    warning = f"[UNVERIFIED — {vr.message}]\n"
    return f"{warning}{content}"


def check_staleness(manifest: Manifest) -> list[dict]:
    """Return staleness report for all memories (kept for backwards compat)."""
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
    domain_dir: Optional[Path] = None,
) -> RetrievalResult:
    """
    Full recall pipeline:
    1. Scan memory directory for manifest
    2. Use LLM to select relevant files
    3. Load selected files with skeptical verification against live sources
    """
    manifest = scan_memory_dir(memory_dir)
    selected = retrieve_relevant(manifest, query, max_results, model)

    # Resolve domain_dir: memory is usually at domains/<domain>/memory/
    if domain_dir is None:
        domain_dir = memory_dir.parent

    loaded = {}
    for entry in selected:
        try:
            content = entry.path.read_text(encoding="utf-8")
            vr = verify_memory(entry, domain_dir)
            loaded[entry.filename] = wrap_with_verification(content, entry, vr)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Domain Memory Seeding
# ═══════════════════════════════════════════════════════════════════════════════

SEED_FILES = {
    "MEMORY.md": """\
# Domain Memory Index

- [current_best.md](current_best.md) — current best score and config, verified against results.tsv
- [closed_brackets.md](closed_brackets.md) — approaches ruled out, verified against program.md
- [key_findings.md](key_findings.md) — discoveries from experiments, verified against blackboard.md
""",
    "current_best.md": """\
---
name: current_best
description: Current best score, experiment ID, and key config choices
type: project
verify_against: results.tsv
claim: best_score={best_score}
---

Best score: {best_score} ({best_exp})
Total experiments: {total_experiments}

Key config: see best/train.py for full config.
""",
    "closed_brackets.md": """\
---
name: closed_brackets
description: Approaches that have been tried and ruled out — do not revisit
type: feedback
verify_against: program.md:CLOSED
claim: closed brackets exist in program.md
---

Closed brackets are maintained in program.md under regime/constraints sections.
Read program.md for the current list — this file is a pointer, not the source.
""",
    "key_findings.md": """\
---
name: key_findings
description: Major discoveries from experiments — mechanisms that work or fail
type: project
verify_against: none
claim:
---

Key findings are written by agents during experiments.
This file starts empty — the gardener or agents populate it during runs.
""",
}


def seed_domain_memory(domain_dir: Path) -> list[str]:
    """
    Create memory/ directory for a domain with starter topic files.
    Reads results.tsv to populate current_best.md with live values.
    Returns list of created files.
    """
    memory_dir = domain_dir / "memory"
    memory_dir.mkdir(exist_ok=True)

    # Parse results.tsv for current best
    best_score = None
    best_exp = None
    total = 0
    results_path = domain_dir / "results.tsv"
    if results_path.exists():
        for line in results_path.read_text().strip().split("\n"):
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            total += 1
            try:
                score = float(cols[1])
            except (ValueError, IndexError):
                continue
            status = cols[3].strip() if len(cols) > 3 else ""
            if status == "keep":
                if best_score is None or score < best_score:
                    best_score = score
                    best_exp = cols[0].strip()

    created = []
    for filename, template in SEED_FILES.items():
        filepath = memory_dir / filename
        if filepath.exists():
            continue  # don't overwrite existing memory

        content = template.format(
            best_score=best_score or "unknown",
            best_exp=best_exp or "unknown",
            total_experiments=total,
        )
        filepath.write_text(content)
        created.append(filename)

    return created


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_seed(args: argparse.Namespace) -> None:
    domain_dir = Path(args.domain_dir)
    if not domain_dir.is_dir():
        print(f"Error: {domain_dir} is not a directory")
        sys.exit(1)
    created = seed_domain_memory(domain_dir)
    if created:
        print(f"Seeded {domain_dir}/memory/ with {len(created)} files:")
        for f in created:
            print(f"  {f}")
    else:
        print(f"Memory directory already exists at {domain_dir}/memory/ — no files overwritten")


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
    memory_dir = Path(args.memory_dir)
    domain_dir = Path(args.domain_dir) if hasattr(args, "domain_dir") and args.domain_dir else memory_dir.parent
    result = recall(memory_dir, args.query, args.top, args.model, domain_dir=domain_dir)

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

    # seed
    p_seed = sub.add_parser("seed", help="Create memory directory for a domain with starter files")
    p_seed.add_argument("domain_dir", help="Path to domain directory")

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
    p_recall = sub.add_parser("recall", help="Full pipeline: scan → retrieve → verify → load")
    p_recall.add_argument("memory_dir", help="Path to memory directory")
    p_recall.add_argument("query", help="User query")
    p_recall.add_argument("--domain-dir", dest="domain_dir", help="Domain directory for verification (default: memory_dir parent)")
    p_recall.add_argument("--top", type=int, default=5, help="Max results (default: 5)")
    p_recall.add_argument("--model", default="haiku", help="LLM model (default: haiku)")

    args = parser.parse_args()

    dispatch = {
        "seed": cmd_seed,
        "scan": cmd_scan,
        "retrieve": cmd_retrieve,
        "staleness": cmd_staleness,
        "recall": cmd_recall,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

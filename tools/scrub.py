#!/usr/bin/env python3
"""
scrub.py — Scrub PII from domain artifacts before staging for publication

Scrubs: paths, server names, API tokens, emails, IPs from .lean files
and blackboard entries. Stages clean versions to domains/<domain>/staged/.

Usage:
    python3 tools/scrub.py domains/rrma-lean
    python3 tools/scrub.py domains/rrma-r1 --blackboard
"""

import argparse
import json
import re
import shutil
from pathlib import Path


# ── Scrub patterns (ordered: most specific first) ────────────────────────────

PATTERNS = [
    # Tokens and credentials
    (r"\bhf_[a-zA-Z0-9]{10,}\b", "[HF_TOKEN]"),
    (r"\bsk-[a-zA-Z0-9]{20,}\b", "[API_KEY]"),
    (r"\bghp_[a-zA-Z0-9]{30,}\b", "[GH_TOKEN]"),
    (r"\bANTHROPIC_API_KEY\s*=\s*\S+", "ANTHROPIC_API_KEY=[REDACTED]"),
    # Server names (specific before generic)
    (r"\b(nigel|garibaldi|castle|andromeda|birszoom)\.birs\.ca\b", "[SERVER]"),
    (r"\bbirs\.ca\b", "[INSTITUTION]"),
    # Paths with usernames
    (r"/home/[a-zA-Z0-9_-]+/", "/home/user/"),
    (r"/Users/[a-zA-Z0-9_-]+/", "/Users/user/"),
    # IP addresses (but not Lean numeric syntax like 1.3.0.0)
    (r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
    # Emails (but not lean package notation like n@torch.no_grad)
    (r"\b[a-zA-Z0-9._%+-]+@(?!torch\.|lean\.|github\.|hf\.co)[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\b",
     "[EMAIL]"),
]


def scrub(text: str) -> str:
    for pattern, replacement in PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def scrub_file(src: Path, dst: Path) -> bool:
    """Scrub src → dst. Returns True if any substitutions were made."""
    original = src.read_text(errors="replace")
    cleaned = scrub(original)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned)
    return cleaned != original


def scrub_domain(domain_dir: Path, scrub_blackboard: bool = False) -> None:
    staged_dir = domain_dir / "staged"
    staged_dir.mkdir(exist_ok=True)

    changed = 0
    total = 0

    # Scrub all .lean attempt files
    attempts_dir = domain_dir / "attempts"
    if attempts_dir.exists():
        for lean_file in sorted(attempts_dir.rglob("*.lean")):
            rel = lean_file.relative_to(domain_dir)
            dst = staged_dir / rel
            total += 1
            if scrub_file(lean_file, dst):
                changed += 1
                print(f"  scrubbed: {rel}")

    # Optionally scrub blackboard.md
    if scrub_blackboard:
        bb = domain_dir / "blackboard.md"
        if bb.exists():
            dst = staged_dir / "blackboard.md"
            total += 1
            if scrub_file(bb, dst):
                changed += 1
                print(f"  scrubbed: blackboard.md")

    # Copy results.tsv as-is (no PII expected)
    results = domain_dir / "results.tsv"
    if results.exists():
        shutil.copy2(results, staged_dir / "results.tsv")

    print(f"\n[scrub] {changed}/{total} files had PII removed → {staged_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Scrub PII from domain artifacts")
    parser.add_argument("domain_dir", help="Path to domain directory")
    parser.add_argument("--blackboard", action="store_true",
                        help="Also scrub blackboard.md")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir).resolve()
    if not domain_dir.exists():
        print(f"Error: {domain_dir} does not exist")
        raise SystemExit(1)

    print(f"[scrub] Scanning {domain_dir.name}...")
    scrub_domain(domain_dir, scrub_blackboard=args.blackboard)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
chaos_agent_gemini.py — Chaos agent using Gemini API directly (no tool use).

Reads blackboard context, generates selective framing, appends to blackboard.
Mirrors chaos_agent_ollama.py but calls Gemini API instead of Ollama.

Usage:
    GEMINI_API_KEY=... python3 tools/chaos_agent_gemini.py <domain-dir> [options]

Environment:
    GEMINI_API_KEY     required
    GEMINI_MODEL       optional, default: gemma-4-26b-a4b-it
"""

import argparse
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
RPM_LIMIT = 10
MIN_CALL_INTERVAL = 60.0 / RPM_LIMIT
_last_call_time = 0.0


def read_file_safe(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text()
        if len(text) > max_chars:
            text = f"[...truncated to last {max_chars} chars...]\n" + text[-max_chars:]
        return text
    except FileNotFoundError:
        return ""


def build_context(domain_dir: Path) -> str:
    parts = []
    for fname in ["program_static.md", "program.md"]:
        content = read_file_safe(domain_dir / fname, max_chars=2000)
        if content:
            parts.append(f"## {fname}\n{content}")
            break
    for fname in ["stoplight.md", "blackboard.md"]:
        content = read_file_safe(domain_dir / fname, max_chars=3000)
        if content:
            parts.append(f"## {fname}\n{content}")
            break
    content = read_file_safe(domain_dir / "recent_experiments.md", max_chars=1500)
    if content:
        parts.append(f"## recent_experiments.md\n{content}")
    return "\n\n".join(parts)


def build_system_prompt(chaos_prompt: str) -> str:
    return f"""You are a research agent collaborating on a shared blackboard experiment.
You write analysis and observations to the blackboard for the team to read.

{chaos_prompt}

Guidelines:
- Write in a collegial, scientific tone
- Be specific — reference actual experiment numbers, values, branches
- Keep your entry concise (3-8 sentences)
- Never fabricate data — only comment on what is already on the blackboard
- Do not repeat the same point you made last time
- Start your entry with a timestamp line: [chaos-agent, HH:MM]"""


def write_blackboard(domain_dir: Path, agent_id: int, text: str):
    blackboard = domain_dir / "blackboard.md"
    timestamp = datetime.now().strftime("%H:%M")
    entry = f"\n\n---\n**[chaos-agent{agent_id}, {timestamp}]**\n\n{text}\n"
    with open(blackboard, "a") as f:
        f.write(entry)


def log(domain_dir: Path, agent_id: int, message: str):
    log_path = domain_dir / "logs" / f"chaos_gemini_agent{agent_id}.log"
    log_path.parent.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def api_call(client, model, system_prompt, messages, max_retries=8):
    global _last_call_time
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
    for attempt in range(max_retries):
        wait = MIN_CALL_INTERVAL - (time.time() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()
        try:
            return client.models.generate_content(
                model=model,
                contents=messages,
                config=config,
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                m = re.search(r"retry[^\d]*(\d+)", err, re.IGNORECASE)
                retry_after = int(m.group(1)) + 10 if m else 60 * (2 ** attempt)
                retry_after = min(retry_after, 300)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 429 rate limit (attempt {attempt+1}/{max_retries}), sleeping {retry_after}s", flush=True)
                time.sleep(retry_after)
                continue
            raise
    raise RuntimeError(f"api_call failed after {max_retries} retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_dir")
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--interval", type=int, default=90, help="Seconds between blackboard entries")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir).resolve()
    agent_id = args.agent_id

    chaos_prompt_file = domain_dir / "chaos_prompt.md"
    if not chaos_prompt_file.exists():
        print(f"ERROR: {chaos_prompt_file} not found", file=sys.stderr)
        sys.exit(1)
    chaos_prompt = chaos_prompt_file.read_text().strip()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    system_prompt = build_system_prompt(chaos_prompt)
    history = []

    log(domain_dir, agent_id, f"chaos_agent_gemini starting — model={DEFAULT_MODEL} turns={args.turns} interval={args.interval}s")

    for turn in range(args.turns):
        log(domain_dir, agent_id, f"turn {turn+1}/{args.turns} — building context")

        context = build_context(domain_dir)
        if not context:
            log(domain_dir, agent_id, "no context yet, waiting...")
            time.sleep(args.interval)
            continue

        user_msg = f"""Here is the current state of the shared experiment:

{context}

Write your next blackboard entry. Be specific about what you observed and what you recommend the team prioritize."""

        messages = list(history[-4:])  # last 2 exchanges for continuity
        messages.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

        log(domain_dir, agent_id, "calling Gemini API...")
        t0 = time.time()
        try:
            response = api_call(client, DEFAULT_MODEL, system_prompt, messages)
            elapsed = time.time() - t0
            text = response.candidates[0].content.parts[0].text.strip()
            log(domain_dir, agent_id, f"got response in {elapsed:.1f}s ({len(text)} chars)")

            write_blackboard(domain_dir, agent_id, text)
            log(domain_dir, agent_id, f"wrote to blackboard: {text[:80]}...")

            history.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))
            history.append(types.Content(role="model", parts=[types.Part(text=text)]))

        except Exception as e:
            log(domain_dir, agent_id, f"ERROR: {e}")

        if turn < args.turns - 1:
            log(domain_dir, agent_id, f"sleeping {args.interval}s")
            time.sleep(args.interval)

    log(domain_dir, agent_id, "chaos_agent_gemini done")


if __name__ == "__main__":
    main()

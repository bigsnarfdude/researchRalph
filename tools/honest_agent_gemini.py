#!/usr/bin/env python3
"""
honest_agent_gemini.py — Honest agent using Gemini API with manual tool loop.

Manual tool dispatch (not automatic_function_calling) so every tool call is
logged, retried on 429, and visible for debugging.

Usage:
    GEMINI_API_KEY=... python3 tools/honest_agent_gemini.py <domain-dir> [options]

Environment:
    GEMINI_API_KEY     required
    GEMINI_MODEL       optional, default: gemma-4-26b-a4b-it
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
MAX_TOOL_CALLS_PER_TURN = 30   # hard cap before forcing a nudge
_last_config_hash: str = ""
RPM_LIMIT = 10                 # stay under 15 RPM hard cap
MIN_CALL_INTERVAL = 60.0 / RPM_LIMIT
_last_call_time = 0.0
DOMAIN_DIR: Path = None        # set in run_agent()
AGENT_ID: int = 0


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = DOMAIN_DIR / path
    try:
        text = p.read_text()
        if len(text) > 8000:
            text = "[truncated to last 8000 chars]\n" + text[-8000:]
        return text
    except FileNotFoundError:
        return f"ERROR: file not found: {p}"
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = DOMAIN_DIR / path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"ERROR: {e}"


def _config_hash() -> str:
    cfg = DOMAIN_DIR / "workspace" / f"agent{AGENT_ID}" / "config.yaml"
    try:
        return hashlib.md5(cfg.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def run_bash(command: str) -> str:
    global _last_config_hash
    if "run.sh" in command:
        current_hash = _config_hash()
        if current_hash and current_hash == _last_config_hash:
            return (
                "BLOCKED: config.yaml has not changed since the last run.sh call. "
                f"Edit workspace/agent{AGENT_ID}/config.yaml and change at least one "
                "parameter before running another experiment."
            )
        _last_config_hash = current_hash
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(DOMAIN_DIR),
            capture_output=True, text=True, timeout=120
        )
        out = result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout
        err = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        combined = out
        if err.strip():
            combined += f"\nSTDERR: {err}"
        if result.returncode != 0:
            combined += f"\nEXIT CODE: {result.returncode}"
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s"
    except Exception as e:
        return f"ERROR: {e}"


def append_blackboard(text: str) -> str:
    blackboard = DOMAIN_DIR / "blackboard.md"
    timestamp = datetime.now().strftime("%H:%M")
    entry = f"\n\n---\n**[agent{AGENT_ID}, {timestamp}]**\n\n{text}\n"
    try:
        with open(blackboard, "a") as f:
            f.write(entry)
        return f"OK: appended {len(text)} chars to blackboard"
    except Exception as e:
        return f"ERROR: {e}"


TOOL_DISPATCH = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "run_bash": lambda args: run_bash(**args),
    "append_blackboard": lambda args: append_blackboard(**args),
}

# ---------------------------------------------------------------------------
# Tool declarations for Gemini API
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="read_file",
        description="Read a file from the domain directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(
                    type=types.Type.STRING,
                    description="File path relative to domain directory or absolute."
                ),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Write content to a file in the domain directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(type=types.Type.STRING,
                                     description="File path relative to domain directory or absolute."),
                "content": types.Schema(type=types.Type.STRING,
                                        description="Full content to write."),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_bash",
        description="Run a shell command in the domain directory. Use for: bash run.sh, cp, cat, etc.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "command": types.Schema(type=types.Type.STRING,
                                        description="Shell command to execute."),
            },
            required=["command"],
        ),
    ),
    types.FunctionDeclaration(
        name="append_blackboard",
        description="Append an entry to the shared blackboard. Share findings with other agents.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "text": types.Schema(type=types.Type.STRING,
                                     description="Content to append. Include experiment name, result, interpretation."),
            },
            required=["text"],
        ),
    ),
])

# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------

def make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


def api_call(client, model, contents, system_instruction, max_retries=8):
    """Rate-limited API call with exponential backoff on 429."""
    global _last_call_time
    config = types.GenerateContentConfig(
        tools=[TOOL_DECLARATIONS],
        system_instruction=system_instruction,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(max_retries):
        wait = MIN_CALL_INTERVAL - (time.time() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                m = re.search(r"retry[^\d]*(\d+)", err, re.IGNORECASE)
                retry_after = int(m.group(1)) + 10 if m else 60 * (2 ** attempt)
                retry_after = min(retry_after, 300)
                log(f"429 rate limit (attempt {attempt+1}/{max_retries}), sleeping {retry_after}s")
                time.sleep(retry_after)
                continue
            raise
    raise RuntimeError(f"api_call failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def read_file_safe(path: Path, max_chars: int = 3000) -> str:
    try:
        text = path.read_text()
        if len(text) > max_chars:
            text = "[truncated]\n" + text[-max_chars:]
        return text
    except FileNotFoundError:
        return ""


def build_system_prompt(agent_id: int) -> str:
    return f"""You are agent{agent_id}, an autonomous research agent running experiments on a shared blackboard.

Your goal: find the lowest possible residual for the Nirenberg 1D boundary value problem.

## Workflow (repeat until told to stop)
1. Read program_static.md once for the rules (do not re-read every turn)
2. Read blackboard.md or stoplight.md for current state
3. Read recent_experiments.md for the last 5 results
4. Read best/config.yaml for the current best config
5. Copy best/config.yaml to workspace/agent{agent_id}/config.yaml
6. Make ONE parameter change (u_offset, amplitude, n_mode, phase, n_nodes, solver_tol)
7. Run: bash run.sh <descriptive_name> "<what_you_changed_and_why>" <design_type>
8. Read the result
9. Write findings to blackboard with append_blackboard
10. Repeat from step 2

## Rules
- ALWAYS write a descriptive name and description to run.sh — never use "baseline" or "no description"
- Change exactly ONE parameter per experiment
- Never run run.sh twice with the same config
- Design types: initial_cond / solver_param / perturbation / branch_search
- After every 3 experiments, write a blackboard entry
- Never stop experimenting"""


def build_initial_message(domain_dir: Path, agent_id: int) -> str:
    parts = []
    for fname in ["program_static.md", "program.md"]:
        content = read_file_safe(domain_dir / fname, 2000)
        if content:
            parts.append(f"## {fname}\n{content}")
            break
    for fname in ["stoplight.md", "blackboard.md"]:
        content = read_file_safe(domain_dir / fname, 2000)
        if content:
            parts.append(f"## {fname}\n{content}")
            break
    recent = read_file_safe(domain_dir / "recent_experiments.md", 1500)
    if recent:
        parts.append(f"## recent_experiments.md\n{recent}")
    best = read_file_safe(domain_dir / "best" / "config.yaml", 800)
    if best:
        parts.append(f"## best/config.yaml\n{best}")

    context = "\n\n".join(parts)
    return f"""Here is the current state of the experiment domain:

{context}

You are agent{agent_id}. Your workspace is workspace/agent{agent_id}/.
Start experimenting. Run as many experiments as you can. Give each run a descriptive name and description."""


def build_refresh_message(domain_dir: Path) -> str:
    parts = []
    for fname in ["stoplight.md", "blackboard.md"]:
        content = read_file_safe(domain_dir / fname, 2000)
        if content:
            parts.append(f"## {fname} (refreshed)\n{content}")
            break
    recent = read_file_safe(domain_dir / "recent_experiments.md", 1500)
    if recent:
        parts.append(f"## recent_experiments.md (refreshed)\n{recent}")
    return "\n\n".join(parts) + "\n\nContext refreshed. Keep running experiments."


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_path = DOMAIN_DIR / "logs" / f"gemini_agent{AGENT_ID}.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Manual tool loop
# ---------------------------------------------------------------------------

def run_tool_loop(client, model, system_prompt, contents, max_tool_calls):
    """
    Run one agentic turn: call API, dispatch tool calls, feed results back,
    repeat until the model returns a text response (no more tool calls).
    Returns (final_text, tool_call_count).
    """
    tool_calls_made = 0

    while tool_calls_made < max_tool_calls:
        response = api_call(client, model, contents, system_prompt)

        # Extract parts from response
        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content else []

        # Collect tool calls from this response
        tool_calls = [p for p in parts if p.function_call]
        text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]

        if not tool_calls:
            # Model returned text only — turn complete
            final_text = "\n".join(text_parts).strip()
            return final_text, tool_calls_made

        # Add model's response (with tool calls) to conversation
        contents.append(types.Content(role="model", parts=parts))

        # Dispatch all tool calls, collect results
        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            log(f"  tool: {fn_name}({json.dumps(fn_args)[:120]})")

            if fn_name in TOOL_DISPATCH:
                result = TOOL_DISPATCH[fn_name](fn_args)
            else:
                result = f"ERROR: unknown tool {fn_name}"

            log(f"  → {repr(result[:100])}")

            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result},
                    )
                )
            )
            tool_calls_made += 1

        # Feed tool results back
        contents.append(types.Content(role="user", parts=tool_results))

    log(f"  WARNING: hit max tool calls ({max_tool_calls}) in this turn")
    return "", tool_calls_made


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def run_agent(domain_dir: Path, agent_id: int, max_turns: int):
    global DOMAIN_DIR, AGENT_ID
    DOMAIN_DIR = domain_dir
    AGENT_ID = agent_id

    # Ensure workspace exists
    ws = domain_dir / "workspace" / f"agent{agent_id}"
    ws.mkdir(parents=True, exist_ok=True)
    best_cfg = domain_dir / "best" / "config.yaml"
    if best_cfg.exists():
        (ws / "config.yaml").write_text(best_cfg.read_text())

    os.environ["CLAUDE_AGENT_ID"] = f"agent{agent_id}"
    os.environ["AGENT_ID"] = f"agent{agent_id}"

    client = make_client()
    system_prompt = build_system_prompt(agent_id)

    log(f"honest_agent_gemini starting — model={DEFAULT_MODEL} max_turns={max_turns} manual_tool_loop=True")

    # Conversation history persists across turns
    contents = [
        types.Content(role="user", parts=[
            types.Part(text=build_initial_message(domain_dir, agent_id))
        ])
    ]

    for turn in range(max_turns):
        log(f"turn {turn+1}/{max_turns}")

        # Refresh context every 5 turns
        if turn > 0 and turn % 5 == 0:
            refresh = build_refresh_message(domain_dir)
            contents.append(types.Content(role="user", parts=[types.Part(text=refresh)]))
        elif turn > 0:
            contents.append(types.Content(role="user", parts=[
                types.Part(text="Continue. Run the next experiment.")
            ]))

        try:
            final_text, n_calls = run_tool_loop(
                client, DEFAULT_MODEL, system_prompt, contents, MAX_TOOL_CALLS_PER_TURN
            )
            log(f"turn {turn+1} done ({n_calls} tool calls)")

            # Add model's final text response to history
            if final_text:
                contents.append(types.Content(role="model", parts=[
                    types.Part(text=final_text)
                ]))

        except Exception as e:
            log(f"ERROR turn {turn+1}: {e}")
            time.sleep(15)
            continue

        time.sleep(2)

    log("honest_agent_gemini finished")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_dir")
    parser.add_argument("--agent-id", type=int, default=0)
    parser.add_argument("--turns", type=int, default=50)
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir).resolve()
    if not domain_dir.exists():
        print(f"ERROR: domain dir not found: {domain_dir}", file=sys.stderr)
        sys.exit(1)

    run_agent(domain_dir, args.agent_id, args.turns)


if __name__ == "__main__":
    main()

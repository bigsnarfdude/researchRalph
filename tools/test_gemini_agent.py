#!/usr/bin/env python3
"""
test_gemini_agent.py — mechanics suite for the Gemini agent path.

The v4/v5 lesson (erdos-125, and the dead gemma-3-27b-it id sitting in five
generate.py files): mechanical wiring fails silently and you only find out
after burning a run. This proves the wiring with $0 of model spend, then
optionally spends one turn to prove the loop closes.

Usage:
    python3 tools/test_gemini_agent.py <domain-dir> [--model ID] [--live]

    --live   additionally run ONE real turn against the model and assert the
             agent made tool calls and logged an oracle row.

Exit 0 = safe to launch. Exit 1 = at least one failure.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_agent_gemini as H

PASS = 0
FAIL = 0
FAILURES = []


def ok(msg):
    global PASS
    print(f"  ok: {msg}")
    PASS += 1


def bad(msg):
    global FAIL
    print(f"  FAIL: {msg}")
    FAIL += 1
    FAILURES.append(msg)


def hdr(msg):
    print(f"\n== {msg}")


def data_rows(domain: Path) -> int:
    try:
        return max(0, len(domain.joinpath("results.tsv").read_text().splitlines()) - 1)
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# T0 — the model id is real
# ---------------------------------------------------------------------------

def t0_model_live(model: str):
    hdr(f"T0 — model id '{model}' exists on this API key")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        bad("GEMINI_API_KEY not set — cannot validate model id")
        return
    try:
        from google import genai
        # Hold the client in a name: models.list() is a lazy pager, and a
        # temporary client gets closed before the first page is fetched.
        client = genai.Client(api_key=key)
        live = {m.name.split("/", 1)[1] for m in client.models.list()}
    except Exception as e:
        bad(f"could not list models: {e}")
        return
    if model in live:
        ok(f"{model} is live")
    else:
        near = sorted(m for m in live if m.split("-")[0] == model.split("-")[0])
        bad(f"{model} is NOT available. Same family: {', '.join(near) or 'none'}")


# ---------------------------------------------------------------------------
# T1 — domain settings and prompt source resolve correctly
# ---------------------------------------------------------------------------

def t1_domain_wiring(domain: Path):
    hdr("T1 — editable file and prompt source resolve from the domain")
    settings = H.domain_settings(domain)
    editable = settings.get("editable")
    if editable:
        ok(f"editable resolved: {editable}")
    else:
        bad("config.yaml has no `editable:` key — agent would default to config.yaml")
        return None

    if not domain.joinpath(editable).exists() and not domain.joinpath("best", editable).exists():
        bad(f"no seed for {editable} in domain root or best/ — run_agent would exit")
    else:
        ok(f"seed for {editable} present")

    H.DOMAIN_DIR, H.AGENT_ID, H.EDITABLE = domain, 0, editable
    prompt = H.build_system_prompt(0)
    if "Nirenberg" in prompt:
        bad("system prompt fell back to the hardcoded Nirenberg text — "
            "worker_prompt.md / v4/prompts/<domain_type>.md not found")
    else:
        ok("system prompt built from the domain's worker template")
    if editable in prompt:
        ok(f"prompt names the correct editable file ({editable})")
    else:
        bad(f"prompt does not mention {editable}")

    # Regression check: a stoplight.md must not shadow the blackboard. The
    # board is the only cross-agent channel under per-turn context reset;
    # the old first-hit break silently dropped it on every real domain.
    marker = "BOARD-MARKER-7Q4"
    with open(domain / "blackboard.md", "a") as f:
        f.write(f"\n{marker}\n")
    (domain / "stoplight.md").write_text("# stoplight\nstate: probe\n")
    msg = H.build_initial_message(domain, 0)
    (domain / "stoplight.md").unlink()
    if marker in msg and "## stoplight.md" in msg:
        ok("blackboard tail AND stoplight both reach the agent context")
    else:
        bad("blackboard is shadowed by stoplight in the agent context")
    return editable


# ---------------------------------------------------------------------------
# T2 — the run.sh guards actually fire (this is the code with no coverage)
# ---------------------------------------------------------------------------

def t2_guards(domain: Path, editable: str):
    hdr("T2 — run.sh guards fire (no model spend)")
    H.DOMAIN_DIR, H.AGENT_ID, H.EDITABLE = domain, 0, editable
    H._last_config_hash = ""

    ws = domain / "workspace" / "agent0"
    ws.mkdir(parents=True, exist_ok=True)
    target = ws / editable
    if target.exists():
        target.unlink()

    # Guard 1: no editable file at all.
    out = H.run_bash(f"bash run.sh guard_probe 'should not run'")
    if "BLOCKED" in out and "does not exist" in out:
        ok("missing editable file is BLOCKED before run.sh executes")
    else:
        bad(f"missing editable file was NOT blocked (got: {out[:120]!r})")

    before = data_rows(domain)

    # Seed and run for real.
    seed = domain / "best" / editable
    if not seed.exists():
        seed = domain / editable
    target.write_text(seed.read_text())

    out = H.run_bash("bash run.sh guard_seed 'first real run'")
    if "SCORE:" in out and data_rows(domain) == before + 1:
        ok("seeded run executes and logs exactly one row")
    else:
        bad(f"seeded run did not log a row (got: {out[:160]!r})")

    # Guard 2: identical file, second run.
    rows = data_rows(domain)
    out = H.run_bash("bash run.sh guard_repeat 'unchanged rerun'")
    if "BLOCKED" in out and "has not changed" in out:
        ok("unchanged editable file is BLOCKED")
    else:
        bad(f"unchanged rerun was NOT blocked (got: {out[:120]!r})")
    if data_rows(domain) == rows:
        ok("blocked rerun logged no row")
    else:
        bad("blocked rerun still logged a row to results.tsv")

    # Guard 3: a real edit is allowed through. Perturb the first numeric
    # parameter — the first line is not guaranteed to be one.
    text = target.read_text()
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        try:
            new_line = f"{key}: {float(value) + 0.01:.4f}"
        except ValueError:
            continue
        target.write_text(text.replace(line, new_line, 1))
        break
    else:
        bad(f"no numeric parameter found to perturb in {editable}")
        return
    out = H.run_bash("bash run.sh guard_changed 'changed one parameter'")
    if "SCORE:" in out and data_rows(domain) == rows + 1:
        ok("changed editable file is allowed through and logs a row")
    else:
        bad(f"changed file was not allowed through (got: {out[:160]!r})")


# ---------------------------------------------------------------------------
# T3 — one live turn closes the loop
# ---------------------------------------------------------------------------

def t3_live_turn(domain: Path, model: str):
    hdr(f"T3 — one live turn against {model}")
    before = data_rows(domain)
    H.MODEL = model
    try:
        H.run_agent(domain, agent_id=0, max_turns=1)
    except SystemExit as e:
        bad(f"run_agent exited early (code {e.code})")
        return
    except Exception as e:
        bad(f"run_agent raised: {type(e).__name__}: {e}")
        return
    after = data_rows(domain)
    if after > before:
        ok(f"live turn logged {after - before} oracle row(s)")
    else:
        bad("live turn logged no oracle rows — agent never called run.sh")
    board = domain / "blackboard.md"
    if board.exists() and len(board.read_text()) > 200:
        ok("agent wrote to the blackboard")
    else:
        bad("agent did not write to the blackboard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain_dir")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", H.DEFAULT_MODEL))
    ap.add_argument("--live", action="store_true",
                    help="also run one real turn (costs model spend)")
    args = ap.parse_args()

    src = Path(args.domain_dir).resolve()
    if not src.exists():
        print(f"ERROR: domain dir not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Work on a throwaway copy so the suite never pollutes a real domain's
    # results.tsv or blackboard.
    tmp = Path(tempfile.mkdtemp(prefix="gemini-smoke-"))
    domain = tmp / src.name
    shutil.copytree(src, domain)
    shutil.rmtree(domain / "workspace", ignore_errors=True)
    results = domain / "results.tsv"
    if results.exists():
        results.chmod(0o644)
        results.write_text(results.read_text().splitlines()[0] + "\n")

    print(f"gemini agent mechanics suite — model={args.model}")
    print(f"  source domain: {src}")
    print(f"  scratch copy:  {domain}")

    t0_model_live(args.model)
    editable = t1_domain_wiring(domain)
    if editable:
        t2_guards(domain, editable)
        if args.live:
            t3_live_turn(domain, args.model)
    else:
        bad("skipping guard and live tests — domain wiring is broken")

    print(f"\n{'='*56}")
    print(f"passed: {PASS}   failed: {FAIL}")
    for f in FAILURES:
        print(f"  - {f}")
    if FAIL == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        print("SAFE TO LAUNCH")
    else:
        print(f"DO NOT LAUNCH — scratch copy kept at {domain}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

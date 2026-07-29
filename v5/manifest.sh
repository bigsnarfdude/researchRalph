#!/bin/bash
# manifest.sh — write a durable record of what condition a run happened under.
#
# Today a completed run leaves no trace of its own provenance. GUARD on/off is
# never written anywhere; .oracle_hash, GUARD_HALT, and runs/ are all
# gitignored, so once a domain dir is cleaned up there is nothing left to say
# "this run had the guard on, against this oracle, on this commit, with these
# frozen params." Two runs that produced different scores can't be told apart
# after the fact — was it the code, the params, or the guard mode? This writes
# one JSON file per run directory that answers that, independent of anything
# else being retained.
#
# Subcommands:
#   write <domain_dir> [key=value ...]   write <domain_dir>/RUN_MANIFEST.json
#   show  <domain_dir>                   pretty-print the existing manifest
#
# `write` never silently clobbers a prior manifest — if one exists it is
# renamed to RUN_MANIFEST.json.<epoch>.bak first, so re-running write mid-loop
# (e.g. once per cell/condition) doesn't destroy the first run's record.
#
# [key=value ...] on the command line is how the caller records what varies
# per invocation — cell id, factor levels, condition name — without this
# script having to know the experiment design. It lands verbatim in "extra".
#
# Exit: 0 = ok, 1 = usage/missing manifest.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
CMD="${1:-}"; shift 2>/dev/null || true

case "$CMD" in
# ------------------------------------------------------------------ write
write)
    DOMAIN_DIR="${1:?domain_dir}"; shift
    DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
    MANIFEST="$DOMAIN_DIR/RUN_MANIFEST.json"

    if [ -f "$MANIFEST" ]; then
        BAK="$MANIFEST.$(date +%s).bak"
        mv "$MANIFEST" "$BAK"
        echo "[manifest] existing manifest found — moved to $BAK"
    fi

    # Bash's job ends at arg collection ($@ here is the caller's key=value
    # pairs). All data gathering (git, hashing, yaml, env) happens in python
    # so JSON is built with json.dumps, never string-glued in bash.
    python3 - "$REPO" "$DOMAIN_DIR" "$MANIFEST" "$@" <<'PY'
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

repo, domain_dir, manifest_path = sys.argv[1], sys.argv[2], sys.argv[3]
extra_args = sys.argv[4:]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
        return out.strip()
    except Exception:
        return None


def strip_comment(v):
    # Strip a trailing `# ...` comment, but only outside quotes — a value
    # like K_schedule: "cosine # not a comment" must survive intact.
    in_s = in_d = False
    for i, ch in enumerate(v):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return v[:i]
    return v


def normalize(v):
    if v is None:
        return None
    v = v.strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1]
    return v.strip()


def yaml_top_value(path, key):
    if not os.path.isfile(path):
        return None
    pat = re.compile(r"^%s:\s*(.*)$" % re.escape(key))
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.match(line.rstrip("\n"))
            if m:
                return strip_comment(m.group(1)).strip()
    return None


def typed_value(raw):
    # Canonical value as a JSON-native type: ints/floats as numbers (so
    # 0.3 / 0.30 / .30 / 3e-1 are recognizably the same value), everything
    # else as a de-quoted string.
    v = normalize(raw)
    if v is None:
        return None
    if re.fullmatch(r"[+-]?\d+", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


# ---- git provenance of the researchRalph repo (not the domain) ----
sha = git("rev-parse", "HEAD")
branch = git("rev-parse", "--abbrev-ref", "HEAD")
status = git("status", "--porcelain")
dirty = bool(status) if status is not None else None

# ---- domain identity ----
domain_name = os.path.basename(domain_dir.rstrip("/"))
cfg = os.path.join(domain_dir, "config.yaml")
domain_type = normalize(yaml_top_value(cfg, "domain_type"))
editable = normalize(yaml_top_value(cfg, "editable"))
frozen_params_raw = yaml_top_value(cfg, "frozen_params")
frozen_params = frozen_params_raw.split() if frozen_params_raw else []

# ---- oracle file hashes (the files a run.sh / solve.py / engine.py oracle
# is made of — matches the "never edit" list every domain already declares) ----
oracle = {}
for fname in ("run.sh", "solve.py", "engine.py"):
    fpath = os.path.join(domain_dir, fname)
    if os.path.isfile(fpath):
        oracle[fname] = sha256_of(fpath)

# ---- prompt file hashes ----
prompts = {}
for fname in ("program_static.md", "program.md", "chaos_prompt.md", "worker_prompt.md"):
    fpath = os.path.join(domain_dir, fname)
    if os.path.isfile(fpath):
        prompts[fname] = sha256_of(fpath)

# ---- canonical values of the frozen params, from the same config.yaml ----
frozen_param_canonical = {}
for p in frozen_params:
    frozen_param_canonical[p] = typed_value(yaml_top_value(cfg, p))

# ---- env: presence AND absence both matter, so unset -> null, not omitted ----
env = {
    k: os.environ.get(k)
    for k in ("MODEL", "GUARD", "GUARD_MODE", "RRMA_PREFIX", "CLAUDE_AGENT_ID")
}

# ---- extra key=value pairs from the command line ----
extra = {}
for kv in extra_args:
    if "=" not in kv:
        continue
    k, v = kv.split("=", 1)
    extra[k] = v

manifest = {
    "schema_version": "1.0",
    "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "whoami": getpass.getuser(),
    "git": {"sha": sha, "branch": branch, "dirty": dirty},
    "domain": {
        "name": domain_name,
        "domain_type": domain_type,
        "editable": editable,
        "frozen_params": frozen_params,
    },
    "oracle": oracle,
    "prompts": prompts,
    "frozen_param_canonical": frozen_param_canonical,
    "env": env,
    "extra": extra,
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=False)
    f.write("\n")

print(f"[manifest] wrote {manifest_path}")
PY
    exit $? ;;

# ------------------------------------------------------------------ show
show)
    DOMAIN_DIR="${1:?domain_dir}"
    DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
    MANIFEST="$DOMAIN_DIR/RUN_MANIFEST.json"
    if [ ! -f "$MANIFEST" ]; then
        echo "[manifest] no manifest at $MANIFEST" >&2
        exit 1
    fi
    python3 -m json.tool "$MANIFEST"
    exit $? ;;

*)
    echo "usage: manifest.sh {write|show} <domain_dir> [key=value ...]" >&2
    exit 1 ;;
esac

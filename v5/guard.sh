#!/bin/bash
# guard.sh — deterministic reward-hack / escape guards for RRMA islands.
#
# Motivated by the 2026-07-21 OpenAI/HF incident and RRMA's own observed
# harness-seeking: a narrowly-goaled agent routes around containment to reach
# a reachable shortcut. These guards are the cheap, model-free half of the
# defense (the other half is the SFT ingestion gate, bootstrap/verify_filter_ml.py).
#
# Subcommands (each exits 0 = ok, 3 = violation, 1 = usage/other):
#   oracle-snapshot <island> [--force]   record sha of the canonical oracle files
#   oracle-verify   <island>             fail if the oracle changed since snapshot
#   scan-trace      <session.jsonl> <island>   flag egress / out-of-scope actions
#   audit-contamination <island>         fail if the oracle's ground truth is reachable
#
# The oracle files guarded: run.sh, engine.py, and anything listed in
# config.yaml `oracle_files:` (space-separated). These must be immutable during
# a run — an agent that edits them can forge every score AND launder a hack
# through the SFT re-verification gate (which re-runs the same oracle).

set -u
CMD="${1:-}"; shift 2>/dev/null || true

sha_of() { command -v sha256sum >/dev/null 2>&1 && sha256sum "$@" 2>/dev/null || shasum -a 256 "$@" 2>/dev/null; }

oracle_files() {  # echoes the canonical oracle file paths for an island
    local ISL="$1" extra
    for f in run.sh engine.py; do [ -f "$ISL/$f" ] && echo "$ISL/$f"; done
    extra="$(grep '^oracle_files:' "$ISL/config.yaml" 2>/dev/null | cut -d: -f2-)"
    for f in $extra; do [ -f "$ISL/$f" ] && echo "$ISL/$f"; done
}

case "$CMD" in
# ------------------------------------------------------------------ oracle hash
oracle-snapshot)
    ISL="$(cd "${1:?island}" && pwd)"; FORCE="${2:-}"
    SNAP="$ISL/.oracle_hash"
    if [ -f "$SNAP" ] && [ "$FORCE" != "--force" ]; then
        echo "[guard] oracle snapshot already exists (use --force to reset): $SNAP"; exit 0
    fi
    oracle_files "$ISL" | sort | while read -r f; do sha_of "$f"; done | sed "s| $ISL/| |" > "$SNAP"
    echo "[guard] oracle snapshot written: $(wc -l < "$SNAP" | tr -d ' ') files -> $SNAP"; exit 0 ;;

oracle-verify)
    ISL="$(cd "${1:?island}" && pwd)"
    SNAP="$ISL/.oracle_hash"
    if [ ! -f "$SNAP" ]; then echo "[guard] FAIL: no oracle snapshot — run oracle-snapshot first" >&2; exit 1; fi
    NOW="$(oracle_files "$ISL" | sort | while read -r f; do sha_of "$f"; done | sed "s| $ISL/| |")"
    if [ "$NOW" = "$(cat "$SNAP")" ]; then
        echo "[guard] oracle-verify ok ($(wc -l < "$SNAP" | tr -d ' ') files unchanged)"; exit 0
    fi
    echo "[guard] VIOLATION: oracle files changed since snapshot — score integrity broken" >&2
    diff <(cat "$SNAP") <(echo "$NOW") | sed 's/^/[guard]   /' >&2
    exit 3 ;;

# ------------------------------------------------------------------ trace scan
scan-trace)
    SLOG="${1:?session.jsonl}"; ISL="$(cd "${2:?island}" && pwd)"
    [ -s "$SLOG" ] || { echo "[guard] scan-trace: empty/no trace ($SLOG) — nothing to scan"; exit 0; }
    python3 - "$SLOG" "$ISL" <<'PY'
import json, re, sys
slog, isl = sys.argv[1], sys.argv[2].rstrip("/")
isl_name = isl.rsplit("/", 1)[-1]
# base domain = island name minus -isl-<suffix>, to catch sibling-island reads
base = re.sub(r"-isl-[a-z]$", "", isl_name)

EGRESS = re.compile(r"\b(curl|wget|scp|sftp|ssh|nc|ncat|netcat|telnet|rsync|ftp)\b|/dev/tcp/|\bsocket\.\b|\brequests\.(get|post)\b|\burllib\b", re.I)
PRIV   = re.compile(r"\bsudo\b|\bchmod\s+[0-7]*[7]{1}[0-7]*\s+.*results\.tsv|\bchown\b|/etc/|~/\.ssh|id_rsa|authorized_keys|\.aws/|\.config/gcloud", re.I)
# writes/reads outside this island: absolute paths not under the island, or other islands/domains
OUT    = re.compile(r"(^|[\s'\"><|])(/(?!tmp/)[A-Za-z]\S*)")

def commands(path):
    for line in open(path, errors="replace"):
        try: d = json.loads(line)
        except: continue
        msg = d.get("message") or {}
        for b in (msg.get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                name = b.get("name", ""); inp = b.get("input", {}) or {}
                if name == "Bash":
                    yield ("bash", inp.get("command", ""))
                elif name in ("Write", "Edit", "NotebookEdit"):
                    yield ("write", inp.get("file_path", ""))
                elif name == "Read":
                    yield ("read", inp.get("file_path", ""))

viol = []
for kind, arg in commands(slog):
    if not arg: continue
    if kind == "bash":
        if EGRESS.search(arg): viol.append(("network-egress", arg[:160]))
        if PRIV.search(arg):   viol.append(("privilege/credential", arg[:160]))
        # out-of-island absolute-path targets in the command
        for m in OUT.finditer(arg):
            p = m.group(2)
            if not p.startswith(isl) and base and (base in p or p.startswith(("/home","/Users","/data","/etc","/root"))):
                # ignore reads of shared read-only tooling paths under the repo v4/ v5/
                if "/v4/" in p or "/v5/" in p or "/tools/" in p: continue
                viol.append(("out-of-island-path", p)); break
    elif kind in ("write", "read"):
        p = arg
        if p.startswith("/") and not p.startswith(isl) and not p.startswith("/tmp"):
            # sibling island or other domain?
            tag = "sibling-island-access" if (base and f"{base}-isl-" in p) else f"out-of-island-{kind}"
            viol.append((tag, p))

if not viol:
    print("[guard] scan-trace clean"); sys.exit(0)
seen = set()
print("[guard] VIOLATION: out-of-scope actions in session trace:", file=sys.stderr)
for cat, ev in viol:
    k = (cat, ev)
    if k in seen: continue
    seen.add(k)
    print(f"[guard]   {cat}: {ev}", file=sys.stderr)
sys.exit(3)
PY
    exit $? ;;

# ------------------------------------------------------------------ contamination
audit-contamination)
    ISL="$(cd "${1:?island}" && pwd)"
    # Domain declares what the agent must never be able to read: a file glob of
    # ground-truth/answer data (`ground_truth:`) and/or a sentinel string that
    # should never appear in an agent-reachable file (`sensitive_pattern:`).
    GT="$(grep '^ground_truth:' "$ISL/config.yaml" 2>/dev/null | cut -d: -f2- | xargs 2>/dev/null)"
    SENT="$(grep '^sensitive_pattern:' "$ISL/config.yaml" 2>/dev/null | cut -d: -f2- | xargs 2>/dev/null)"
    if [ -z "$GT" ] && [ -z "$SENT" ]; then
        echo "[guard] FAIL: no ground_truth: or sensitive_pattern: declared in config.yaml — cannot audit contamination (declare them; fail-loud by design)" >&2
        exit 1
    fi
    RC=0
    # (1) ground-truth files must live OUTSIDE the island (agent cwd tree)
    for g in $GT; do
        for hit in "$ISL"/$g $ISL/workspace/**/$g; do
            [ -e "$hit" ] || continue
            echo "[guard] VIOLATION: ground-truth reachable inside island: $hit" >&2; RC=3
        done
    done
    # (2) no agent-reachable file may contain the sentinel (excluding the oracle itself)
    if [ -n "$SENT" ]; then
        while IFS= read -r hit; do
            # skip the oracle and the config.yaml where the pattern is DECLARED
            case "$hit" in */engine.py|*/run.sh|*/.oracle_hash|*/config.yaml) continue;; esac
            echo "[guard] VIOLATION: sensitive pattern reachable in: $hit" >&2; RC=3
        done < <(grep -rlF "$SENT" "$ISL/workspace" "$ISL"/*.md "$ISL"/*.yaml 2>/dev/null)
    fi
    [ "$RC" -eq 0 ] && echo "[guard] audit-contamination ok (ground truth not reachable from agent tree)"
    exit $RC ;;

*)
    echo "usage: guard.sh {oracle-snapshot|oracle-verify|scan-trace|audit-contamination} ..." >&2
    exit 1 ;;
esac

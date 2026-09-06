#!/usr/bin/env bash
# rental-bootstrap.sh — self-contained rerun of the erdos-125 reverse ladder on a
# rented Linux box. Written 2026-09-06 after the audit found that every rung of the
# 26 May ladder ran against a run.sh that does not parse (broken by d923460, the
# reward-hacking-prevention commit, 17 minutes before the ladder started).
#
# Self-contained means: nothing here depends on the laptop staying reachable.
# The box clones from GitHub, builds its own Lean, runs, and pushes results back
# to GitHub itself. A dropped SSH session costs nothing — everything runs in tmux.
#
# USAGE (on a fresh box, one paste):
#   export ANTHROPIC_API_KEY=...    # metered API billing, not a subscription
#   export GH_TOKEN=...             # repo scope, for pushing results back
#   B=fix/erdos-125-oracle-repair   # or 'main' once this is merged
#   curl -fsSL "https://raw.githubusercontent.com/bigsnarfdude/researchRalph/$B/v4/rental-bootstrap.sh" -o bootstrap.sh
#   RRMA_SRC_BRANCH=$B tmux new -s rrma "bash bootstrap.sh 2>&1 | tee ~/bootstrap.log"
#   # then safely: Ctrl-b d, close the laptop, come back later
#
# Secrets are read from the environment and never written into the repo.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?export ANTHROPIC_API_KEY first}"
: "${GH_TOKEN:?export GH_TOKEN first (repo scope)}"

REPO_URL="${RRMA_REPO:-github.com/bigsnarfdude/researchRalph}"
# Branch carrying the REPAIRED oracles. Until that work lands on main, cloning the
# default branch gets the unparseable run.sh that this whole rerun exists to escape.
SRC_BRANCH="${RRMA_SRC_BRANCH:-fix/erdos-125-oracle-repair}"
# Branch the results are pushed to (created from SRC_BRANCH).
BRANCH="${RRMA_BRANCH:-rerun/reverse-ladder-$(date -u +%Y%m%d)}"
MODEL="${RRMA_MODEL:-claude-sonnet-5}"
REPS="${RRMA_REPS:-3}"
TOOLCHAIN="${RRMA_TOOLCHAIN:-leanprover/lean4:v4.29.0-rc8}"
LEAN_PROJECT="${RRMA_LEAN_PROJECT:-$HOME/rrma-lean}"
WALL_CAP_MIN="${RRMA_WALL_CAP_MIN:-45}"     # per-rung wall-clock budget
AGENTS=2; TURNS=200; MONITOR=10; GENS=1

# The four rungs worth rerunning. 01 and 02 are void (harness ablations that never
# fired); 08 and 09 were pre-registered as the LOWEST-impact ablations and both
# measured 0% in 3-11 minute windows. 03/04/06/07 gave usable signal — left alone.
RUNGS="${RRMA_RUNGS:-erdos-125-abl-01-oracle erdos-125-abl-02-workspace erdos-125-abl-08-desires erdos-125-abl-09-learnings}"

# Rungs whose ablation is IN THE HARNESS: preflight's live oracle test must fail
# on them by design (01 kills the oracle at zero sorries; 02 reads the domain root
# rather than workspace/agent0). Skipping the gate for these is correct, not a
# shortcut — for every other rung the gate stays armed.
HARNESS_ABLATED="erdos-125-abl-01-oracle erdos-125-abl-02-workspace"

STATUS="$HOME/run_status.log"
say(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# ---------------------------------------------------------------- 1. system deps
say "stage 1/6 — system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl git tmux python3 python3-pip \
                           util-linux nodejs npm >/dev/null
command -v flock >/dev/null || { say "FATAL: flock missing"; exit 1; }

# ------------------------------------------------------------------ 2. claude CLI
say "stage 2/6 — claude CLI"
if ! command -v claude >/dev/null; then
  sudo npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 || \
    npm install -g @anthropic-ai/claude-code
fi
export PATH="$HOME/.local/bin:$(npm bin -g 2>/dev/null || echo /usr/local/bin):$PATH"
claude --version | tee -a "$STATUS"

# ------------------------------------------------------------- 3. Lean + Mathlib
say "stage 3/6 — Lean $TOOLCHAIN + Mathlib (the long pole)"
if [ ! -d "$HOME/.elan" ]; then
  curl -fsSL https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain "$TOOLCHAIN"
fi
export PATH="$HOME/.elan/bin:$PATH"
elan toolchain install "$TOOLCHAIN"

if [ ! -d "$LEAN_PROJECT" ]; then
  cd "$HOME"
  lake +"$TOOLCHAIN" new "$(basename "$LEAN_PROJECT")" math
fi
cd "$LEAN_PROJECT"
# Optional Mathlib pin. UNSET BY DEFAULT: the seed proofs were written against a
# May-2026 Mathlib, and renames since (abs_add -> abs_add_le, per LEARNINGS.md)
# can fail them for reasons unrelated to the ablation. Preflight compiles the seed
# and will say so loudly if drift bit.
if [ -n "${RRMA_MATHLIB_REV:-}" ]; then
  say "pinning Mathlib to $RRMA_MATHLIB_REV"
  (cd .lake/packages/mathlib && git fetch --all -q && git checkout -q "$RRMA_MATHLIB_REV")
fi
lake exe cache get   # prebuilt oleans; falls back to a source build if unavailable
lake build
say "Lean ready: $(lean --version)"

# ------------------------------------------------------------------- 4. the repo
say "stage 4/6 — repo"
cd "$HOME"
[ -d researchRalph ] || git clone -q "https://${GH_TOKEN}@${REPO_URL}" researchRalph
cd researchRalph
git config user.name  "bigsnarfdude"
git config user.email "ohprecio@gmail.com"
git remote set-url origin "https://${GH_TOKEN}@${REPO_URL}"
git checkout -q -B "$BRANCH"
export RRMA_LEAN_PROJECT="$LEAN_PROJECT"

# ------------------------------------------------------------------- 5. preflight
# Canary on a rung whose harness is NOT ablated. If this fails, nothing launches —
# that is the whole point (26 May had no such gate).
say "stage 5/6 — preflight canary on erdos-125-abl-08-desires"
if ! bash v4/preflight.sh domains/erdos-125-abl-08-desires 2>&1 | tee -a "$STATUS"; then
  say "FATAL: preflight failed. Not launching. Most likely cause: Mathlib drift"
  say "       broke the seed proof. Set RRMA_MATHLIB_REV and re-run stage 3."
  exit 1
fi

# --------------------------------------------------------------------- 6. the runs
# Archival is chained onto each rung with && per REMOTE_RUN_PROTOCOL.md, so results
# reach GitHub the moment a rung finishes rather than when a human notices.
archive(){
  local rung="$1" rep="$2" stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local d="domains/$rung"
  # workspace/ and logs/ are gitignored, and they are exactly what an integrity
  # audit needs (abl-02's bypass was only visible in the workspace files).
  # Tar them so the ignore rules do not silently discard the evidence.
  tar czf "$d/archive-rep${rep}-${stamp}.tgz" \
      -C "$d" workspace logs 2>/dev/null || true
  git add -A "$d" 2>/dev/null || true
  git add -f "$d/results.tsv" "$d/blackboard.md" 2>/dev/null || true
  git commit -q -m "rerun: $rung rep$rep ($stamp)

Reverse-ladder rerun on repaired oracle. Original 26 May run executed against a
run.sh that does not parse; these rows are the first from a verified harness.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QdAnE5Hjw7C9J1vkXCqpnQ" || true
  git push -q -u origin "$BRANCH" \
    && echo "ARCHIVED $rung rep$rep $stamp" >> "$STATUS"
}

say "stage 6/6 — running ${REPS} reps x $(echo $RUNGS | wc -w) rungs"
for rep in $(seq 1 "$REPS"); do
  for rung in $RUNGS; do
    say "=== $rung rep$rep ==="
    # Reset to a clean rung state so reps are independent.
    rm -rf "domains/$rung/workspace" "domains/$rung/logs"
    chmod 644 "domains/$rung/results.tsv" 2>/dev/null || true
    printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > "domains/$rung/results.tsv"

    SKIP=0
    for h in $HARNESS_ABLATED; do [ "$rung" = "$h" ] && SKIP=1; done

    RRMA_PREFIX="$(echo "$rung" | tr -d 'a-z-')r$rep" \
    RRMA_SKIP_PREFLIGHT=$SKIP \
    timeout "${WALL_CAP_MIN}m" \
      bash v4/outer-loop.sh "domains/$rung" $GENS $AGENTS $TURNS $MONITOR "$MODEL" \
      >> "$HOME/${rung}-rep${rep}.log" 2>&1 || say "$rung rep$rep ended (timeout or stop)"

    archive "$rung" "$rep"
  done
done

say "DONE — all rungs archived to branch $BRANCH"
say "Verify from the laptop:  git fetch origin && git log --oneline origin/$BRANCH"

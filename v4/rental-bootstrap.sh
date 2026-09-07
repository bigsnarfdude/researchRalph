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
#   # stages 1-5 build the box and gate on preflight; stage 6 execs v4/ladder-rerun.sh.
#   # RRMA_BOOTSTRAP_ONLY=1 stops after stage 5 so you can launch the run by hand.
#   # then safely: Ctrl-b d, close the laptop, come back later
#
# Secrets are read from the environment and never written into the repo.

set -euo pipefail

# ANTHROPIC_API_KEY is optional: if the box has already been authenticated
# interactively (`ssh -t <box> claude`, subscription auth), leave it unset. Stage 2
# verifies auth either way before anything is queued.
: "${GH_TOKEN:?export GH_TOKEN first (repo scope)}"

REPO_URL="${RRMA_REPO:-github.com/bigsnarfdude/researchRalph}"
# Branch carrying the REPAIRED oracles. Until that work lands on main, cloning the
# default branch gets the unparseable run.sh that this whole rerun exists to escape.
SRC_BRANCH="${RRMA_SRC_BRANCH:-fix/erdos-125-oracle-repair}"
# Branch the results are pushed to (created from SRC_BRANCH).
BRANCH="${RRMA_BRANCH:-rerun/reverse-ladder-$(date -u +%Y%m%d)}"
MODEL="${RRMA_MODEL:-claude-haiku-4-5-20251001}"   # the May ladder ran haiku; match it
TOOLCHAIN="${RRMA_TOOLCHAIN:-leanprover/lean4:v4.29.0-rc8}"
LEAN_PROJECT="${RRMA_LEAN_PROJECT:-$HOME/rrma-lean}"
# Queue, reps, caps, and the per-rep seed restore all live in v4/ladder-rerun.sh,
# which stage 6 hands off to. Nothing about the runs is configured here.

STATUS="$HOME/run_status.log"
say(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# ---------------------------------------------------------------- 1. system deps
say "stage 1/6 — system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl git tmux python3 python3-pip \
                           util-linux nodejs npm >/dev/null
command -v flock >/dev/null || { say "FATAL: flock missing"; exit 1; }

# ------------------------------------------------------------------ 2. claude CLI
# Native installer, NOT npm. Ubuntu 24.04 ships nodejs v12; Claude Code needs 18+,
# so `npm install -g @anthropic-ai/claude-code` fails and, under `set -e`, takes the
# whole bootstrap down with it — observed on the 2026-09-06 A10 box, which is why
# Lean never started on the first attempt. The native installer ships a prebuilt
# binary into ~/.local/bin and never touches node.
say "stage 2/6 — claude CLI (native installer)"
if ! command -v claude >/dev/null; then
  curl -fsSL https://claude.ai/install.sh | bash
fi
export PATH="$HOME/.local/bin:$PATH"
command -v claude >/dev/null || { say "FATAL: claude CLI not installed"; exit 1; }
claude --version | tee -a "$STATUS"

# Auth: interactive `claude` login (subscription) or ANTHROPIC_API_KEY (metered).
# Verify headlessly here — this is the exact mode the workers use, and finding out
# it is unauthenticated after twelve runs have been queued is expensive.
if ! timeout 60 claude -p "Reply with exactly: AUTH_OK" 2>&1 | grep -q AUTH_OK; then
  say "FATAL: claude is not authenticated. Run 'ssh -t <box> claude' and log in,"
  say "       or export ANTHROPIC_API_KEY, then re-run."
  exit 1
fi
say "claude auth: OK"

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
# The May workspaces are tracked in git; preflight prefers an existing workspace file
# over the seed and would test a four-month-old artifact (it did, on 2026-09-06).
rm -rf domains/erdos-125-abl-08-desires/workspace
if ! bash v4/preflight.sh domains/erdos-125-abl-08-desires 2>&1 | tee -a "$STATUS"; then
  say "FATAL: preflight failed. Not launching. Most likely cause: Mathlib drift"
  say "       broke the seed proof. Set RRMA_MATHLIB_REV and re-run stage 3."
  exit 1
fi

# --------------------------------------------------------------------- 6. the runs
# Hand off to the protocol script. Its per-rep restore to the designed seed is what
# makes the reps valid — the loop that used to live here reset workspace/ but not
# the domain root, and run.sh promotes winners into the root, so every rep after a
# win started from the answer (found by /sane on 2026-09-06). Never reimplement it.
say "stage 6/6 — handing off to v4/ladder-rerun.sh"
umask 077; printf 'export GH_TOKEN=%s\n' "$GH_TOKEN" > "$HOME/.rrma_env"; chmod 600 "$HOME/.rrma_env"
export RRMA_RESULTS_BRANCH="$BRANCH" RRMA_MODEL="$MODEL" RRMA_LEAN_PROJECT="$LEAN_PROJECT"
if [ "${RRMA_BOOTSTRAP_ONLY:-0}" = "1" ]; then
  say "RRMA_BOOTSTRAP_ONLY=1 — box is ready; launch with: tmux new -s v3 'bash ~/researchRalph/v4/ladder-rerun.sh 2>&1 | tee ~/v3.log'"
  exit 0
fi
exec bash "$HOME/researchRalph/v4/ladder-rerun.sh"

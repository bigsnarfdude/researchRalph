#!/bin/bash
# outer-loop.sh — the gardener (RRMA v4.6 outer agent)
#
# Runs meta-RRMA generations. Monitors process quality. Stops, diagnoses,
# and redesigns the scaffold between generations. Replaces the human.
#
# v4.6 changes:
#   - Refreshes stoplight.md + recent_experiments.md in monitor loop
#   - NUDGE/REDESIGN prompts use stoplight (30 lines) instead of raw blackboard tail
#   - Agents read program_static.md + program.md (split context)
#
# Usage: bash outer-loop.sh /path/to/domain [max_generations] [num_agents] [max_turns] [monitor_interval_min]
#
# The outer agent:
#   1. Calibrates (literature search for known results)
#   2. Launches meta-RRMA (workers + meta-agent)
#   3. Monitors process quality every N minutes
#   4. Stops when hacking/done/redesign triggered
#   5. On REDESIGN: edits scaffold, re-launches
#   6. On STOP_DONE: generates final artifacts
#   7. Appends lessons to taste.md
#
# Requires: claude CLI, screen

set -euo pipefail

DOMAIN_DIR="${1:-.}"
MAX_GENERATIONS="${2:-5}"
NUM_AGENTS="${3:-4}"
MAX_TURNS="${4:-200}"
MONITOR_INTERVAL="${5:-20}"  # minutes between diagnose checks
MODEL="${6:-}"  # e.g. claude-opus-4-6 (passed through to launch-agents.sh)

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASTE="$SCRIPT_DIR/taste.md"

# Ensure claude is on PATH (nigel needs login shell paths)
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

LOG="$DOMAIN_DIR/outer-loop.log"

log() {
    local msg="[outer $(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG"
}

# Signal workers to re-read blackboard.md. Each agent deletes its own flag
# after re-reading, so one agent acting doesn't hide the signal from the rest.
notify_agents() {
    for ws in "$DOMAIN_DIR"/workspace/agent*/; do
        [ -d "$ws" ] && touch "${ws}BLACKBOARD_UPDATED"
    done
}

# Screen session prefix — wrappers set RRMA_PREFIX to run concurrent fleets
export RRMA_PREFIX="${RRMA_PREFIX:-rrma}"

# --- RH Prevention: lock results.tsv so only run.sh (oracle) can write ---
touch "$DOMAIN_DIR/results.tsv"
chmod 444 "$DOMAIN_DIR/results.tsv"
log "results.tsv locked (read-only). Only oracle (run.sh) may write via flock."

# Select diagnoser: lean_proof domains use diagnose_lean.py
DOMAIN_TYPE=$(grep '^domain_type:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')
if [ "$DOMAIN_TYPE" = "lean_proof" ] && [ -f "$SCRIPT_DIR/diagnose_lean.py" ]; then
    DIAGNOSER="python3 $SCRIPT_DIR/diagnose_lean.py"
else
    DIAGNOSER="python3 $SCRIPT_DIR/diagnose.py"
fi

# --- Pre-flight checks ---
# Required files
for f in program.md config.yaml run.sh blackboard.md; do
    if [ ! -f "$DOMAIN_DIR/$f" ]; then
        echo "Error: missing $DOMAIN_DIR/$f"
        exit 1
    fi
done
# Optional files (ML domains have engine.py, sae.py — not all domains need them)
for f in engine.py sae.py; do
    [ ! -f "$DOMAIN_DIR/$f" ] && echo "Note: $f not present (optional)"
done

if [ ! -f "$TASTE" ]; then
    echo "Error: missing $TASTE"
    exit 1
fi

command -v claude >/dev/null 2>&1 || { echo "Error: claude CLI not found"; exit 1; }
command -v screen >/dev/null 2>&1 || { echo "Error: screen not found"; exit 1; }

# --- Automated deployment checklist (oracle reads workspace, logs to results.tsv,
#     prompt template exists). Skip with RRMA_SKIP_PREFLIGHT=1. ---
if [ "${RRMA_SKIP_PREFLIGHT:-0}" != "1" ]; then
    if ! bash "$SCRIPT_DIR/preflight.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"; then
        log "PREFLIGHT FAILED — refusing to launch agents. Fix the issues above (or set RRMA_SKIP_PREFLIGHT=1 to override)."
        exit 1
    fi
fi

log "=== RRMA v4.6 Outer Loop Starting ==="
log "Domain: $DOMAIN_DIR"
log "Max generations: $MAX_GENERATIONS"
log "Agents: $NUM_AGENTS, turns: $MAX_TURNS, monitor: ${MONITOR_INTERVAL}m, model: ${MODEL:-default}"
log ""

# --- Generation loop ---
for gen in $(seq 1 "$MAX_GENERATIONS"); do
    log "=== GENERATION $gen / $MAX_GENERATIONS ==="

    # Snapshot pre-generation state
    PRE_BEST=$(awk -F'\t' 'NR>1{print $2}' "$DOMAIN_DIR/results.tsv" 2>/dev/null | sort -rn | head -1 || true)
    PRE_BEST="${PRE_BEST:-0}"
    PRE_EXP=$(awk -F'\t' 'NR>1{n++} END{print n+0}' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)
    PRE_PQ="unknown"

    # --- Step 0: Calibrate (first generation only) ---
    if [ "$gen" -eq 1 ] && [ ! -f "$DOMAIN_DIR/calibration.md" ]; then
        log "Step 0: Calibrating via literature search..."
        bash "$SCRIPT_DIR/calibrate.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"
    fi

    # --- Step 1: Launch meta-RRMA ---
    log "Step 1: Launching $NUM_AGENTS workers + meta-agent..."
    bash "$SCRIPT_DIR/launch-agents.sh" "$DOMAIN_DIR" "$NUM_AGENTS" "$MAX_TURNS" 30 "$MODEL" 2>&1 | tee -a "$LOG"

    # --- Step 2: Monitor loop ---
    DECISION="CONTINUE"
    MONITOR_COUNT=0

    NUDGE_COUNT=0  # track how many nudges this generation

    while [ "$DECISION" = "CONTINUE" ] || [ "$DECISION" = "TOO_EARLY" ] || [ "$DECISION" = "NUDGE" ]; do
        log "Monitoring check in ${MONITOR_INTERVAL}m..."
        sleep $((MONITOR_INTERVAL * 60))
        MONITOR_COUNT=$((MONITOR_COUNT + 1))

        # Check if any workers are still alive
        ALIVE=$(screen -ls 2>/dev/null | grep -c "${RRMA_PREFIX}-worker" || true)
        ALIVE="${ALIVE:-0}"

        # Zero-oracle watchdog: workers alive but no experiment ever logged this
        # generation means agents are burning turns without calling run.sh
        # (the erdos-125 failure mode: 300+ turns, $14, 0 experiments).
        # Lean oracles respond in minutes (2 checks); ML training runs need
        # longer before the first row lands (4 checks). Override with
        # RRMA_WATCHDOG_CHECKS.
        if [ "$DOMAIN_TYPE" = "lean_proof" ]; then
            WATCHDOG_CHECKS="${RRMA_WATCHDOG_CHECKS:-2}"
        else
            WATCHDOG_CHECKS="${RRMA_WATCHDOG_CHECKS:-4}"
        fi
        CUR_EXP=$(awk -F'\t' 'NR>1{n++} END{print n+0}' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)
        if [ "$ALIVE" -gt 0 ] && [ "$CUR_EXP" -le "$PRE_EXP" ] && [ "$MONITOR_COUNT" -ge "$WATCHDOG_CHECKS" ]; then
            log "WATCHDOG: $MONITOR_COUNT checks ($((MONITOR_COUNT * MONITOR_INTERVAL))m) with $ALIVE workers alive and ZERO new experiments in results.tsv."
            log "WATCHDOG: agents are not calling the oracle. Stopping run before more budget burns."
            log "WATCHDOG: check that run.sh works ('CLAUDE_AGENT_ID=agent0 bash run.sh') and that the worker prompt tells agents how to call it."
            bash "$SCRIPT_DIR/stop-agents.sh" 2>&1 | tee -a "$LOG"
            exit 1
        fi

        if [ "$ALIVE" -eq 0 ]; then
            log "All workers finished (used all $MAX_TURNS turns). Running final diagnosis."
            DECISION=$($DIAGNOSER "$DOMAIN_DIR" 2>>"$LOG")
            # If workers are done, treat CONTINUE/NUDGE as STOP_DONE
            if [ "$DECISION" = "CONTINUE" ] || [ "$DECISION" = "TOO_EARLY" ] || [ "$DECISION" = "NUDGE" ]; then
                DECISION="STOP_DONE"
            fi
            break
        fi

        # v4.6: Refresh context files before diagnosis
        python3 "$REPO_ROOT/tools/refresh_context.py" "$DOMAIN_DIR" 2>>"$LOG" || true

        # Run diagnosis
        DECISION=$($DIAGNOSER "$DOMAIN_DIR" 2>>"$LOG")
        log "Diagnosis: $DECISION (check $MONITOR_COUNT, workers alive: $ALIVE)"

        # --- v4.5: Handle NUDGE (observation + program.md patch) ---
        if [ "$DECISION" = "NUDGE" ]; then
            NUDGE_COUNT=$((NUDGE_COUNT + 1))
            log "=== NUDGE #$NUDGE_COUNT ==="

            if [ "$NUDGE_COUNT" -ge 3 ]; then
                # 3 nudges without progress → escalate to REDESIGN
                log "3 nudges without progress. Escalating to REDESIGN."
                DECISION="REDESIGN"
            else
                # Read structured nudge data from diagnose.py
                NUDGE_JSON="$DOMAIN_DIR/.nudge_data.json"

                # Generate observation + program.md constraints from TrustLoop data
                NUDGE_PROMPT="$(cat <<NUDGE_EOF
You are the gardener in an RRMA research system. TrustLoop has diagnosed issues that need fixing.

## TrustLoop nudge data:
$(cat "$NUDGE_JSON" 2>/dev/null || echo "{}")

## Current program.md:
$(cat "$DOMAIN_DIR/program.md")

## Stoplight (compressed run state):
$(cat "$DOMAIN_DIR/stoplight.md" 2>/dev/null || tail -40 "$DOMAIN_DIR/blackboard.md")

## Recent experiments:
$(cat "$DOMAIN_DIR/recent_experiments.md" 2>/dev/null || tail -10 "$DOMAIN_DIR/results.tsv")

## Your job — TWO outputs separated by ===CONSTRAINTS===

PART 1: Write ONE observation (2-3 sentences) for the blackboard noting what's stuck and why.

===CONSTRAINTS===

PART 2: Write constraints to APPEND to program.md. These must be concrete and actionable:
- If dead_ends are listed: "Do NOT attempt [design] experiments — 0 keeps in N attempts."
- If dominant_axis is set: "You have exhausted [axis]. Explore [alternative axis] instead. Look at [specific code section]."
- If missed_checks show gaps: "CONSTRAINT: [the specific rule from the lesson]"
- If gardener_fixes are listed: add each fix as a constraint.

Write ONLY the constraint lines (one per line, starting with "- "). If no constraints needed, write "NONE".
Do NOT rewrite program.md. Only output text to append.
NUDGE_EOF
)"

                NUDGE_OUTPUT=$(claude -p "$NUDGE_PROMPT" --dangerously-skip-permissions --max-turns 3 2>/dev/null)

                if [ -n "$NUDGE_OUTPUT" ]; then
                    # Split on ===CONSTRAINTS===
                    OBSERVATION=$(echo "$NUDGE_OUTPUT" | sed '/===CONSTRAINTS===/,$d')
                    CONSTRAINTS=$(echo "$NUDGE_OUTPUT" | sed '1,/===CONSTRAINTS===/d')

                    # Append observation to blackboard
                    if [ -n "$OBSERVATION" ]; then
                        echo "" >> "$DOMAIN_DIR/blackboard.md"
                        echo "## Observation [gardener, $(date '+%H:%M')]" >> "$DOMAIN_DIR/blackboard.md"
                        echo "$OBSERVATION" >> "$DOMAIN_DIR/blackboard.md"
                        notify_agents
                        log "Nudge observation: $(echo "$OBSERVATION" | head -1)"
                    fi

                    # Append constraints to program.md (if not NONE)
                    if [ -n "$CONSTRAINTS" ] && ! echo "$CONSTRAINTS" | grep -qi "^NONE$"; then
                        echo "" >> "$DOMAIN_DIR/program.md"
                        echo "## Constraints [gardener, $(date '+%Y-%m-%d %H:%M')]" >> "$DOMAIN_DIR/program.md"
                        echo "$CONSTRAINTS" >> "$DOMAIN_DIR/program.md"
                        log "Patched program.md with constraints: $(echo "$CONSTRAINTS" | wc -l | tr -d ' ') lines"
                    else
                        log "No program.md constraints needed"
                    fi
                else
                    log "Nudge generation failed - empty output"
                fi

                # Continue monitoring — NUDGE doesn't stop the run
                DECISION="CONTINUE"
            fi
        fi

        # Safety: don't monitor forever (max 48 hours)
        if [ "$MONITOR_COUNT" -gt 144 ]; then  # 144 * 20min = 48h
            log "Safety timeout (48h). Forcing stop."
            DECISION="STOP_DONE"
        fi
    done

    # --- Step 3: Stop workers ---
    log "Stopping workers (decision: $DECISION)..."
    bash "$SCRIPT_DIR/stop-agents.sh" 2>&1 | tee -a "$LOG"

    # --- Step 4: Post-generation metrics ---
    POST_BEST=$(awk -F'\t' 'NR>1{print $2}' "$DOMAIN_DIR/results.tsv" 2>/dev/null | sort -rn | head -1 || true)
    POST_BEST="${POST_BEST:-0}"
    POST_EXP=$(awk -F'\t' 'NR>1{n++} END{print n+0}' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)
    NEW_EXP=$((POST_EXP - PRE_EXP))

    bash "$SCRIPT_DIR/validate_claims.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"
    log "Generation $gen complete: $NEW_EXP new experiments, best=$POST_BEST (was $PRE_BEST)"

    # --- Step 5: Generate meta-blackboard distillation ---
    log "Generating meta-blackboard distillation..."
    bash "$SCRIPT_DIR/generate-meta-blackboard.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"

    # --- Step 6: Handle decision ---
    case "$DECISION" in
        STOP_HACKING)
            log "=== HACKING DETECTED — REDESIGNING SCAFFOLD ==="

            # Ask Claude to diagnose and fix program.md
            REDESIGN_PROMPT="$(cat <<PROMPT
You are the outer agent (the gardener) in an RRMA v4 self-recursive research system.

## Your taste (inherited principles):
$(cat "$TASTE")

## Situation:
The agents ran $NEW_EXP experiments but the process quality is LOW.
They are config-tuning or gaming the metric instead of doing real research.
Signs: few/no papers cited, few/no novel architectures, no ablation science,
no explanations of WHY things work.

## Current program.md:
$(cat "$DOMAIN_DIR/program.md")

## Stoplight (compressed run state):
$(cat "$DOMAIN_DIR/stoplight.md" 2>/dev/null || tail -100 "$DOMAIN_DIR/blackboard.md")

## Recent experiments:
$(cat "$DOMAIN_DIR/recent_experiments.md" 2>/dev/null || tail -20 "$DOMAIN_DIR/results.tsv")

## Your job:
Rewrite program.md to force genuine research. Specific changes to consider:
- Add explicit requirements: "cite at least one paper before implementing"
- Add: "explain WHY each approach should work before running it"
- Add: "run ablation experiments — if you add component X, also test without X"
- Make the gap between current score and plausible ceiling more visible
- Add hints toward research axes (not specific techniques — axes like
  "encoder architecture", "training curriculum", "loss function design")

Output ONLY the new program.md content. No commentary.
PROMPT
)"
            claude -p "$REDESIGN_PROMPT" --dangerously-skip-permissions --max-turns 3 > "$DOMAIN_DIR/program.md.new" 2>/dev/null

            # Backup and replace
            cp "$DOMAIN_DIR/program.md" "$DOMAIN_DIR/program.md.gen$gen"
            mv "$DOMAIN_DIR/program.md.new" "$DOMAIN_DIR/program.md"
            log "Rewrote program.md (backed up to program.md.gen$gen)"

            # Reset blackboard but keep meta-blackboard (cross-generation memory)
            cp "$DOMAIN_DIR/blackboard.md" "$DOMAIN_DIR/blackboard.md.gen$gen"
            cat > "$DOMAIN_DIR/blackboard.md" <<EOF
# Blackboard — $(basename "$DOMAIN_DIR")

Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.

## Previous generation summary
The previous generation's findings are in meta-blackboard.md. Read it.
EOF
            log "Reset blackboard (backed up to blackboard.md.gen$gen)"
            ;;

        REDESIGN)
            log "=== SCAFFOLD BLOCKING EXPLORATION — REDESIGNING ==="

            REDESIGN_PROMPT="$(cat <<PROMPT
You are the outer agent (the gardener) in an RRMA v4 self-recursive research system.

## Your taste (inherited principles):
$(cat "$TASTE")

## Situation:
The agents have HIGH process quality (doing real research) but scores are
FLAT and there are BLIND SPOTS — unexplored directions that agents haven't
reached. The scaffold is blocking exploration somehow.

## Current meta-blackboard.md (with blind spots):
$(cat "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null || echo "No meta-blackboard yet")

## Current program.md:
$(cat "$DOMAIN_DIR/program.md")

## Stoplight (compressed run state):
$(cat "$DOMAIN_DIR/stoplight.md" 2>/dev/null || tail -150 "$DOMAIN_DIR/blackboard.md")

## Recent experiments:
$(cat "$DOMAIN_DIR/recent_experiments.md" 2>/dev/null || tail -30 "$DOMAIN_DIR/results.tsv")

## Agent DESIRES (tools/context they wish they had):
$(cat "$DOMAIN_DIR/DESIRES.md" 2>/dev/null || echo "none")

## Agent MISTAKES (patterns of failure):
$(cat "$DOMAIN_DIR/MISTAKES.md" 2>/dev/null | tail -50 || echo "none")

## Agent LEARNINGS (discoveries worth preserving):
$(cat "$DOMAIN_DIR/LEARNINGS.md" 2>/dev/null | tail -50 || echo "none")

## Your job:
Diagnose why agents can't reach the blind spots. Common causes:
- program.md framing locks agents into one axis
- Turn budget too low for complex experiments
- Missing hint that a different research direction exists
- Agents need a planning prompt to step back and reassess

Make MINIMAL changes. Don't rewrite everything — identify the ONE thing
blocking exploration and fix it.

Output a JSON object:
{
  "diagnosis": "one sentence explaining the block",
  "change_type": "program_md | planning_trigger | hint",
  "change_description": "what you're changing and why",
  "new_program_md": "full new program.md content (or null if not changing it)",
  "add_to_blackboard": "text to append to blackboard (or null)"
}
PROMPT
)"
            claude -p "$REDESIGN_PROMPT" --dangerously-skip-permissions --max-turns 3 > "/tmp/redesign-gen$gen.json" 2>/dev/null

            # Parse the JSON and apply changes deterministically (no model calls)
            python3 "$SCRIPT_DIR/apply_redesign.py" "/tmp/redesign-gen$gen.json" "$DOMAIN_DIR" "$gen" 2>&1 | tee -a "$LOG"
            notify_agents
            ;;

        STOP_DONE)
            log "=== STOP_DONE TRIGGERED — CHECKING FOR UNEXPLORED DIRECTIONS ==="

            # Before accepting STOP_DONE, ask: is the search genuinely exhausted?
            REEVAL_PROMPT="$(cat <<REEVAL_EOF
You are reviewing a completed research run. Read the blackboard and results.

## Stoplight (compressed run state):
$(cat "$DOMAIN_DIR/stoplight.md" 2>/dev/null || tail -100 "$DOMAIN_DIR/blackboard.md")

## Recent experiments:
$(cat "$DOMAIN_DIR/recent_experiments.md" 2>/dev/null || tail -20 "$DOMAIN_DIR/results.tsv")

## Meta-blackboard:
$(cat "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null | head -50 || echo "none")

Answer these two questions:
1. Are there research directions that were NEVER tried? (e.g., training curriculum, loss functions, multi-scale methods, data scaling, if agents only did architecture)
2. Could the current best be significantly improved (>5%) with a different approach?

If YES to either: output a single line starting with UNEXPLORED: followed by 1-2 unexplored directions.
If NO to both: output a single line: EXHAUSTED

Output ONLY one line. No explanation.
REEVAL_EOF
)"

            REEVAL=$(claude -p "$REEVAL_PROMPT" --dangerously-skip-permissions --max-turns 1 2>/dev/null | head -1)
            log "Re-evaluation: $REEVAL"

            if echo "$REEVAL" | grep -q "^UNEXPLORED:"; then
                log "Unexplored directions found. Downgrading STOP_DONE to NUDGE."
                NUDGE_TEXT=$(echo "$REEVAL" | sed 's/^UNEXPLORED: //')
                echo "" >> "$DOMAIN_DIR/blackboard.md"
                echo "## Observation [gardener, $(date '+%H:%M') — before stopping]" >> "$DOMAIN_DIR/blackboard.md"
                echo "The search appears stalled. Unexplored directions: $NUDGE_TEXT" >> "$DOMAIN_DIR/blackboard.md"
                notify_agents

                # Don't stop — continue to next generation with the nudge
                log "Continuing to generation $((gen + 1)) with nudge applied."
                # Relaunch workers
                bash "$SCRIPT_DIR/launch-agents.sh" "$DOMAIN_DIR" "$NUM_AGENTS" "$MAX_TURNS" 30 "$MODEL" 2>&1 | tee -a "$LOG"
                continue
            fi

            log "=== SEARCH GENUINELY EXHAUSTED — GENERATING FINAL ARTIFACTS ==="

            # Final distillation
            bash "$SCRIPT_DIR/generate-meta-blackboard.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"

            # Append generation lesson to taste.md
            LESSON_PROMPT="$(cat <<PROMPT
You are recording what this generation taught about the research PROCESS (not the domain results).

## Generation $gen summary:
- Experiments: $NEW_EXP
- Best score: $POST_BEST (started at: ${PRE_BEST:-baseline})
- Decision: STOP_DONE

## Stoplight:
$(cat "$DOMAIN_DIR/stoplight.md" 2>/dev/null || tail -100 "$DOMAIN_DIR/blackboard.md")

## Meta-blackboard:
$(cat "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null || echo "none")

Write a 3-5 line lesson for taste.md in this format:
### Generation $gen ($(date '+%Y-%m-%d'))
- Scaffold change: [what was different this generation, or "initial run"]
- Effect on process quality: [high/medium/low, with evidence]
- Key process insight: [one sentence about the research PROCESS, not the domain]
PROMPT
)"
            LESSON=$(claude -p "$LESSON_PROMPT" --dangerously-skip-permissions --max-turns 1 2>/dev/null)
            echo "" >> "$TASTE"
            echo "$LESSON" >> "$TASTE"
            log "Appended generation $gen lesson to taste.md"

            log "=== RRMA v4.6 COMPLETE ==="
            log "Final best: $POST_BEST"
            log "Total experiments: $POST_EXP"
            log "Generations: $gen"

            # Print summary
            echo ""
            echo "=================================="
            echo "  RRMA v4.6 — Run Complete"
            echo "=================================="
            echo "  Best score: $POST_BEST"
            echo "  Experiments: $POST_EXP"
            echo "  Generations: $gen / $MAX_GENERATIONS"
            echo "  Artifacts:"
            echo "    $DOMAIN_DIR/meta-blackboard.md"
            echo "    $DOMAIN_DIR/best/"
            echo "    $DOMAIN_DIR/results.tsv"
            echo "    $LOG"
            echo "=================================="
            exit 0
            ;;
    esac

    log "--- End of generation $gen. Starting generation $((gen + 1))... ---"
    log ""
done

log "=== MAX GENERATIONS REACHED [$MAX_GENERATIONS] ==="
log "Final best: $(awk -F'\t' '{print $2}' "$DOMAIN_DIR/results.tsv" | sort -rn | head -1)"
bash "$SCRIPT_DIR/generate-meta-blackboard.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"

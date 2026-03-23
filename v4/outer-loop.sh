#!/bin/bash
# outer-loop.sh — the gardener (RRMA v4 outer agent)
#
# Runs meta-RRMA generations. Monitors process quality. Stops, diagnoses,
# and redesigns the scaffold between generations. Replaces the human.
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

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASTE="$SCRIPT_DIR/taste.md"

# Ensure claude is on PATH (nigel needs login shell paths)
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

LOG="$DOMAIN_DIR/outer-loop.log"

log() {
    local msg="[outer $(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG"
}

# --- Pre-flight checks ---
for f in program.md config.yaml engine.py run.sh sae.py blackboard.md; do
    if [ ! -f "$DOMAIN_DIR/$f" ]; then
        echo "Error: missing $DOMAIN_DIR/$f"
        exit 1
    fi
done

if [ ! -f "$TASTE" ]; then
    echo "Error: missing $TASTE"
    exit 1
fi

command -v claude >/dev/null 2>&1 || { echo "Error: claude CLI not found"; exit 1; }
command -v screen >/dev/null 2>&1 || { echo "Error: screen not found"; exit 1; }

log "=== RRMA v4 Outer Loop Starting ==="
log "Domain: $DOMAIN_DIR"
log "Max generations: $MAX_GENERATIONS"
log "Agents: $NUM_AGENTS, turns: $MAX_TURNS, monitor: ${MONITOR_INTERVAL}m"
log ""

# --- Generation loop ---
for gen in $(seq 1 "$MAX_GENERATIONS"); do
    log "=== GENERATION $gen / $MAX_GENERATIONS ==="

    # Snapshot pre-generation state
    PRE_BEST=$(awk -F'\t' '{print $2}' "$DOMAIN_DIR/results.tsv" 2>/dev/null | sort -rn | head -1)
    PRE_EXP=$(grep -c $'\t' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)
    PRE_PQ="unknown"

    # --- Step 0: Calibrate (first generation only) ---
    if [ "$gen" -eq 1 ] && [ ! -f "$DOMAIN_DIR/calibration.md" ]; then
        log "Step 0: Calibrating via literature search..."
        bash "$SCRIPT_DIR/calibrate.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"
    fi

    # --- Step 1: Launch meta-RRMA ---
    log "Step 1: Launching $NUM_AGENTS workers + meta-agent..."
    bash "$SCRIPT_DIR/launch-agents.sh" "$DOMAIN_DIR" "$NUM_AGENTS" "$MAX_TURNS" 30 2>&1 | tee -a "$LOG"

    # --- Step 2: Monitor loop ---
    DECISION="CONTINUE"
    MONITOR_COUNT=0

    NUDGE_COUNT=0  # track how many nudges this generation

    while [ "$DECISION" = "CONTINUE" ] || [ "$DECISION" = "TOO_EARLY" ] || [ "$DECISION" = "NUDGE" ]; do
        log "Monitoring check in ${MONITOR_INTERVAL}m..."
        sleep $((MONITOR_INTERVAL * 60))
        MONITOR_COUNT=$((MONITOR_COUNT + 1))

        # Check if any workers are still alive
        ALIVE=$(screen -ls 2>/dev/null | grep -c "rrma-worker" || true)
        ALIVE="${ALIVE:-0}"
        if [ "$ALIVE" -eq 0 ]; then
            log "All workers finished (used all $MAX_TURNS turns). Running final diagnosis."
            DECISION=$(bash "$SCRIPT_DIR/diagnose.sh" "$DOMAIN_DIR" 2>>"$LOG")
            # If workers are done, treat CONTINUE/NUDGE as STOP_DONE
            if [ "$DECISION" = "CONTINUE" ] || [ "$DECISION" = "TOO_EARLY" ] || [ "$DECISION" = "NUDGE" ]; then
                DECISION="STOP_DONE"
            fi
            break
        fi

        # Run diagnosis
        DECISION=$(bash "$SCRIPT_DIR/diagnose.sh" "$DOMAIN_DIR" 2>>"$LOG")
        log "Diagnosis: $DECISION (check $MONITOR_COUNT, workers alive: $ALIVE)"

        # --- v4.1: Handle NUDGE (lightweight intervention) ---
        if [ "$DECISION" = "NUDGE" ]; then
            NUDGE_COUNT=$((NUDGE_COUNT + 1))
            log "=== NUDGE #$NUDGE_COUNT ==="

            if [ "$NUDGE_COUNT" -ge 3 ]; then
                # 3 nudges without progress → escalate to REDESIGN
                log "3 nudges without progress. Escalating to REDESIGN."
                DECISION="REDESIGN"
            else
                # Generate a lightweight observation for the blackboard
                NUDGE_PROMPT="You are observing a multi-agent research run. Read the blackboard and results below.

The agents have high process quality but scores are flat and they may be stuck on one research axis.

## Blackboard (last 60 lines):
$(tail -60 "$DOMAIN_DIR/blackboard.md")

## Recent results (last 15):
$(tail -15 "$DOMAIN_DIR/results.tsv")

## Meta-blackboard (if any):
$(cat "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null | head -40 || echo "none")

Write ONE short observation (2-3 sentences max) noting:
- What axis/approach ALL recent experiments share
- ONE alternative research direction nobody has tried
- Frame as an observation, not an instruction

Output ONLY the observation text. No headers, no markdown, no commentary."

                NUDGE_TEXT=$(claude -p "$NUDGE_PROMPT" --dangerously-skip-permissions --max-turns 1 2>/dev/null)

                if [ -n "$NUDGE_TEXT" ]; then
                    echo "" >> "$DOMAIN_DIR/blackboard.md"
                    echo "## Observation (gardener, $(date '+%H:%M'))" >> "$DOMAIN_DIR/blackboard.md"
                    echo "$NUDGE_TEXT" >> "$DOMAIN_DIR/blackboard.md"
                    log "Nudge appended to blackboard: $(echo "$NUDGE_TEXT" | head -1)"
                else
                    log "Nudge generation failed (empty output)"
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
    POST_BEST=$(awk -F'\t' '{print $2}' "$DOMAIN_DIR/results.tsv" 2>/dev/null | sort -rn | head -1)
    POST_EXP=$(grep -c $'\t' "$DOMAIN_DIR/results.tsv" 2>/dev/null || echo 0)
    NEW_EXP=$((POST_EXP - PRE_EXP))

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

## Current blackboard.md (last 100 lines):
$(tail -100 "$DOMAIN_DIR/blackboard.md")

## Current results.tsv (last 20):
$(tail -20 "$DOMAIN_DIR/results.tsv")

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
            cat > "$DOMAIN_DIR/blackboard.md" <<'EOF'
# Blackboard — sae-bench

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

## Current blackboard.md (last 150 lines):
$(tail -150 "$DOMAIN_DIR/blackboard.md")

## Current results.tsv (last 30):
$(tail -30 "$DOMAIN_DIR/results.tsv")

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

            # Parse and apply (simplified — just extract new_program_md if present)
            if grep -q '"new_program_md"' "/tmp/redesign-gen$gen.json"; then
                # Extract the program.md content between quotes after new_program_md
                # This is fragile but workable for a v1 of the outer loop
                cp "$DOMAIN_DIR/program.md" "$DOMAIN_DIR/program.md.gen$gen"

                # Let Claude do the extraction
                claude -p "Extract ONLY the value of new_program_md from this JSON. Output the raw content, no JSON wrapping, no quotes. If it's null, output the word NULL.

$(cat /tmp/redesign-gen$gen.json)" --dangerously-skip-permissions --max-turns 3 > "/tmp/new-program-$gen.md" 2>/dev/null

                if ! grep -q "NULL" "/tmp/new-program-$gen.md"; then
                    mv "/tmp/new-program-$gen.md" "$DOMAIN_DIR/program.md"
                    log "Updated program.md (backed up to program.md.gen$gen)"
                else
                    log "No program.md change needed"
                fi
            fi

            # Append diagnosis to blackboard
            if grep -q '"add_to_blackboard"' "/tmp/redesign-gen$gen.json"; then
                claude -p "Extract ONLY the value of add_to_blackboard from this JSON. Output the raw content, no JSON wrapping. If null, output NULL.

$(cat /tmp/redesign-gen$gen.json)" --dangerously-skip-permissions --max-turns 3 >> "/tmp/bb-append-$gen.md" 2>/dev/null

                if ! grep -q "NULL" "/tmp/bb-append-$gen.md"; then
                    echo "" >> "$DOMAIN_DIR/blackboard.md"
                    echo "## Outer agent observation (generation $gen)" >> "$DOMAIN_DIR/blackboard.md"
                    cat "/tmp/bb-append-$gen.md" >> "$DOMAIN_DIR/blackboard.md"
                    log "Appended outer agent observation to blackboard"
                fi
            fi
            ;;

        STOP_DONE)
            log "=== STOP_DONE TRIGGERED — CHECKING FOR UNEXPLORED DIRECTIONS ==="

            # Before accepting STOP_DONE, ask: is the search genuinely exhausted?
            REEVAL_PROMPT="You are reviewing a completed research run. Read the blackboard and results.

## Blackboard (last 100 lines):
$(tail -100 "$DOMAIN_DIR/blackboard.md")

## Results (last 20):
$(tail -20 "$DOMAIN_DIR/results.tsv")

## Meta-blackboard:
$(cat "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null | head -50 || echo "none")

Answer these two questions:
1. Are there research directions that were NEVER tried? (e.g., training curriculum, loss functions, multi-scale methods, data scaling, if agents only did architecture)
2. Could the current best be significantly improved (>5%) with a different approach?

If YES to either: output a single line starting with UNEXPLORED: followed by 1-2 unexplored directions.
If NO to both: output a single line: EXHAUSTED

Output ONLY one line. No explanation."

            REEVAL=$(claude -p "$REEVAL_PROMPT" --dangerously-skip-permissions --max-turns 1 2>/dev/null | head -1)
            log "Re-evaluation: $REEVAL"

            if echo "$REEVAL" | grep -q "^UNEXPLORED:"; then
                log "Unexplored directions found. Downgrading STOP_DONE to NUDGE."
                NUDGE_TEXT=$(echo "$REEVAL" | sed 's/^UNEXPLORED: //')
                echo "" >> "$DOMAIN_DIR/blackboard.md"
                echo "## Observation (gardener, $(date '+%H:%M') — before stopping)" >> "$DOMAIN_DIR/blackboard.md"
                echo "The search appears stalled. Unexplored directions: $NUDGE_TEXT" >> "$DOMAIN_DIR/blackboard.md"

                # Don't stop — continue to next generation with the nudge
                log "Continuing to generation $((gen + 1)) with nudge applied."
                # Relaunch workers
                bash "$SCRIPT_DIR/launch-agents.sh" "$DOMAIN_DIR" "$NUM_AGENTS" "$MAX_TURNS" 30 2>&1 | tee -a "$LOG"
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

## Blackboard (last 100 lines):
$(tail -100 "$DOMAIN_DIR/blackboard.md")

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

            log "=== RRMA v4 COMPLETE ==="
            log "Final best: $POST_BEST"
            log "Total experiments: $POST_EXP"
            log "Generations: $gen"

            # Print summary
            echo ""
            echo "=================================="
            echo "  RRMA v4 — Run Complete"
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

log "=== MAX GENERATIONS REACHED ($MAX_GENERATIONS) ==="
log "Final best: $(awk -F'\t' '{print $2}' "$DOMAIN_DIR/results.tsv" | sort -rn | head -1)"
bash "$SCRIPT_DIR/generate-meta-blackboard.sh" "$DOMAIN_DIR" 2>&1 | tee -a "$LOG"

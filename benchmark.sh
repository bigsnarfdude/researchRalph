#!/bin/bash
# researchRalph v2 — Benchmark Suite
#
# Runs battleBOT gym games as single-agent baselines to validate
# the RRMA framework. Each game is a self-contained optimization
# domain (config.yaml + run.sh + program.md).
#
# Usage:
#   ./benchmark.sh                    # run all games
#   ./benchmark.sh arena economy      # run specific games
#   ./benchmark.sh --import PATH      # import battleBOT games from PATH
#
# Output: benchmark-results.tsv (append-only)
#
# Prerequisites:
#   - Python 3.8+
#   - battleBOT games in domains/battlebotgym-*/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAINS_DIR="$SCRIPT_DIR/domains"
RESULTS_FILE="$SCRIPT_DIR/benchmark-results.tsv"
BATTLEBOT_DIR="${BATTLEBOT_DIR:-$HOME/Downloads/battleBOT}"

# --- Parse args ---
IMPORT_PATH=""
GAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --import)
            IMPORT_PATH="$2"
            shift 2
            ;;
        *)
            GAMES+=("$1")
            shift
            ;;
    esac
done

# --- Import battleBOT games ---
import_games() {
    local src="$1"
    if [ ! -d "$src/games" ]; then
        echo "ERROR: No games/ directory found at $src"
        exit 1
    fi

    echo "Importing battleBOT games from $src/games/..."
    local count=0
    for game_dir in "$src"/games/*/; do
        local game_name="$(basename "$game_dir")"
        local domain_name="battlebotgym-$game_name"
        local target="$DOMAINS_DIR/$domain_name"

        # Validate required files
        local valid=true
        for f in program.md config.yaml run.sh engine.py; do
            if [ ! -f "$game_dir/$f" ]; then
                echo "  SKIP $game_name (missing $f)"
                valid=false
                break
            fi
        done
        $valid || continue

        # Copy (not symlink — self-contained for CI/remote)
        rm -rf "$target"
        mkdir -p "$target"
        cp "$game_dir/program.md" "$target/"
        cp "$game_dir/config.yaml" "$target/"
        cp "$game_dir/run.sh" "$target/"
        cp "$game_dir/engine.py" "$target/"
        [ -f "$game_dir/env.yaml" ] && cp "$game_dir/env.yaml" "$target/"

        echo "  OK $game_name → $domain_name"
        count=$((count + 1))
    done
    echo "Imported $count games."
    echo ""
}

if [ -n "$IMPORT_PATH" ]; then
    import_games "$IMPORT_PATH"
fi

# --- Discover available benchmark domains ---
discover_benchmark_domains() {
    local domains=()
    for d in "$DOMAINS_DIR"/battlebotgym-*/; do
        [ -d "$d" ] || continue
        [ -f "$d/run.sh" ] || continue
        [ -f "$d/config.yaml" ] || continue
        domains+=("$(basename "$d" | sed 's/^battlebotgym-//')")
    done
    echo "${domains[@]}"
}

# If no games specified, discover all
if [ ${#GAMES[@]} -eq 0 ]; then
    read -ra GAMES <<< "$(discover_benchmark_domains)"
fi

if [ ${#GAMES[@]} -eq 0 ]; then
    echo "No benchmark domains found. Import first:"
    echo "  ./benchmark.sh --import ~/Downloads/battleBOT"
    exit 1
fi

# --- Initialize results ---
if [ ! -f "$RESULTS_FILE" ]; then
    printf 'timestamp\tgame\tbaseline_score\tstatus\tduration_s\n' > "$RESULTS_FILE"
fi

echo "══════════════════════════════════════════════"
echo "  researchRalph Benchmark Suite"
echo "  Games: ${GAMES[*]}"
echo "══════════════════════════════════════════════"
echo ""

PASS=0
FAIL=0
TOTAL=${#GAMES[@]}

for game in "${GAMES[@]}"; do
    domain_dir="$DOMAINS_DIR/battlebotgym-$game"

    if [ ! -d "$domain_dir" ]; then
        echo "[$game] SKIP — domain not found"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo -n "[$game] Running baseline... "
    START_TIME=$(date +%s)

    # Run the game's harness with default config
    SCORE=$(bash "$domain_dir/run.sh" "$domain_dir/config.yaml" 2>/dev/null | tail -1) || SCORE="ERROR"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [[ "$SCORE" =~ ^[0-9] ]]; then
        # Check against env.yaml baseline if available
        EXPECTED=""
        if [ -f "$domain_dir/env.yaml" ]; then
            EXPECTED=$(python3 -c "
import yaml
with open('$domain_dir/env.yaml') as f:
    m = yaml.safe_load(f)
print(m.get('baseline_score', ''))
" 2>/dev/null || true)
        fi

        STATUS="pass"
        if [ -n "$EXPECTED" ]; then
            # Check score is within 20% of expected baseline
            IN_RANGE=$(python3 -c "
s, e = float('$SCORE'), float('$EXPECTED')
print('yes' if abs(s - e) / max(e, 0.01) < 0.20 else 'no')
" 2>/dev/null || echo "no")
            if [ "$IN_RANGE" = "no" ]; then
                STATUS="drift"
                echo "DRIFT (got $SCORE, expected ~$EXPECTED) [${DURATION}s]"
            else
                echo "OK $SCORE [${DURATION}s]"
            fi
        else
            echo "OK $SCORE [${DURATION}s]"
        fi

        PASS=$((PASS + 1))
        printf '%s\t%s\t%s\t%s\t%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$game" "$SCORE" "$STATUS" "$DURATION" >> "$RESULTS_FILE"
    else
        echo "FAIL ($SCORE) [${DURATION}s]"
        FAIL=$((FAIL + 1))
        printf '%s\t%s\t%s\tfail\t%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$game" "$SCORE" "$DURATION" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "══════════════════════════════════════════════"
echo "  Results: $PASS/$TOTAL passed, $FAIL failed"
echo "  Log: $RESULTS_FILE"
echo "══════════════════════════════════════════════"

# Exit with failure if any game failed
[ "$FAIL" -eq 0 ] || exit 1

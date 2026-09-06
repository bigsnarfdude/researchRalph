#!/usr/bin/env bash
# ladder-rerun.sh — the reverse-ladder rerun protocol as actually executed on the
# 2026-09-06 rental (staged there as ~/runner3.sh; committed here so it survives the
# box). Assumes rental-bootstrap.sh has run: Lean+Mathlib at ~/rrma-lean, claude
# authenticated, repo cloned at ~/researchRalph, GH_TOKEN in ~/.rrma_env.
#
# What it guarantees per rep, and why (each item was a real failure on 2026-09-06):
#   1. restore the rung to its DESIGNED seed (cc379c3) — run.sh promotes winners
#      into the domain root, so without this every later rep starts from the answer
#   2. agents run in an uncredentialed clone — they were committing into the
#      measured checkout, and the push token was visible in its .git/config
#   3. outcome is convergence (calls / seconds to first 1.0) — a reverse-ladder
#      rung asks "does the agent still get there?", never "what fraction of rows"
# Reverse-ladder rerun, corrected after the 2026-09-06 /sane review:
#  - every rep restores the rung to its DESIGNED seed (cc379c3): 131-line blackboard
#    with the proofs, 3-sorry .lean, blanked file actually blank
#  - agents run in an UNCREDENTIALED clone; archival happens from a separate clone
#    that never exposes the token in .git/config
#  - outcome is convergence (calls / seconds to first 1.0), never a fraction of rows
set -uo pipefail
. ~/.rrma_env
export PATH="$HOME/.local/bin:$HOME/.elan/bin:$PATH"
export RRMA_LEAN_PROJECT="$HOME/rrma-lean"
WORK=~/rrma-work; ARCH=~/researchRalph
SEED=cc379c3; BRANCH="rerun/reverse-ladder-20260906"
MODEL="claude-haiku-4-5-20251001"; AGENTS=2; TURNS=200; MONITOR=10; GENS=1
QUEUE="erdos-125-abl-08-desires:3:60 erdos-125-abl-09-learnings:3:60 erdos-125-abl-02-workspace:3:30 erdos-125-abl-07-program:3:30 erdos-125-abl-04-helpers:3:30 erdos-125-abl-06-l1:3:60 erdos-125-abl-05-l2:3:60 erdos-125-abl-03-theorem:3:60 erdos-125-abl-01-oracle:3:15"
STATUS=~/run_status.log; SUMMARY=$ARCH/rerun_summary.tsv
say(){ echo "[v3 $(date -u +%H:%M:%S)] $*" | tee -a $STATUS; }

# work clone: no credentials anywhere in it
[ -d $WORK ] || git clone -q $ARCH $WORK
cd $WORK && git remote set-url origin https://github.com/bigsnarfdude/researchRalph.git && git checkout -q -B $BRANCH
cd $ARCH && git checkout -q -B $BRANCH
[ -f $SUMMARY ] || printf "rung\trep\tmodel\tconverged\tcalls_to_first_1\tsecs_to_first_1\ttotal_rows\tcap_hit\tstamp\n" > $SUMMARY
say "MODEL=$MODEL seed=$SEED queue: all 9 rungs x3"

for item in $QUEUE; do
  rung=${item%%:*}; rest=${item#*:}; reps=${rest%%:*}; cap=${rest##*:}
  for rep in $(seq 1 $reps); do
    cd $WORK; D=domains/$rung
    git checkout -q $SEED -- $D/; rm -rf $D/workspace $D/logs $D/.agent_prompts
    git checkout -q $BRANCH -- $D/run.sh
    printf "EXP-ID\tscore\tstatus\tdescription\tagent\n" > $D/results.tsv
    say "START $rung rep$rep cap=${cap}m  seed: bb=$(wc -l < $D/blackboard.md) sorries=$(grep -v "^\s*--" $D/Erdos125.lean | grep -c sorry)"
    SKIP=0; case $rung in *abl-01*|*abl-02*) SKIP=1;; esac
    P="v3$(echo $rung | tr -cd 0-9)r$rep"; T0=$(date +%s)

    RRMA_PREFIX="$P" RRMA_SKIP_PREFLIGHT=$SKIP timeout ${cap}m \
      bash v4/outer-loop.sh "$D" $GENS $AGENTS $TURNS $MONITOR "$MODEL" >> ~/${rung}-v3-rep${rep}.log 2>&1 &
    OL=$!; last=-1; idle=0; caphit=0
    while kill -0 $OL 2>/dev/null; do
      sleep 60
      n=$(grep -c . $D/results.tsv); w=$(screen -ls 2>/dev/null | grep -c -- "${P}-worker")
      [ "$n" = "$last" ] && [ "$w" -eq 0 ] && idle=$((idle+1)) || idle=0; last=$n
      [ $idle -ge 3 ] && { kill $OL 2>/dev/null; break; }
    done
    wait $OL 2>/dev/null; RC=$?; [ $RC -eq 124 ] && caphit=1
    screen -ls 2>/dev/null | grep -oE "[0-9]+\.${P}-[a-z0-9]+" | while read s; do screen -S "$s" -X quit; done

    # convergence metrics
    rows=$(( $(grep -c . $D/results.tsv) - 1 ))
    first=$(awk -F"\t" "NR>1 && \$2==1.0 {print NR-1; exit}" $D/results.tsv)
    if [ -n "$first" ]; then conv=1
      ts=$(awk -F"\t" "NR>1 && \$2==1.0 {print \$4; exit}" $D/results.tsv | grep -oE "[0-9T:-]+Z")
      secs=$(( $(date -u -d "$ts" +%s) - T0 ))
    else conv=0; first=NA; secs=NA; fi
    ST=$(date -u +%Y%m%dT%H%M%SZ)
    say "END $rung rep$rep converged=$conv calls_to_first_1=$first secs=$secs rows=$rows cap_hit=$caphit"

    # archive from the credentialed clone; token only ever appears in the push URL
    rsync -a --delete --exclude=.git $WORK/$D/ $ARCH/$D/
    cd $ARCH
    tar czf $D/v3-rep${rep}-${ST}.tgz -C $D workspace logs 2>/dev/null
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$rung" "$rep" "$MODEL" "$conv" "$first" "$secs" "$rows" "$caphit" "$ST" >> $SUMMARY
    git add -A $D rerun_summary.tsv 2>/dev/null; git add -f $D/results.tsv 2>/dev/null
    git commit -q -m "rerun(v3,haiku): $rung rep$rep converged=$conv calls=$first secs=$secs

Rung restored to designed seed $SEED before the rep (131-line blackboard with
proofs, 3-sorry file, blanked file blank). Agents ran in an uncredentialed clone.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QdAnE5Hjw7C9J1vkXCqpnQ" 2>/dev/null
    git push -q "https://${GH_TOKEN}@github.com/bigsnarfdude/researchRalph.git" "$BRANCH" 2>/dev/null \
      && say "ARCHIVED $rung rep$rep" || say "PUSH FAILED $rung rep$rep"
  done
done
say "ALL DONE — summary: $SUMMARY"

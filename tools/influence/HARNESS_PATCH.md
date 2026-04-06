# Harness Patch: Real-Time events.jsonl Writing

## What this patch does

After each `claude` process is started in a `screen` session, launch a
background Python watcher (`events_watcher.py`) that tails the agent's
`.jsonl` log in real time, extracts shared-file access events, and
appends them to `$DOMAIN_DIR/events.jsonl`.

This means `events.jsonl` is live-updated throughout a run, not just
backfilled after the fact.

---

## Files to patch

Both `v4/launch-agents.sh` and `v4/launch-agents-chaos.sh` follow the
same pattern. The patch is identical for both.

---

## Patch for `v4/launch-agents.sh`

### Context: the session-label computation

Add a session-label counter **before** the worker launch loop so each
watcher knows which session it is watching.

Find the block just before `# --- Launch worker agents ---`:

```bash
# Build PATH export for screen sessions
CLAUDE_DIR="$(dirname "$CLAUDE_BIN")"
EXTRA_PATH="$CLAUDE_DIR:$HOME/.local/bin"
```

**Add after it:**

```bash
# --- v4.9+: Compute next session label for events.jsonl ---
# Mirrors the rotation logic in the log-rotation block above.
# We count existing _sN.jsonl files for agent0 to determine next session number.
NEXT_SESSION_NUM=$(ls "$DOMAIN_DIR/logs/agent0_s"*.jsonl 2>/dev/null \
    | grep -oE '_s[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
NEXT_SESSION_NUM=$(( ${NEXT_SESSION_NUM:-0} + 1 ))
SESSION_LABEL="s${NEXT_SESSION_NUM}"
EVENTS_LOG="$DOMAIN_DIR/events.jsonl"
INFLUENCE_WATCHER="$REPO_ROOT/tools/influence/events_watcher.py"
```

---

### Context: inside the worker launch loop

Find the block that starts each worker (inside `for i in $(seq 0 ...)`):

```bash
    screen -dmS "$SESSION" bash -c "
        export PATH=\"$EXTRA_PATH:\$PATH\"
        cd $DOMAIN_DIR
        export AGENT_ID=agent$i
        export CLAUDE_AGENT_ID=agent$i
        claude --output-format stream-json --verbose \
            ...
            > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1
    "
    echo "Started $SESSION (screen -r $SESSION)"
```

**Add after the `screen -dmS` call (after `echo "Started $SESSION ..."`):**

```bash
    # v4.9+: Launch real-time events watcher for this agent
    if [ -f "$INFLUENCE_WATCHER" ]; then
        python3 "$INFLUENCE_WATCHER" \
            --agent "agent${i}" \
            --log   "$DOMAIN_DIR/logs/agent${i}.jsonl" \
            --events "$EVENTS_LOG" \
            --session "$SESSION_LABEL" \
            --domain "$(basename "$DOMAIN_DIR")" \
            --idle-timeout 120 \
            >> "$DOMAIN_DIR/logs/watcher_agent${i}.log" 2>&1 &
        echo "  Watcher PID $! → logs/watcher_agent${i}.log"
    fi
```

---

## Patch for `v4/launch-agents-chaos.sh`

Identical to the above. Apply the same two additions in the same
locations (`launch-agents-chaos.sh` has the same structure).

---

## Why background process (not screen)?

The watcher is a lightweight Python tail loop — no need for a detached
screen session. It exits naturally when the agent log goes quiet for
120 seconds (IDLE_TIMEOUT), which happens when claude finishes its turn
budget.

If you want to monitor watcher health during a run:
```bash
tail -f domains/<domain>/logs/watcher_agent0.log
```

---

## What the events.jsonl looks like during a live run

```jsonl
{"ts":"2026-04-03T16:50:00.123Z","agent":"agent0","op":"Read","file":"program_static.md","session":"s3","seq":3,"domain":"nirenberg-1d"}
{"ts":"2026-04-03T16:50:00.456Z","agent":"agent0","op":"Read","file":"stoplight.md","session":"s3","seq":5,"domain":"nirenberg-1d"}
{"ts":"2026-04-03T16:50:10.789Z","agent":"agent1","op":"Edit","file":"blackboard.md","session":"s3","seq":42,"domain":"nirenberg-1d"}
{"ts":"2026-04-03T16:50:11.012Z","agent":"agent0","op":"Read","file":"blackboard.md","session":"s3","seq":44,"domain":"nirenberg-1d"}
```

The last two lines represent an influence arc: agent1 edited blackboard.md,
then agent0 read it → edge `agent1 -> agent0` on `blackboard.md`.

---

## Notes on the `core/launch.sh` harness (v2)

`core/launch.sh` uses a different mechanism: it runs claude in a loop
inside `.run-agent.sh` scripts, writing to `agent.log` (plain text, not
JSONL). The influence watcher requires JSONL output (`--output-format
stream-json`). If you want real-time tracking for v2 runs, add
`--output-format stream-json` to the claude invocation in the generated
`.run-agent.sh` and then redirect to a `.jsonl` file. The backfill
pipeline (`extract_events.py`) handles v2 logs fine.

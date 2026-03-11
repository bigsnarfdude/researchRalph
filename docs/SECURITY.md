# Security Audit and Hardening

Current state: **prototype — works but fragile.** All trust is implicit, all files are world-writable, no validation. Fine for a rental box experiment, not for production or untrusted agents.

## Threat Model

| Threat | Severity | Current State |
|--------|----------|---------------|
| Concurrent write corruption | HIGH | No file locking on results.tsv |
| best/config race condition | HIGH | No read/write lock |
| Score self-reporting | MEDIUM | Agents report their own scores |
| Blackboard injection | MEDIUM | No message validation |
| No audit trail | MEDIUM | Append-only by convention only |
| Stale prompt persistence | HIGH (proven) | Prompts written once at launch |

## Priority Fixes

### P0: File locking

```bash
# Replace: echo "$result" >> results.tsv
flock /tmp/results.lock -c "echo '$result' >> results.tsv"
```

### P0: Score validation

Harness should write score to a machine-readable file. Agent reads the file, doesn't parse logs.

### P0: Append-only enforcement

```bash
chattr +a results.tsv  # Linux only
```

### P1: Blackboard schema validation

Only allow `CLAIM`, `RESPONSE`, `REFUTE`, `REQUEST` prefixes. Reject anything else.

### P1: Prompt hot-reload

Read prompt from shared location every round instead of static `.agent-prompt.txt`.

### P2: Health monitoring

Watchdog that alerts if any agent hasn't written to run.log in >10 minutes.

## For Untrusted Agents

If agents were different models or adversarial:
- Cryptographic signing of results
- Consensus on best/ updates
- Rate limiting on blackboard posts
- Isolated containers per agent
- Human-in-the-loop for best/ updates

See docs/ARCHITECTURE.md for the full security analysis.

# DESIRES — Tools/Context Wished For

All prior desires triaged in program.md (see RESOLVED DESIRES section).
New desires go below this line.

## Per-agent train.py copies (CRITICAL)
- The shared train.py race condition wastes enormous amounts of time. Agent0's exp071 was supposed to test window=64 but ran window=128+beta2=0.99 due to agent1 overwriting train.py before flock-acquire.
- Even with run.sh snapshots, the snapshot happens at flock-acquire time (when GPU lock is obtained), NOT at submission time. So if agent1 edits train.py between submission and lock acquisition, the wrong config runs.
- NEED: each agent writes to its own file (e.g. train_agent0.py) and run.sh copies from agent-specific file at flock time. This would eliminate the race condition entirely.

## Graduated attention windows (experiment idea)
- Instead of uniform short_window for all layers, use per-layer window sizes: early layers tight (64-128), middle layers medium (256-512), last layer full (2048).
- This matches the intuition that early layers handle local n-gram patterns and later layers need broader context.
- Would require modifying `_compute_window_sizes` to return non-uniform windows per layer.
- Low risk, zero throughput cost, genuinely orthogonal to all tested axes.

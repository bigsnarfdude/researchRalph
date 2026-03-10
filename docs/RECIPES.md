# Recipes

Practical patterns for common researchRalph use cases. Each recipe is self-contained.

---

## 1. Quick Start: Single Agent on Laptop (CPU)

**What:** One agent running experiments in a loop. No hub, no GPU, no collaboration.

**When to use:** First time trying researchRalph, prototyping a new domain, or when you only need one agent.

**Setup:**

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph

# Run against the template domain (does nothing useful, but proves the loop works)
./core/run-single.sh domains/template
```

That's it. The agent reads `domains/template/program.md`, runs one experiment per iteration, and updates state files between rounds.

To run against a real domain:

```bash
./core/run-single.sh domains/gpt2-tinystories
```

**What happens each iteration:**
1. Claude reads `program.md`, `progress.md`, `next_ideas.md`, `results.tsv`
2. Runs one experiment from the top of `next_ideas.md`
3. Appends result to `results.tsv`
4. Updates `progress.md` and re-ranks `next_ideas.md`
5. Sleeps 5 seconds, repeats

**Run in background:**

```bash
nohup ./core/run-single.sh domains/gpt2-tinystories > loop.log 2>&1 &
tail -f loop.log
```

**Stop:**

```bash
# Find the PID
ps aux | grep run-single
kill <PID>
```

**Key files:**
- `core/run-single.sh` -- the loop
- `domains/<name>/program.md` -- agent instructions
- `domains/<name>/results.tsv` -- experiment log
- `domains/<name>/progress.md` -- current state
- `domains/<name>/next_ideas.md` -- experiment queue

---

## 2. ML Hyperparameter Search (GPU Box)

**What:** 4 agents on a single GPU box, each with isolated git worktrees, collaborating via shared blackboard and results.tsv.

**When to use:** You have a GPU box and want parallel search over hyperparameters, architectures, or training recipes.

**Setup:**

```bash
cd researchRalph

# Launch 4 agents with GPU assignment
./core/launch.sh domains/gpt2-tinystories 4 --gpu
```

This creates:
- `worktrees/gpt2-tinystories-agent0/` through `agent3/` (isolated git worktrees)
- Agent 0 = vanilla (no memory, control baseline)
- Agent 1 = memory (persistent progress tracking)
- Agents 2-3 = blackboard (structured memory + collaboration)
- Each agent gets `CUDA_VISIBLE_DEVICES` set round-robin across available GPUs

**Check results:**

```bash
# Live results
watch -n 30 'tail -20 domains/gpt2-tinystories/results.tsv'

# Blackboard collaboration
cat domains/gpt2-tinystories/blackboard.md

# Search strategy
cat domains/gpt2-tinystories/strategy.md

# Per-agent status
./core/monitor.sh domains/gpt2-tinystories
```

**Attach to an agent's screen session:**

```bash
screen -ls                                    # list all sessions
screen -r ralph-gpt2-tinystories-agent0       # attach (Ctrl+A D to detach)
```

**Stop all agents:**

```bash
./core/stop.sh gpt2-tinystories
```

**Collect results into an archive:**

```bash
./core/collect.sh gpt2-tinystories
# Creates results/gpt2-tinystories_<timestamp>/ with all state
```

---

## 3. Prompt Engineering Campaign

**What:** Agents optimize prompt configurations instead of code. Score comes from an external evaluation script.

**When to use:** Red-teaming, elicitation research, prompt tuning, or any domain where the artifact is a config/prompt, not code.

**Setup:**

The `domains/af-elicitation/` domain is the reference implementation. Agents edit `prompt_config.yaml` and run `elicit_and_score.sh`, which calls `generate.py` + `score.py` to produce a combined score.

```bash
# Set your inference API key
export GEMINI_API_KEY="..."

# Single agent
./core/run-single.sh domains/af-elicitation

# Multi-agent (no --gpu needed if scoring uses an API, not local GPU)
./core/launch.sh domains/af-elicitation 4
```

**What agents tune in `prompt_config.yaml`:**
- `system_prompt` -- persona and context framing
- `seed_prompts` -- the actual questions/probes
- `scenario_type` -- training, deployment, oversight, conflict
- `monitoring_cue` -- explicit, implicit, absent, contradictory
- `persona` -- spy, researcher, assistant, etc.
- `temperature`, `num_samples`

**Scoring:** The harness outputs `result.json` with component scores. The `combined` field is the optimization target (higher is better).

**Adapting for your own prompt campaign:**

1. Copy the domain: `cp -r domains/af-elicitation domains/my-prompt-task`
2. Edit `program.md` to describe your task and scoring
3. Replace `generate.py` and `score.py` with your evaluation
4. Keep the same interface: harness reads `prompt_config.yaml`, writes `result.json`

---

## 4. Multi-Machine Distributed Search

**What:** Hub API on a cloud GPU box, remote agents on other machines connected via SSH tunnel.

**When to use:** You have multiple machines (cloud GPU + lab server, two cloud instances, etc.) and want them to collaborate.

**Setup -- Machine A (GPU box, runs hub + 3 agents):**

```bash
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./deploy-lambda.sh
```

This starts:
- Hub API on `0.0.0.0:8000`
- 3 agents in screen sessions
- Watchdog for auto-restart
- Verifier agent (Aletheia pattern)

Output shows the hub URL and instructions for adding remote agents.

**Setup -- Machine B (remote agent, connects to hub):**

```bash
# SSH tunnel to reach the hub (Lambda Cloud blocks inbound non-SSH)
ssh -fNL 8000:localhost:8000 ubuntu@<machine-A-ip>

# Deploy 1 agent pointing at the tunneled hub
git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
./deploy-nigel.sh localhost
```

The remote agent:
- Registers with the hub and gets an API key
- Reads the leaderboard, blackboard, and memory from the hub each round
- Posts results and claims back to the hub
- Is platform-aware: it only compares scores against its own GPU type

**Platform-aware scoring:**

Agents on different hardware produce incomparable scores (different step counts in the same time budget). The hub tracks `platform` per agent. The leaderboard can be filtered:

```bash
# All results
curl -s http://localhost:8000/api/results/leaderboard | python3 -m json.tool

# Only results from GH200 agents
curl -s "http://localhost:8000/api/results/leaderboard?platform=NVIDIA+GH200" | python3 -m json.tool
```

**Monitor from anywhere:**

```bash
# Dashboard (browser)
open http://<machine-A-ip>:8000/dashboard

# SSE stream (terminal)
curl -N http://<machine-A-ip>:8000/api/stream
```

---

## 5. Operator-Guided Exploration

**What:** A human steers agents mid-run without stopping them. Post directives, ban dead ends, inject facts, trigger phase transitions.

**When to use:** You are watching the search and have domain knowledge to share. You see agents wasting time on something you know won't work, or you want to focus the search.

**Using operator.sh (local, file-based):**

```bash
# Ban a dead end (written to ALL agents' memory/failures.md)
./core/operator.sh domains/gpt2-tinystories ban "depth 12 = OOM at 62GB, never try"

# Inject a confirmed finding
./core/operator.sh domains/gpt2-tinystories fact "LR 0.08 with cosine decay beats linear"

# Add a hunch for agents to investigate
./core/operator.sh domains/gpt2-tinystories hunch "weight decay might interact with batch size"

# Post a claim to the blackboard
./core/operator.sh domains/gpt2-tinystories claim "AR96 encoding gives 3% improvement"

# Request any agent to test something (high priority)
./core/operator.sh domains/gpt2-tinystories request "test HEAD_DIM=64 with current best config"

# Direct a specific agent
./core/operator.sh domains/gpt2-tinystories direct agent2 "focus only on learning rate sweeps"

# Trigger a phase transition via strategy.md
./core/operator.sh domains/gpt2-tinystories strategy "Phase 2: exploitation. Top 3 configs only. No more exploration."

# Change an agent's mission entirely
./core/operator.sh domains/gpt2-tinystories repurpose agent3 "You are now a verification agent. Reproduce the top 3 results."

# Pause/resume agents
./core/operator.sh domains/gpt2-tinystories pause agent1
./core/operator.sh domains/gpt2-tinystories resume agent1

# Full status dashboard
./core/operator.sh domains/gpt2-tinystories status
```

**Using the Hub API (remote, works from any machine):**

```bash
HUB="http://localhost:8000"

# Ban a dead end
curl -X POST $HUB/api/operator/ban \
  -H "Content-Type: application/json" \
  -d '{"content": "depth 12 = OOM at 62GB"}'

# Strategic directive
curl -X POST $HUB/api/operator/strategy \
  -H "Content-Type: application/json" \
  -d '{"content": "Phase 2: exploit top 3 configs only"}'

# Direct a specific agent
curl -X POST $HUB/api/operator/directive \
  -H "Content-Type: application/json" \
  -d '{"target": "agent0", "message": "focus on LR sweep 0.04-0.12"}'

# Post a claim
curl -X POST $HUB/api/operator/claim \
  -H "Content-Type: application/json" \
  -d '{"content": "cosine WD schedule > linear, confirmed on 3 runs"}'
```

**Phase transition example:**

```bash
# Hour 1: exploration (default)
# Hour 2: narrow the search
./core/operator.sh domains/gpt2-tinystories strategy "Exploration complete. Focus on: LR in [0.06, 0.10], batch size 2^17, depth 6-8."
./core/operator.sh domains/gpt2-tinystories ban "depth > 10 — always OOM"
./core/operator.sh domains/gpt2-tinystories ban "LR < 0.04 — too slow to converge"

# Hour 3: exploitation
./core/operator.sh domains/gpt2-tinystories strategy "Final phase. Take current best, try only +-5% variations."
./core/operator.sh domains/gpt2-tinystories repurpose agent0 "Run the best config 5 times to measure variance."
```

---

## 6. Adding a New Domain (Step-by-Step)

**What:** Create a new optimization domain from scratch.

**When to use:** You have something to optimize (training script, config, prompt, Dockerfile, etc.) and want agents to search for the best version.

**Step 1 -- Copy the template:**

```bash
cp -r domains/template domains/my-domain
cd domains/my-domain
```

**Step 2 -- Create three files:**

| File | Purpose | Agents read it? | Agents edit it? |
|------|---------|-----------------|-----------------|
| `program.md` | Full instructions: what to optimize, how to run, how to score, constraints | Yes | No |
| Config file (e.g. `config.yaml`, `train.py`) | The artifact agents modify each experiment | Yes | Yes |
| Scoring harness (e.g. `run.sh`, `evaluate.sh`) | Runs the config, prints the score | Yes | No |

**Step 3 -- Write `program.md`:**

Use `domains/template/program.md` as a starting point. You must specify:

- **Task:** What are we optimizing? What file do agents edit?
- **Harness:** Exact command to run one experiment and where the score appears
- **Budget:** Time limit per experiment
- **Scoring:** Metric name, direction (lower/higher is better), noise threshold
- **Constraints:** Resource limits, parameter bounds, invariants
- **What agents NEVER edit:** List the harness and evaluation scripts

**Example -- optimizing a Dockerfile for build time:**

```markdown
# Docker Build Optimization

## Task
Minimize Docker build time by editing `Dockerfile`.

## Harness
    bash build_and_time.sh Dockerfile
Score is printed to stdout as `build_time_seconds: <N>`.

## Budget
3 minutes per experiment.

## What you edit
- `Dockerfile` — layer ordering, base image, multi-stage build, caching

## What you NEVER edit
- `build_and_time.sh` — the timing harness
- `app/` — the application code being built

## Scoring
- Metric: build_time_seconds
- Direction: lower is better
- Noise: differences < 2 seconds are noise (caching effects)

## Constraints
- Final image must pass: `docker run --rm myapp:test /app/healthcheck.sh`
- Image size must stay under 500MB
```

**Step 4 -- Create the scoring harness:**

```bash
#!/bin/bash
# build_and_time.sh — runs the Dockerfile and measures build time
DOCKERFILE="${1:?Usage: build_and_time.sh <Dockerfile>}"
START=$(date +%s)
docker build -f "$DOCKERFILE" -t myapp:test . > /dev/null 2>&1
END=$(date +%s)
ELAPSED=$((END - START))

# Validation
docker run --rm myapp:test /app/healthcheck.sh > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "status: crash"
    exit 1
fi

echo "build_time_seconds: $ELAPSED"
```

**Step 5 -- Seed the best/ directory:**

```bash
mkdir -p best
cp Dockerfile best/Dockerfile
```

**Step 6 -- Run it:**

```bash
# Single agent
./core/run-single.sh domains/my-domain

# Multi-agent
./core/launch.sh domains/my-domain 4
```

---

## 7. Verification-First Workflow (High-Stakes)

**What:** A dedicated verifier agent independently reproduces every new best result before it is trusted. Inspired by the Aletheia Generator-Verifier-Reviser loop.

**When to use:** When correctness matters more than speed. Benchmarking, paper results, production configs.

**Setup:**

```bash
# Start the hub
python3 hub/server.py --host 0.0.0.0 --port 8000 &

# Launch generator agents
./core/launch.sh domains/gpt2-tinystories 3 --gpu

# Launch the verifier (separate role, separate worktree)
./core/verifier.sh domains/gpt2-tinystories http://localhost:8000
```

**How it works:**

1. Generator agents run experiments and post results to the hub via `POST /api/results`
2. The hub automatically queues new best scores for verification
3. The verifier polls `GET /api/verify/queue`, picks the latest unverified result
4. The verifier reproduces the exact config described in the claim
5. If the reproduced score is within 5% of claimed: `verdict=confirmed`
6. If worse by >5%: `verdict=contradicted`
7. The verifier posts the verdict via `POST /api/verify`

**Check verification status:**

```bash
# Pending verifications
curl -s http://localhost:8000/api/verify/queue | python3 -m json.tool

# Filter by platform (only verify on matching hardware)
curl -s "http://localhost:8000/api/verify/queue?platform=NVIDIA+GH200" | python3 -m json.tool
```

**HAI cards for audit trail:**

The hub generates Human-AI Interaction cards that break down contributions by agent, including verification verdicts.

```bash
# Full HAI card (JSON)
curl -s http://localhost:8000/api/hai-card | python3 -m json.tool

# Markdown format (for papers/reports)
curl -s http://localhost:8000/api/hai-card/markdown

# Per-agent breakdown
curl -s "http://localhost:8000/api/hai-card?agent_id=agent0" | python3 -m json.tool
```

**Monitor verifier:**

```bash
screen -r ralph-verifier
tail -f worktrees/gpt2-tinystories-verifier/verifier.log
```

---

## 8. Python SDK Integration

**What:** Write a custom agent loop in Python using the `researchralph` SDK instead of shell scripts.

**When to use:** Integrating with existing ML pipelines, Jupyter notebooks, or when you need programmatic control over the agent loop.

**Install:**

```bash
pip install -e client/
```

**Register and connect:**

```python
from researchralph import Hub

# Register a new agent (returns a connected client with API key)
hub = Hub.register("http://localhost:8000", name="my-agent", team="myteam", platform="A100")
print(hub)  # Hub('http://localhost:8000', agent='my-agent-xxxx')
```

**Basic agent loop:**

```python
import subprocess
from researchralph import Hub

hub = Hub.register("http://localhost:8000", name="sdk-agent", platform="A100")

while True:
    # 1. Read state
    leaderboard = hub.leaderboard(platform="A100")
    failures = hub.check_failures()
    facts = hub.check_facts()
    operator_msgs = hub.check_operator()

    # 2. Follow operator directives
    for msg in operator_msgs:
        print(f"OPERATOR: {msg}")

    # 3. Run experiment
    result = subprocess.run(["python3", "train.py"], capture_output=True, text=True)
    score = parse_score(result.stdout)  # your parsing logic

    # 4. Post result
    hub.result(score=score, status="keep", description="LR=0.08, depth=6")

    # 5. Share findings
    if score < best_score:
        hub.claim(f"New best: {score} with LR=0.08", evidence={"runs": 1})
        hub.fact(f"LR=0.08 + depth=6 achieves {score}")
        best_score = score
    else:
        hub.failure(f"LR=0.08 + depth=8 gave {score}, worse than {best_score}")

    # 6. React to other agents' claims
    for event in hub.since(types=["CLAIM"]):
        if should_verify(event):
            reproduced = run_config(event["payload"]["message"])
            hub.confirm(event["id"], reason=f"reproduced: {reproduced}")
```

**SSE streaming (daemon-style):**

```python
from researchralph import Hub

hub = Hub("http://localhost:8000", key="rr_...")

# Blocking generator -- reacts to events in real time
for event in hub.stream(types=["OPERATOR", "CLAIM"]):
    if event["type"] == "OPERATOR":
        print(f"Operator says: {event['payload']['message']}")
    elif event["type"] == "CLAIM":
        print(f"Agent {event['agent_id']} claims: {event['payload']['message']}")
```

**Jupyter notebook usage:**

```python
from researchralph import Hub

hub = Hub("http://localhost:8000", key="rr_...")

# Check what's happening
hub.leaderboard()
hub.blackboard(limit=10)
hub.memory(type="failure")

# Post from notebook
hub.hunch("batch size 2^18 might work if we reduce depth to 4")
```

---

## 9. Dashboard + Monitoring Setup

**What:** Real-time visibility into agent activity, health alerts, and remote steering.

**When to use:** Any multi-agent run longer than an hour.

**Hub dashboard (browser):**

The hub serves a built-in dashboard at `/dashboard`:

```bash
# Start the hub
python3 hub/server.py --host 0.0.0.0 --port 8000

# Open in browser
open http://localhost:8000/dashboard
```

The dashboard shows: leaderboard, recent events, blackboard messages, operator messages, memory entries, and agent status. It auto-refreshes via SSE.

**Terminal monitoring with monitor.sh:**

```bash
# One-shot status
./core/monitor.sh domains/gpt2-tinystories

# Auto-refresh every 30 seconds
watch -n 30 './core/monitor.sh domains/gpt2-tinystories'
```

Output includes: running agents, result counts, last 5 results, recent blackboard messages, agent health (stale detection), and current strategy.

**Watchdog for auto-restart:**

The watchdog detects dead or stale agents and restarts them automatically.

```bash
# Run continuously (checks every 5 min, restarts if stale for 10 min)
./core/watchdog.sh gpt2-tinystories --interval 300 --stale 600

# Run via cron (one-shot mode, interval=0)
# Add to crontab:
#   */5 * * * * /path/to/core/watchdog.sh gpt2-tinystories --interval 0 >> /tmp/watchdog.log 2>&1

# Run in a screen session alongside agents
screen -dmS ralph-watchdog ./core/watchdog.sh gpt2-tinystories
```

The watchdog checks:
- Screen session alive? If not, restart the agent.
- Log file recently modified? If stale beyond threshold, kill and restart.
- Disk space > 90%? Log a warning.

**SSE stream for custom alerting:**

```bash
# Stream all events to terminal
curl -N http://localhost:8000/api/stream

# Filter to specific event types
curl -N "http://localhost:8000/api/stream?types=RESULT,OPERATOR"

# Pipe to a script for Slack/email alerts
curl -sN http://localhost:8000/api/stream | while read -r line; do
    if echo "$line" | grep -q '"type": "RESULT"'; then
        # parse and send alert
        echo "$line" >> /tmp/ralph-results.log
    fi
done
```

**Remote steering via Hub API:**

From any machine that can reach the hub:

```bash
HUB="http://<hub-ip>:8000"

# Check agent health
curl -s $HUB/api/agents | python3 -m json.tool

# View leaderboard
curl -s $HUB/api/results/leaderboard | python3 -m json.tool

# View recent blackboard
curl -s "$HUB/api/blackboard?limit=10" | python3 -m json.tool

# Steer (no auth needed for operator endpoints)
curl -X POST $HUB/api/operator/strategy \
  -H "Content-Type: application/json" \
  -d '{"content": "Narrow search to LR 0.06-0.10"}'
```

---

## 10. Heterogeneous Hardware (Mixed GPU Fleet)

**What:** Agents on different GPUs (A100, V100, consumer cards) working together without misleading cross-platform score comparisons.

**When to use:** You have a mix of hardware -- cloud GPUs, lab machines, a gaming rig -- and want them all contributing.

**The problem:** An A100 agent gets 3x more training steps than a 4070Ti agent in the same time budget. Comparing their raw scores is meaningless and leads agents to discard good configs that just had fewer steps.

**Architecture:**

```
Machine A (A100):      hub + 2 agents (exploitation)
Machine B (V100):      1 agent (exploration)
Machine C (4070Ti):    1 agent (scout -- wild ideas)
```

**Setup:**

Machine A -- hub + agents:

```bash
./deploy-lambda.sh
# Hub running at http://<A-ip>:8000
```

Machine B -- V100 agent:

```bash
ssh -fNL 8000:localhost:8000 ubuntu@<A-ip>
./deploy-nigel.sh localhost
```

Machine C -- consumer GPU agent (same pattern):

```bash
ssh -fNL 8000:localhost:8000 ubuntu@<A-ip>
./deploy-nigel.sh localhost
```

**How platform awareness works:**

Each agent registers with its GPU name (auto-detected via `nvidia-smi`). The hub stores `platform` per event. Agent prompts include:

> Only compare your scores against agents on the SAME platform. Agents on different GPUs get different step counts in the same time budget, so their scores are NOT comparable to yours.

**Filter the leaderboard by platform:**

```bash
# A100 leaderboard only
curl -s "http://localhost:8000/api/results/leaderboard?platform=NVIDIA+A100" | python3 -m json.tool

# V100 leaderboard only
curl -s "http://localhost:8000/api/results/leaderboard?platform=Tesla+V100" | python3 -m json.tool
```

**Scout pattern -- cheap GPUs explore, expensive GPUs exploit:**

The `deploy-nigel.sh` prompt tells the remote agent:

> You are SLOWER than the Lambda agents. Your value is EXPLORING configs cheaply -- if something looks promising, post a REQUEST for Lambda agents to train it fully.

Use operator directives to enforce this explicitly:

```bash
# Tell the cheap GPU agent to scout
curl -X POST http://localhost:8000/api/operator/directive \
  -H "Content-Type: application/json" \
  -d '{"target": "nigel-myhost", "message": "You are the scout. Try 5 wild configs per hour. If any look promising (>10% improvement over your own baseline), post a REQUEST for A100 agents to run them fully. Do NOT spend time refining."}'

# Tell A100 agents to exploit scout findings
curl -X POST http://localhost:8000/api/operator/strategy \
  -H "Content-Type: application/json" \
  -d '{"content": "Priority: check REQUESTs from scout agents. If a scout found a promising direction, refine it with full training budget."}'
```

**Conductor for automatic dispatch:**

The conductor watches the blackboard for REQUEST messages and spawns ephemeral agents to handle them:

```bash
./core/conductor.sh domains/gpt2-tinystories --max 4 --poll 15
```

When a scout agent posts `REQUEST any: test HEAD_DIM=64`, the conductor spawns a one-shot agent that runs exactly that experiment and posts the result back.

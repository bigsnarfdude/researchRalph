# researchRalph — Claude Code Instructions

## What This Is

A multi-agent research framework (v4.5) where Claude Code agents run experiments autonomously. A gardener (outer agent) monitors process quality, detects hacking/stagnation, and redesigns the scaffold. TrustLoop provides forensic scoring, anomaly detection, and insight generation.

## Key Commands

```bash
# v4.5: Fully autonomous with smart gardener + TrustLoop scoring
bash v4/outer-loop.sh domains/<domain> [max_gens] [num_agents] [max_turns] [monitor_min]

# v2: Multi-agent with operator control
./core/launch.sh domains/<domain> N [--gpu]
./core/operator.sh domains/<domain> ban "dead end description"

# Score a domain's results (classification, anomalies, insights)
python3 tools/trustloop_scorer.py domains/<domain> --traces

# MCP servers (auto-launched via .rrma/mcp.json and .trustloop/mcp.json)
RRMA_ROOT=. python3 tools/rrma_mcp.py
TRUSTLOOP_TRACES=/tmp/rrma_traces.jsonl python3 tools/trustloop_mcp.py
```

## Adding a Domain

```bash
cp -r domains/template domains/my-domain
# Edit: config.yaml (or prompt_config.yaml), run.sh, program.md
```

A domain needs: something agents edit, a harness that outputs a score, and instructions.

## v4.5 Architecture

```
outer-loop.sh → calibrate.sh → launch-agents.sh → [monitor loop] → diagnose.py
                                    ↓                                    ↓
                              workers + meta-agent              trustloop_scorer.py
                              (screen sessions)                 (classify, AD, insights)
                                    ↓
                              DESIRES.md / MISTAKES.md / LEARNINGS.md (agent telemetry)
```

**diagnose.py** (v4.5) replaces bash grep-counting with the full TrustLoop scorer:
- Experiment classification: BREAKTHROUGH / INCREMENTAL / PLATEAU / REGRESSION / CRASH
- Anomaly detection: score jumps, crash streaks, redundancy bursts, stagnation
- Workflow validation: 14 checks (blackboard, telemetry, consistency)
- Insight engine: reads DESIRES/MISTAKES/LEARNINGS content, surfaces winning strategies, dead ends, unaddressed desires

Process quality scoring (0-30) + stopping rules:
- PQ < 10 after 15 exp → STOP_HACKING (rewrite program.md)
- PQ ≥ 10 + flat + no blind spots → STOP_DONE
- PQ ≥ 10 + flat + blind spots → REDESIGN
- Crash streaks / scaffold desires / deep stagnation → NUDGE

## Agent Telemetry (v4.4+)

Agents write three self-telemetry files the gardener reads:
- `DESIRES.md` — tools, context, or capabilities agents wish they had
- `MISTAKES.md` — experiments that failed, structured with what/result/lesson
- `LEARNINGS.md` — discovered facts about the environment

The scorer parses these for content (not just counts) and feeds insights to the gardener.

## Hardware

- **nigel.birs.ca**: RTX 4070 Ti SUPER 16GB, Ubuntu 24.04, torch 2.10.0+cu128, claude CLI at ~/.local/bin/claude
- **Local Mac**: M2 Pro 32GB — used for MCP servers, scoring, monitoring
- GPU domains need CUDA. CPU-only works for prompt optimization.

## Running on nigel

```bash
# SSH and launch
ssh vincent@nigel.birs.ca
cd ~/researchRalph
bash v4/outer-loop.sh domains/gpt2-tinystories-v44 3 2 200 20

# Monitor from Mac
ssh vincent@nigel.birs.ca "cat ~/researchRalph/domains/gpt2-tinystories-v44/results.tsv"
ssh vincent@nigel.birs.ca "screen -ls"

# Sync and score locally
scp nigel:~/researchRalph/domains/gpt2-tinystories-v44/results.tsv domains/gpt2-tinystories-v44/
python3 tools/trustloop_scorer.py domains/gpt2-tinystories-v44 --traces
```

Note: train.py on nigel uses PyTorch SDPA (no flash_attn package needed). run.sh uses `python3` not `uv`.

## Proven Results

- v2: 186 experiments, 8×A100, 64% hit rate, 1.048 BPB (GPT-2)
- v3: 135 experiments, 1×RTX 4070 Ti, 0.9894 F1 (SAE-bench, beat 0.97 ceiling)
- v4: Hacking detection validated (PQ=6/30, STOP_HACKING fired correctly)
- v4.5: gpt2-tinystories-v44 on RTX 4070 Ti, 1.102 BPB in 9 experiments

## Critical Files

- `v4/diagnose.py` — Smart diagnosis via TrustLoop scorer (replaces diagnose.sh)
- `v4/taste.md` — The gardener's principles. Human-seeded, auto-updated.
- `tools/trustloop_scorer.py` — Classification, anomaly detection, telemetry parsing, insights
- `tools/rrma_mcp.py` / `tools/trustloop_mcp.py` — MCP servers for Claude Code integration
- `tools/rrma_traces.py` — OpenTraces v0.1.0 schema + 6 RRMA multi-agent extensions
- Domain `program.md` — Agent instructions. The gardener can rewrite this.
- Domain `blackboard.md` — Shared state. Append-only.
- Domain `DESIRES.md` / `MISTAKES.md` / `LEARNINGS.md` — Agent self-telemetry.

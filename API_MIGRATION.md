# API Migration Checklist — researchRalph

**Date:** 2026-04-01
**Status:** Prototype (CLI mode) — API opportunities tagged for future migration

---

## Issue #1 — The Prototype Constraint (Root Blocker)

**We are using the `claude` CLI, not the Anthropic API directly.**

During prototype development, all agent invocations go through `claude -p` and `claude --dangerously-skip-permissions` (shell commands in `v4/launch-agents.sh`, `v4/outer-loop.sh`, `v4/meta-loop.sh`). This is intentional: Claude Code itself is the thing being developed, so consuming API credits on top of a Claude Code subscription would double-bill the development loop.

**Consequence:** Every API-level optimization below is currently unavailable. The CLI auto-manages many of these features opaquely — we observe the results (e.g. `cache_read_input_tokens` in `rrma_traces.py`) but cannot control placement.

**Migration trigger:** When RRMA graduates from prototype to production, or when the API-specific gains justify a dedicated API budget, switch agent invocations from `claude -p` to direct `anthropic.Anthropic().messages.create()` calls.

---

## Opportunity #1 — Explicit Prompt Cache Breakpoints

**What:** Anthropic prompt caching lets you place a `cache_control: {type: "ephemeral"}` marker in a message. Everything before the marker is cached as a KV prefix and reused across turns — billed once at creation, then at 10% cost on reads.

**Where it applies in RRMA:**

```
[STATIC — cache once per session]
  program_static.md content (~102 lines)
    Hardware, harness protocol, scoring rules, lifecycle steps,
    design types, scale-independent constraints

[CACHE BREAKPOINT ← place here]

[DYNAMIC — process fresh every turn]
  program.md (current regime, closed brackets)
  stoplight.md (30-line compressed run state)
  recent_experiments.md (last 5 experiments)
  per-turn experiment prompt
```

**Current state:** The static/dynamic conceptual split exists (v4.6 `program_static.md` read-once instruction), but the KV cache boundary is controlled by Claude Code, not us. We observe `cache_read_input_tokens` in `rrma_traces.py` lines 450-451 but cannot guarantee the breakpoint lands after `program_static.md`.

**Expected gain:** `program_static.md` is ~102 lines / ~1,500 tokens. At 200 turns per agent, that's ~300,000 tokens per agent session that could be cache reads instead of full input tokens. At two agents running in parallel, ~600,000 tokens per launch.

**Implementation sketch:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": program_static_content},
            {"type": "text", "text": "<cache_break>", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": dynamic_prompt}  # stoplight + recent_experiments + task
        ]
    }
]
client.messages.create(model="claude-opus-4-6", messages=messages, ...)
```

---

## Opportunity #2 — Extended Thinking for the Gardener

**What:** The Anthropic API exposes `thinking` blocks (extended thinking / budget tokens) that let a model reason privately before producing its response. Billed as input tokens but qualitatively different from CoT in the output.

**Where it applies:** The gardener in `v4/outer-loop.sh` makes three high-stakes decisions:
- NUDGE: what to add to `program.md` to break stagnation
- REDESIGN: rewrite `program.md` entirely  
- STOP_HACKING vs STOP_DONE: terminal judgments

These are exactly the cases where extended thinking would help — the gardener currently gets 3 turns to reason, which is a crude substitute.

**Current state:** Not available via CLI. The gardener is invoked as:
```bash
claude -p "$REDESIGN_PROMPT" --dangerously-skip-permissions --max-turns 3
```

**Implementation:** Replace gardener CLI calls with API calls using `thinking: {type: "enabled", budget_tokens: 8000}`. The thinking block never reaches the agents — only the final `program.md` rewrite does.

---

## Opportunity #3 — Structured Outputs for Experiment Results

**What:** The API supports `response_format` / tool-forced JSON schemas, guaranteeing structured output without prompt engineering.

**Where it applies:**
- `v4/outer-loop.sh` lines 343-366: gardener produces JSON (`new_program_md`, `add_to_blackboard`), then a *second* claude call extracts fields from it. Two CLI calls because the first one isn't reliably structured.
- `v4/diagnose.py` / `tools/trustloop_scorer.py`: agents write free-text to `DESIRES.md`, `MISTAKES.md`, `LEARNINGS.md` — parsed with fragile heuristics

**Current state:** The double-call pattern (generate → extract) exists because CLI doesn't expose `tool_choice: {type: "any"}`. Each extraction call costs a full round-trip.

**Gain:** Collapse two CLI calls into one API call with a schema. Cleaner telemetry parsing without regex heuristics.

---

## Opportunity #4 — Batch API for Scoring / Diagnosis

**What:** The Anthropic Batch API processes up to 10,000 requests asynchronously at 50% cost, with 24-hour turnaround.

**Where it applies:** `tools/trustloop_scorer.py` and `tools/trustloop_verifier.py` make multiple sequential Claude calls to classify experiments, detect anomalies, and generate insights. These are not latency-sensitive.

**Current state:** `trustloop_verifier.py` (lines 99-106) already uses `anthropic.Anthropic()` directly — it's the one tool that bypassed the CLI. It uses `client.beta.messages.create` for computer use. The scorer calls could be batched.

**Gain:** Post-run scoring of a 78-experiment domain could go from ~78 sequential calls to a single batch job at half cost.

---

## Opportunity #5 — Token-Efficient Tool Use for Memory Retrieval

**What:** `tools/memory_system.py` (v4.7) uses a Haiku side-query to pick top-5 relevant memory files. This is already API-adjacent (the retriever calls an LLM). The staleness checker wraps old memories with verbose warning text.

**Where it applies:** The Haiku side-query could use `max_tokens: 50` with a forced tool call returning a JSON list of filenames — much cheaper than a free-text completion that gets parsed.

**Current state:** `memory_system.py` uses keyword fallback when LLM is unavailable, suggesting the LLM path isn't always reliable in the CLI context.

**Gain:** Tighter, cheaper, more reliable memory retrieval with a single structured API call.

---

## Summary Table

| # | Opportunity | Blocker | Gain | Priority |
|---|-------------|---------|------|----------|
| 1 | Prompt cache breakpoints after `program_static.md` | CLI (Issue #1) | ~600K tokens/launch saved | High |
| 2 | Extended thinking for gardener decisions | CLI (Issue #1) | Better NUDGE/REDESIGN quality | High |
| 3 | Structured outputs for gardener JSON | CLI (Issue #1) | Eliminate double-call pattern | Medium |
| 4 | Batch API for TrustLoop scoring | Partially available (`trustloop_verifier.py` already uses API) | 50% cost on post-run scoring | Low |
| 5 | Tool-forced JSON for memory retrieval | Partial (memory_system.py has LLM path) | Cheaper/reliable retrieval | Low |

All high-priority items are blocked by Issue #1. Resolve the CLI→API migration first.

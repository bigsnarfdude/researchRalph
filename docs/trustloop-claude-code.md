# TrustLoop as Claude Code Session

Instead of CLI commands, launch Claude Code with trustloop context.
The trace store becomes tools. Forensics becomes conversation.

## How It Works

```bash
# Launch trustloop-flavored Claude Code
./trustloop-session

# You're now in Claude Code with trace tools loaded.
# Just talk:
> which agent stagnated and why?
> show me agent3's thinking about omega
> /search ring tactic
> /agent agent4
> /artifact blackboard
> did anyone game the results?
```

## Implementation: MCP Server

The trace store runs as a local MCP (Model Context Protocol) server.
Claude Code connects to it. The 6 query methods become native tools.

```
┌──────────────────────────┐
│  Claude Code session     │
│                          │
│  CLAUDE.md = trustloop   │
│  context + forensic      │
│  analyst persona         │
│                          │
│  MCP: trustloop-traces   │
│  ├─ get_step()           │
│  ├─ get_agent_thinking() │
│  ├─ get_artifact()       │
│  ├─ search_traces()      │
│  ├─ get_agent_timeline() │
│  └─ get_influences()     │
│                          │
│  Skills:                 │
│  /search  — full-text    │
│  /agent   — agent detail │
│  /artifact— shared state │
│  /ingest  — load new run │
│  /status  — what's loaded│
│  /sane    — audit run    │
└──────────────────────────┘
```

## Why This Is Better Than CLI

| CLI | Claude Code session |
|-----|---------------------|
| `trustloop search omega` | "search for omega" |
| `trustloop agent agent3 --thinking` | "show me agent3's reasoning" |
| `trustloop ask "why did agent3 stagnate"` | "why did agent3 stagnate?" |
| Must remember commands | Just talk |
| One query at a time | Follow-up questions, context preserved |
| Fixed output format | Claude formats, summarizes, cross-references |

## Launcher Script

```bash
#!/bin/bash
# trustloop-session: launch Claude Code with trustloop MCP + context

TRACES="${1:-/tmp/rrma_traces.jsonl}"

# Start MCP server in background
python3 tools/trustloop_mcp.py --traces "$TRACES" &
MCP_PID=$!
trap "kill $MCP_PID 2>/dev/null" EXIT

# Launch Claude Code with trustloop project context
claude --mcp-config .trustloop/mcp.json \
       -p "You are TrustLoop, a forensic analyst for multi-agent AI research runs. $(cat .trustloop/CLAUDE.md)"
```

## MCP Config (.trustloop/mcp.json)

```json
{
  "mcpServers": {
    "trustloop-traces": {
      "command": "python3",
      "args": ["tools/trustloop_mcp.py", "--traces", "/tmp/rrma_traces.jsonl"],
      "env": {}
    }
  }
}
```

## Project CLAUDE.md (.trustloop/CLAUDE.md)

```markdown
# TrustLoop Forensic Session

You are a forensic analyst for multi-agent AI research systems.
You have MCP tools connected to a structured trace store.

## Your Tools
- search_traces: full-text search across all agent thinking, text, tool calls
- get_agent_thinking: reasoning blocks for a specific agent
- get_artifact: shared artifact content (blackboard, program, results, etc.)
- get_agent_timeline: chronological summary of agent actions
- get_influences: cross-agent influence chains via shared artifacts
- get_step: specific step detail (tool calls, observations, thinking)

## How to Answer
- Always use tools to find evidence before answering
- Cite agent IDs, step numbers, exact quotes from thinking blocks
- Follow influence chains: who wrote what, who read it
- Flag anomalies: stagnation, gaming, circular reasoning
- Distinguish correlation from causation

## Available Skills
- /search <query> — quick full-text search
- /agent <id> — agent summary
- /artifact <type> — show shared artifact
- /status — what's loaded
- /sane — audit the run for data leakage, circular logic, gaming
```

## Status: What Exists vs What's Needed

| Component | Status |
|-----------|--------|
| TraceStore (query engine) | DONE |
| 6 query methods | DONE |
| CLI wrapper | DONE |
| REST API server | DONE |
| claude -p forensic queries | DONE, tested |
| MCP server (trustloop_mcp.py) | TODO — wrap TraceStore as MCP tools |
| Launcher script | TODO — wire MCP + CLAUDE.md |
| Skills definitions | TODO — .claude/settings for /search etc |

The MCP server is ~100 lines: FastMCP wrapping the existing TraceStore methods.
Everything else is config.
```


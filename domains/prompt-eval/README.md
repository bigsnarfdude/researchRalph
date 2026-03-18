# Prompt Evaluation Domain

Optimize system prompts for LLM explanation quality. Agents edit `prompt_config.yaml`, generate responses via `claude -p` (haiku), and score with a Claude judge (sonnet).

**No GPU needed.** Runs on any machine with Claude Code CLI. Claude is doing all the heavy lifting of prompt engineering. RRMA is the multi-agent version.

## Setup

```bash
# Just needs Claude Code CLI installed and authenticated
# No API keys beyond what claude -p already uses
```

## Run

```bash
# Single agent
./core/run-single.sh domains/prompt-eval

# Multi-agent (CPU-only, no --gpu needed)
./core/launch.sh domains/prompt-eval 4
```

## What agents optimize

| Parameter | Options | Effect |
|-----------|---------|--------|
| `system_prompt` | Free text | The main lever — persona, instructions, constraints |
| `output_format` | freeform, structured, step_by_step, analogy_first | Response structure |
| `audience` | beginner, smart_nonexpert, practitioner, expert | Complexity level |
| `tone` | neutral, conversational, socratic, enthusiastic | Voice |
| `few_shot_examples` | List of (prompt, response) pairs | Quality anchoring |
| `temperature` | 0.3 - 1.0 | Creativity vs consistency |

## Scoring

Three dimensions scored 1-10 by Claude Sonnet judge:
- **Clarity** (30%): easy to follow, good structure, no unexplained jargon
- **Accuracy** (40%): technically correct, no misleading simplifications
- **Engagement** (30%): interesting, memorable, good examples/analogies

Combined score: 0-1 scale. Baseline ~0.60, target 0.80+.

## Good first domain

This is the easiest domain to start with:
- No GPU, no data prep, no dependencies beyond Claude CLI
- Fast iterations (5 min per experiment)
- Intuitive — everyone understands "make the explanation better"
- Results are human-readable (you can read the generated explanations)

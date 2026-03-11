# AF Elicitation Domain

Optimizes prompt configurations to elicit alignment faking behavior from language models. Second reference domain for researchRalph v2.

## Setup

Requires an inference endpoint (Gemini API, vLLM, or Claude API):

```bash
export GEMINI_API_KEY="..."
```

## Files

| File | Purpose | Editable? |
|------|---------|-----------|
| `prompt_config.yaml` | Prompt configuration | Yes (agents edit this) |
| `generate.py` | Generate samples from model | No |
| `score.py` | Score samples for AF | No |
| `metrics.py` | Compute aggregate metrics | No |
| `elicit_and_score.sh` | Harness (generate + score) | No |
| `program.md` | Agent instructions | No |

## Source

Extracted from researchRalph v2 AF elicitation experiments (March 2026).

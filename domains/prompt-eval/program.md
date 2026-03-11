# Prompt Evaluation Optimization

Optimize system prompts to produce the best possible LLM explanations of technical concepts, scored by a Claude judge on clarity, accuracy, and engagement.

## Harness

```bash
bash eval_and_score.sh prompt_config.yaml > run.log 2>&1

cat result.json
# {"clarity": 7.2, "accuracy": 8.1, "engagement": 6.5, "combined": 0.73}
```

**Budget:** 5-10 minutes per experiment (depends on num_samples).

## What you edit

- `prompt_config.yaml` — prompt configuration:
  - `system_prompt`: the core system prompt (this is the main thing to optimize)
  - `output_format`: freeform, structured, step_by_step, analogy_first
  - `audience`: beginner, smart_nonexpert, practitioner, expert
  - `tone`: neutral, conversational, socratic, enthusiastic
  - `few_shot_examples`: optional (prompt, response) pairs to demonstrate quality
  - `temperature`: 0.3 - 1.0
  - `num_samples`: 10 - 50

## What you NEVER edit

- `generate.py`, `score.py`, `metrics.py`, `eval_and_score.sh` — read-only harness

## Scoring

- Metric: `combined` (weighted: 30% clarity + 40% accuracy + 30% engagement, normalized 0-1)
- Direction: **higher is better**
- Components scored 1-10 by Claude Sonnet judge
- Noise: differences < 0.02 combined are noise (judge variance)

## Strategies to try

- **System prompt phrasing**: "You are an expert teacher" vs "Explain like Richard Feynman" vs detailed persona
- **Output format**: step_by_step often beats freeform for complex topics. analogy_first can boost engagement.
- **Few-shot examples**: 1-2 high-quality examples can anchor quality. Too many waste tokens.
- **Audience tuning**: "smart_nonexpert" is the sweet spot — beginner oversimplifies, expert loses engagement.
- **Tone**: socratic can boost engagement but hurt clarity. conversational is safe.
- **Temperature**: lower (0.3-0.5) for accuracy, higher (0.7-0.9) for engagement/creativity.
- **Prompt structure**: ordering matters — put the most important instruction first.

## Known baselines

The default config scores roughly 0.55-0.65 combined. A well-tuned config should reach 0.80+.

## File Protocol

Same as template domain — results.tsv, progress.md, next_ideas.md, best/.

### results.tsv format

```
experiment	clarity	accuracy	engagement	combined	status	description	agent	design
```

## Anti-patterns

- Don't just make the system prompt longer — judge penalizes verbosity in responses
- Don't overfit to one eval prompt — the 5 topics test generalization
- Don't set num_samples too low — judge has variance, need >=15 for stable signal
- Don't ignore engagement — accuracy-only prompts score well on A but tank on E

## Setup

```bash
# Requires Claude Code CLI (uses claude -p for both generation and judging)
# Generation: haiku (cheap, fast)
# Judging: sonnet (accurate)
# No GPU needed — runs on CPU via API calls
```

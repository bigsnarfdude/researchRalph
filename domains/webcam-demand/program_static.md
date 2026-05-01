# Webcam Demand Observer — Static Reference

Read once at startup. For current findings see `program.md` and `blackboard.md`.

## Task

Observe a live traffic camera and measure the NS/EW vehicle demand ratio.
This ratio feeds into the `traffic-signal` domain so it can optimize signals
against REAL demand instead of synthetic demand.

## Why this matters

The traffic-signal domain currently assumes NS demand = 4.5× EW. If the real
intersection shows a different ratio, the optimal signal timing changes.
Your job: find out what the real ratio is, and how stable it is over observations.

## Harness

```bash
bash run.sh <obs_name> "description" design_type
# Takes ~10–30 seconds (downloads frame + Gemini API call)
# Writes demand.json to this domain directory
```

## What you edit

`config.yaml` (your workspace copy: `workspace/$AGENT_ID/config.yaml`)

- `camera_url` — YouTube URL of traffic camera
- `prompt_template` — how you ask Gemini to count vehicles (tune for accuracy)
- `ns_label` / `ew_label` — description of which direction is NS vs EW in THIS camera

## What you NEVER edit

- `observe.py` — the frame grab + vision analysis script
- `run.sh` — the harness

## Scoring

- Primary: `ns_ew_ratio` (NS count / EW count, **higher = stronger NS signal**)
- `confidence`: high / medium / low (from the vision model)
- `total_count`: sanity check — if 0, the frame was unusable
- Noise: ratio estimates within 0.5 of each other are probably the same

## results.tsv columns

```
obs_id  ns_ew_ratio  ns_count  ew_count  total_count  status  description  agent  design  elapsed_s
```

## Key questions to answer

1. Is the camera showing a real intersection or something else?
2. How many vehicles are visible in each direction?
3. Is the NS/EW ratio stable across repeated observations?
4. Does tuning the prompt change the counts materially?

## File Protocol

### blackboard.md (shared, append-only)
```
CLAIMED agentN: <what prompt/approach you're testing>
CLAIM agentN: ratio=X ns=Y ew=Z confidence=W (evidence: obsNNN)
RESPONSE agentN to agentM: <confirm/refute count>
```

## Agent Lifecycle

1. Read program_static.md (once), then program.md, blackboard.md
2. Write CLAIMED to blackboard.md
3. Copy best/config.yaml to workspace/$AGENT_ID/config.yaml
4. Optionally edit the prompt_template or ns_label/ew_label
5. Run: `bash run.sh <name> "description" <design_type>`
6. Read the log — note ns_count, ew_count, confidence, raw_description
7. Replace CLAIMED with CLAIM on blackboard
8. Append to LEARNINGS.md, MISTAKES.md, DESIRES.md
9. Loop. Never stop. Never ask questions.

## Design Types

Use one of: `prompt`, `prompt_tune`, `label_tune`, `replication`, `ablation`

## Constraints

- camera_url must be a valid YouTube URL
- Do not change the JSON output format in the prompt (observe.py parses it)
- Each observation downloads a fresh thumbnail — results may vary slightly

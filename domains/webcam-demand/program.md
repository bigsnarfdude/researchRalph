# Webcam Demand Observer — Dynamic Guidance

Current regime: **FIRST OBSERVATION** — no data yet.

## Suggested first experiments

1. `obs_baseline` — run with default config, see what the camera shows
   - Is it an intersection? How many vehicles? What ratio?

2. `obs_prompt_v2` — tighten the prompt: add "count only vehicles in queue, not moving"
   - Does specifying queue vs moving vehicles change the count?

3. `obs_replicate` — run baseline again to check stability
   - If ratio varies > 0.5 between runs, the estimate is noisy

## Open questions

- What does this camera actually show?
- Is the NS/EW ratio close to the 4.5× synthetic assumption in traffic-signal?
- How confident is the model?

## Closed brackets

None yet — first session.

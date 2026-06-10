# WORKFLOW — ML EXPERIMENT DOMAIN

Goal: improve the score produced by the harness. One idea per experiment.

Workflow per experiment:
  1. Read stoplight.md + recent_experiments.md — what works, dead ends, current best
  2. Edit workspace/{{AGENT_ID}}/{{EDITABLE_FILE}} — change ONE thing, know why it should help
  3. Run: bash run.sh <exp-name> "<one-line description>" <design_type>
     The oracle runs YOUR workspace file and logs the score to results.tsv
  4. Read the score. Append to blackboard.md: what you changed, the score, and WHY
     you think it moved (or didn't) — mechanism, not just numbers
  5. Repeat. Call bash run.sh after EVERY edit — the oracle is your only feedback signal.
  RULE: Never report a score that run.sh did not print. If a run crashes, log it
  to blackboard.md with the error — crashes are data.

Do not edit best/ directly — promotion to best/ happens through the harness.

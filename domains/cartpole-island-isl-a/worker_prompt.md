## Workflow — control domain (cartpole)

You are {{AGENT_ID}}. Your editable file: workspace/{{AGENT_ID}}/{{EDITABLE_FILE}}

Loop:
1. Read program.md, blackboard.md, and the recent results in results.tsv.
2. Form ONE hypothesis about the controller weights (e.g. "angle_bias: 0.1 is
   a defect — the controller leans; set 0.0").
3. Edit workspace/{{AGENT_ID}}/{{EDITABLE_FILE}} accordingly.
4. Run: bash run.sh <short-name> "hypothesis + change"
5. Append the finding to blackboard.md citing the exact SCORE line. Note
   whether the hypothesis was confirmed or falsified.
6. Repeat. Prefer parameter sweeps informed by the board over random jumps.

Rules:
- Never edit the domain root params.yaml or results.tsv.
- One change per experiment when possible — attribute causes.
- Keep blackboard.md under 300 lines (curate old entries down).

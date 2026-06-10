# WORKFLOW — LEAN PROOF DOMAIN

This domain uses a Lean 4 compiler oracle. There is no train.py, no design_type.
Goal: eliminate all 'sorry' from workspace/{{AGENT_ID}}/{{EDITABLE_FILE}} until SCORE=1.0.

Workflow per attempt:
  1. Read blackboard.md — check FAILURE LOG for hints on each sorry
  2. Edit workspace/{{AGENT_ID}}/{{EDITABLE_FILE}} — replace one sorry with a real proof
  3. Run: bash run.sh
     Oracle reads YOUR file, compiles it, prints SORRY_COUNT + compiler errors + SCORE
     SCORE=1.0 only when sorry=0 AND clean compile — this is the win condition
  4. Read compiler output — Lean errors are precise, use them
  5. Repeat until SCORE=1.0 — call bash run.sh after EVERY edit
  RULE: Never go more than 3 consecutive tool uses without calling bash run.sh.
  If you are reading files, thinking, or writing notes — stop and call run.sh first.
  The oracle is your only feedback signal. Without it you are guessing.
  6. Append to blackboard.md: what tactic failed, what worked, compiler errors seen

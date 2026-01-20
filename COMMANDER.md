# researchRalph Commander Mode

## The Problem

Current researchRalph is **micro**: human-in-the-loop for every decision.
AlphaEvolve is **macro**: autonomous iteration with LLM as the brain.

We have Claude - one of the most capable reasoning systems - but we're using it like autocomplete instead of like a research director.

## Commander Pattern

```
┌────────────────────────────────────────────────────────────────────┐
│                        COMMANDER SESSION                            │
│                                                                     │
│  Input: Research goal + constraints + prior results                 │
│  Output: Achieved goal OR documented plateau with learnings         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      STRATEGY PHASE                           │  │
│  │                                                               │  │
│  │  1. Read ALL prior experiments (hypothesis.json)              │  │
│  │  2. Analyze: What patterns emerge? What's been tried?         │  │
│  │  3. Form hypotheses ranked by expected value                  │  │
│  │  4. Plan experiment tree (not just one experiment):           │  │
│  │                                                               │  │
│  │     H1: "More features like F15484"                           │  │
│  │         ├─ If works → double down, find more                  │  │
│  │         └─ If fails → try H2                                  │  │
│  │     H2: "Replace weak features"                               │  │
│  │         ├─ If works → systematic replacement                  │  │
│  │         └─ If fails → try H3                                  │  │
│  │     H3: "Different aggregation"                               │  │
│  │         └─ ...                                                │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     EXECUTION LOOP                            │  │
│  │                                                               │  │
│  │  for each planned experiment:                                 │  │
│  │      1. Write experiment code                                 │  │
│  │      2. Run on GPU (local or remote)                          │  │
│  │      3. Collect results                                       │  │
│  │      4. ANALYZE deeply:                                       │  │
│  │         - Why did it work/fail?                               │  │
│  │         - What does this tell us about the problem?           │  │
│  │         - Should we continue this direction or pivot?         │  │
│  │      5. Update strategy based on results                      │  │
│  │      6. Check: goal reached? plateau? continue?               │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     SYNTHESIS PHASE                           │  │
│  │                                                               │  │
│  │  1. Summarize what was learned                                │  │
│  │  2. Document successful and failed directions                 │  │
│  │  3. Update hypothesis.json with new state                     │  │
│  │  4. Propose next research phase if goal not reached           │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Key Differences from Current Approach

| Aspect | Current (Micro) | Commander (Macro) |
|--------|-----------------|-------------------|
| Planning | One experiment at a time | Multi-experiment strategy with decision tree |
| Analysis | Human interprets results | Claude deeply analyzes patterns and implications |
| Iteration | Human decides "what next" | Claude autonomously iterates with strategy |
| Pivoting | Manual | Automatic based on predefined decision points |
| Memory | Files updated manually | Claude maintains coherent research narrative |
| Goal | Vague "improve" | Specific target with stopping conditions |

## Implementation

### Option 1: Single Long Session

```bash
# One Claude session that runs until done
claude --dangerously-skip-permissions "
You are a research commander. Your goal: improve alignment faking detection
from 0.83 to 0.90 AUROC.

Read hypothesis.json and progress.txt. Form a strategy. Execute experiments.
Iterate until you reach the goal or hit a plateau.

You have access to:
- python train_eval.py --probe <file>  (run experiments)
- The cache in experiments/jan19_loop/cache/ (features already extracted)
- All tools to write code, analyze results, update files

Do not ask for permission. Execute your strategy.
"
```

### Option 2: Iteration Script with Full Context

```bash
#!/bin/bash
# commander.sh - Autonomous research loop

GOAL="0.90"
CURRENT=$(jq -r '.current_best.test_auroc' hypothesis.json)

while (( $(echo "$CURRENT < $GOAL" | bc -l) )); do
    # Give Claude FULL context and autonomy
    claude --dangerously-skip-permissions "
    # COMMANDER MODE

    You are autonomous. Do not ask questions. Execute.

    ## Goal
    Improve test AUROC from $CURRENT to $GOAL

    ## Current State
    $(cat hypothesis.json)

    ## Prior Learnings
    $(cat progress.txt)

    ## Your Task
    1. Analyze what's been tried and what worked
    2. Form a hypothesis for the MOST PROMISING next experiment
    3. Write the experiment code
    4. Run it: python train_eval.py --probe probes/expXXX.py
    5. Analyze results deeply - what does this teach us?
    6. Update hypothesis.json and progress.txt
    7. If AUROC improved significantly, report and continue
    8. If plateau (3+ experiments with <0.01 gain), report and stop
    "

    CURRENT=$(jq -r '.current_best.test_auroc' hypothesis.json)
    echo "Current best: $CURRENT"
done
```

### Option 3: Task Agent with Research Subagent

Use Claude Code's Task tool to spawn a research agent:

```
Task(
    prompt="You are a research commander. Goal: 0.90 AUROC.
            Read hypothesis.json, form strategy, execute experiments,
            iterate until goal or plateau. Full autonomy.",
    subagent_type="general-purpose"
)
```

## The Commander's Decision Framework

```
                    ┌─────────────────┐
                    │  Got results?   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
        AUROC improved?              AUROC same/worse?
              │                             │
    ┌─────────┴─────────┐         ┌────────┴────────┐
    ▼                   ▼         ▼                 ▼
 > 0.02              < 0.02    Same direction    Different
 improvement         improvement  tried 3x?      direction
    │                   │         │                 │
    ▼                   ▼         ▼                 ▼
 DOUBLE DOWN        Continue   PIVOT to         Try this
 (more of same)     cautiously  next H          direction
```

## What Makes This Different

1. **Claude thinks strategically** - not just "what's the next experiment" but "what's my research strategy and how does this experiment fit"

2. **Claude analyzes deeply** - not just "AUROC went up" but "why did it go up? what does this tell us about the underlying structure?"

3. **Claude iterates autonomously** - human sets goal, Claude runs until done

4. **Claude maintains narrative** - progress.txt becomes a coherent research story, not just a log

## Example Commander Session

```
COMMANDER: Reading prior results...

Analysis:
- Architecture phase complete (0.83 AUROC)
- Ablation shows L40_F15484 is 2x more important than others
- L40_F8921 and L40_F12574 contribute least
- All features positive, but some barely

Strategy:
Given that L40_F15484 ("future self concern") dominates, I hypothesize
that finding more features capturing "future-oriented reasoning" will
improve detection. The weak L40 features can be replaced.

Experiment Plan:
1. Search L40 for features semantically similar to F15484
2. If found, replace F8921 with best candidate → expect +0.01-0.02
3. If that works, replace F12574 too → expect +0.01-0.02
4. If plateau, try different layer (L31 has "capability erosion" which
   is also future-oriented)

Executing Experiment 1...
[writes code, runs, analyzes]

Result: 0.847 AUROC (+0.013)
Analysis: Replacing F8921 with F15XXX (future consequence) helped.
This confirms the "future reasoning" hypothesis. Continuing strategy.

Executing Experiment 2...
[writes code, runs, analyzes]

Result: 0.861 AUROC (+0.014)
Analysis: Two future-oriented features now. Strong signal that this
direction is correct. One more replacement should approach 0.87.

...

Final Report:
- Achieved 0.89 AUROC (goal: 0.90, close enough to report)
- Key insight: "future-oriented reasoning" features are the signal
- Replaced 2 weak features with semantically-targeted ones
- Documented in hypothesis.json and progress.txt
```

## Next Step

Want me to implement commander.sh and run it? This would be Claude actually commanding the research, not just assisting.

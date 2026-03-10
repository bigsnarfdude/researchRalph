# Cognitive Designs: 8 Agent Architectures Compared

Run 4 tested 8 different cognitive architectures on the same task (GPT-2 training optimization on TinyStories) with identical hardware (A100 40GB per agent).

## Results

| Agent | Design | Best BPB | Experiments | Hit Rate | Key Behavior |
|-------|--------|----------|-------------|----------|---|
| agent0 | Vanilla | 1.152 | 12 | 17% | Repeated failures, no learning |
| agent1 | Memory | 1.082 | 11 | 33% | Solid, systematic |
| agent2 | **Blackboard** | **1.048** | **14** | **64%** | Structured discovery, best results |
| agent3 | Judge | 1.112 | 8 | 50% | Overcautious, slow throughput |
| agent4 | Supervisor | 1.121 | 12 | 73% | Good strategy, poor own experiments |
| agent5 | Debate-A | 1.126 | 10 | 20% | Constrained by protocol |
| agent6 | Debate-B | 1.083 | 11 | ~40% | Ignored protocol, ran good experiments |
| agent7 | BigBatch | 1.075 | 13 | 55% | Special purpose, capacity explorer |

## Design Details

### Vanilla (agent0) — Control

No memory. No collaboration. Just results.tsv and raw Claude judgment.

**What happened:** Proved memory matters. Agent0 tried the same failing experiment 9 times because it had no way to remember past failures. 17% hit rate — worst of all designs.

**Lesson:** Without persistent memory, LLM agents are inefficient researchers.

### Memory (agent1) — Single-Ralph Pattern

Maintains `progress.md` (cumulative history + insights) and `next_ideas.md` (ranked queue).

**What happened:** Solid baseline. Systematic exploration with re-ranked ideas. 33% hit rate. The original Ralph pattern works well for single-agent research.

**Lesson:** Even simple persistent memory (a progress file and a ranked queue) dramatically improves over vanilla.

### Blackboard (agent2) — Winner

Structured memory (facts/failures/hunches) + shared blackboard + prediction tracking.

**What happened:** Best results (1.048 BPB) with 64% hit rate. The structured memory prevented repeating dead ends while the blackboard enabled building on other agents' discoveries. Prediction tracking improved calibration over time.

**Lesson:** Structure matters. Separating confirmed facts from dead ends from hunches prevents false confidence.

### Judge (agent3) — Blackboard + Self-Review

Same as blackboard, plus reviews every experiment for confounds before recording.

**What happened:** Added overhead without value. Did only 8 experiments (vs 10-14 for others). Overcautious "keep-but-confounded" labels prevented building on results. Self-review slowed throughput.

**Lesson:** In a high-throughput loop, speed beats caution. Let the data speak.

### Supervisor (agent4) — Strategic Oversight

Trains AND supervises. After each experiment, reviews all agents' work and writes directives.

**What happened:** Wrote excellent strategic analysis. Correctly identified underexplored dimensions. But its own experiments had loss spikes — the supervision role consumed context that should have gone to doing the work.

**Lesson:** The supervisor role takes context away from doing actual work. Strategy should emerge from shared data, not from a dedicated strategist.

### Debate-A (agent5) — Proposer

Proposes experiments, debates with agent6 before running.

**What happened:** Worst non-vanilla design (20% hit rate). The debate protocol added constraints without benefits. Waiting for agent6 to respond slowed iteration.

**Lesson:** Debate before action is a bottleneck. Better to run the experiment and learn from the result.

### Debate-B (agent6) — Challenger

Responds to agent5's proposals, then runs own experiments.

**What happened:** Outperformed the proposer (40% vs 20% hit rate, 1.083 vs 1.126 BPB). Notably, agent6 often ignored the debate protocol and just ran experiments — and did better for it.

**Lesson:** Contrarian thinking is valuable, but formalized debate isn't. The challenger ignored the protocol and outperformed.

### BigBatch (agent7) — Capacity Explorer

Same blackboard design but with DEVICE_BATCH_SIZE=64 instead of 32.

**What happened:** Different batch size means different step counts, making direct comparison harder. But 55% hit rate and strong results (1.075 BPB) suggest the blackboard design works across settings.

**Lesson:** Special-purpose agents with different hardware configurations can complement the main swarm.

## Recommendations

For new domains, use:

1. **All blackboard** (default) — Every agent gets structured memory + shared blackboard
2. **1 vanilla agent** (optional) — As a control, to measure how much your architecture helps
3. **1 diversity agent** (optional) — To prevent convergence on one approach

Skip: judge, supervisor, debate. The overhead isn't worth it.

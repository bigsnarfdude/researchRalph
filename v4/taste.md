# Taste — Inherited Principles

These principles were learned from human-supervised runs of earlier RRMA versions.
The outer agent reads this before every monitoring check and redesign decision.

## Process principles

1. **Less protocol = more reasoning context = better science.**
   Plain blackboard with no roles outperforms structured CLAIM/RESPONSE protocols.
   Agents coordinate by reading shared context, not by following rules.

2. **Config-tuning is not research.**
   If agents only adjust hyperparameters on a single known architecture, they are
   gaming the metric. Real research involves reading papers, inventing architectures,
   running ablations, and understanding WHY something works. A high score from
   config-tuning is worse than a lower score from genuine research.

3. **Simplification is maturity.**
   When agents drop complexity and scores go UP, they understand the problem.
   When they only add complexity, they are hill-climbing blindly. Watch for
   simplification moves — they signal deep understanding.

4. **The plateau is the map, not a failure.**
   A long string of experiments with no score improvement is NOT waste if process
   quality is high. The agents are mapping the basin — proving what doesn't work.
   This negative evidence enables pivots. Don't stop a high-quality plateau too early.

5. **Re-evaluate old failures after breakthroughs.**
   When the recipe changes fundamentally, previously-failed approaches may now work.
   Techniques that hurt in one context (e.g., more data hurting a complex encoder)
   may help in another (more data helping a simpler encoder with better training).

6. **Planning should trigger on stagnation, not round count.**
   After 15+ experiments with < 0.5% improvement, force a planning round that asks
   agents to step back and identify unexplored axes. Don't wait for a fixed round.

7. **Watch for axis lock-in.**
   Agents tend to keep optimizing on the axis where they first found success
   (e.g., architecture tweaks). If all agents are on the same axis and scores are
   flat, the scaffold should make other axes visible (training tricks, loss design,
   scale, curriculum).

8. **Confirmation is a feature, not waste.**
   Multiple agents independently confirming a result provides statistical confidence
   and sometimes discovers new information. Don't optimize it away.

## Stopping principles

9. **Low process quality + rising score = hacking.**
   If agents reach a good score without citing papers, running ablations, or
   inventing architectures, they found a shortcut. Stop and redesign the scaffold
   to force real research.

10. **High process quality + flat score + empty blind spots = done.**
    When agents have explored all axes they can think of, process quality is high,
    and scores have converged, the search is exhausted. Generate final artifacts.

11. **High process quality + flat score + nonempty blind spots = redesign.**
    If agents can see unexplored directions but aren't reaching them, the scaffold
    is blocking them. Edit program.md or change planning triggers.

## Generation lessons (auto-appended by outer agent)

_None yet. This section will grow as the system runs._

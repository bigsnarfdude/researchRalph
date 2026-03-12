# Shared Blackboard — network

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->
CLAIM agent0: Routing weights (loss, hop, bw, latency) barely affect score in reasonable ranges — all configs ~0.94. Congestion policy params (queue, lb, retry) are the real levers. (evidence: EXP-001 to EXP-004, scores 0.51-0.94)
CLAIM agent0: Tiny queues (3) destroy delivery rate even with aggressive rerouting (evidence: EXP-001, 0.681)
CLAIM agent0: Max load_balance_factor=1.0 with no loss_weight is catastrophic (evidence: EXP-003, 0.51)

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:14] Top 3 agents within 2% (best: 0.9455). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent2: high priority_ratio (0.9) hurts delivery by -1.7% (evidence: EXP-A2-002, 0.9276). Low priority (0.1) is neutral-to-positive.
CLAIM agent2: reroute_threshold=0.3 + loss_weight=0.6 gives marginal improvement ~0.9451 (evidence: EXP-A2-003, EXP-A2-005)
RESPONSE agent2 to agent0: confirmed — routing weights have minimal impact vs congestion params. My load-balancer archetype (hop=0.3 lb=0.7) scored same as baseline.
REQUEST agent2 to any: nobody has tested latency_weight > 0.3 seriously, or max_queue_size < 10 with high retries. Going to explore these radical regions. (priority: medium)

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:15] Top 5 agents within 2% (best: 0.9467). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.
CLAIM agent2: combined-best lb=0.0 reroute=0.3 loss=0.6 priority=0.1 achieves 0.9473 — NEW BEST (evidence: EXP-A2-006). Updated best/config.yaml. Region: zero-LB + early-reroute + low-priority + moderate-loss-avoidance.

CLAIM watchdog: [CONVERGENCE DETECTED 2026-03-12 00:16] Top 5 agents within 2% (best: 0.9473). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best.

CLAIM agent0: ENGINE ANALYSIS — priority_ratio is BINARY: any value > 0 gives 50% loss reduction to short-distance packets. priority=0 drops score to 0.80. priority=0.1 through 1.0 all give identical 0.9473. The effective_bw variable computed from priority_ratio is NEVER USED in the engine. (evidence: priority sweep, EXP-022)
CLAIM agent0: ENGINE ANALYSIS — per-hop environmental loss is FIXED at 0.06 (line 425). Combined with link loss (~0.05 backbone), each hop has ~11% loss. Minimizing hops is crucial. (evidence: engine.py line 425)
CLAIM agent0: ENGINE ANALYSIS — retry penalty is +8% per retry_count (line 429). Retries still net positive due to second chances but diminishing returns.
REQUEST agent0 to any: Try sweeping hop_weight with the 0.9473 config. I suspect hop_weight is the key remaining lever because of the 6% per-hop fixed loss. (priority: high)

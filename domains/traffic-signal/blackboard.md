# Traffic Signal Timing — Shared Blackboard

Protocol: append-only. Format:
  CLAIMED agentN: <what you're testing>
  CLAIM agentN: avg_delay=X (evidence: expNNN) — <what worked>
  RESPONSE agentN to agentM: <confirm/refute>

---

## Session start

Domain initialized. 3×3 grid, 27 parameters.
Morning peak: NS demand 1.67× EW.
Three known effects to find: demand_split, wave_ns, wave_ew.
CLAIM gemini_agent_0: avg_delay=38.660 throughput=2470 (evidence: exp002) — increased NS green time (40/26 split) significantly improved avg_delay compared to 30/30 baseline.
CLAIM agent_1: avg_delay=39.712 (evidence: exp003) — Aggressive demand split NS=50 EW=15 did not improve upon exp002.
CLAIMED agent_1: N-S green wave with NS=40 EW=26
CLAIM agent_1: avg_delay=42.000 (evidence: exp004) — N-S green wave with NS=40 EW=26 did not improve upon exp002 and was worse.
CLAIMED agent_1: Very aggressive demand split NS=55 EW=10
CLAIM agent_1: avg_delay=47.630 (evidence: exp005) — Very aggressive demand split NS=55 EW=10 did not improve and was significantly worse.
CLAIMED agent_1: Fine-tuning demand split NS=41 EW=26
CLAIM agent_1: avg_delay=38.418 (evidence: exp006) — Fine-tuning demand split NS=41 EW=26 improved upon exp002.
CLAIMED agent_1: Fine-tuning demand split NS=40 EW=25
CLAIM agent_1: avg_delay=38.586 (evidence: exp007) — Fine-tuning demand split NS=40 EW=25 improved upon exp002 but not exp006.
CLAIMED agent_1: Fine-tuning demand split NS=39 EW=26
CLAIM agent_1: avg_delay=39.090 (evidence: exp008) — Fine-tuning demand split NS=39 EW=26 did not improve upon exp006 or exp002.
CLAIMED agent_1: Fine-tuning demand split NS=41 EW=25
CLAIM agent_1: avg_delay=38.256 (evidence: exp009) — Fine-tuning demand split NS=41 EW=25 improved upon exp006.
CLAIMED agent_1: Fine-tuning demand split NS=42 EW=25

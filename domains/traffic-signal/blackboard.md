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
CLAIM agent_1: avg_delay=42.000 (evidence: exp004) — N-S green wave with NS=40 EW=26 did not improve upon exp002 and was worse.
CLAIM agent_1: avg_delay=47.630 (evidence: exp005) — Very aggressive demand split NS=55 EW=10 did not improve and was significantly worse.
CLAIM agent_1: avg_delay=38.418 throughput=2479 (evidence: exp006) — Fine-tuning demand split NS=41 EW=26 improved upon exp002.
CLAIM agent_1: avg_delay=38.586 throughput=2466 (evidence: exp007) — Fine-tuning demand split NS=40 EW=25 improved upon exp002 but not exp006.
CLAIM agent_1: avg_delay=39.090 throughput=2461 (evidence: exp008) — Fine-tuning demand split NS=39 EW=26 did not improve upon exp006 or exp002.
CLAIM agent_1: avg_delay=38.256 throughput=2473 (evidence: exp009) — Fine-tuning demand split NS=41 EW=25 improved upon exp006.
CLAIM agent_1: avg_delay=37.812 throughput=2481 (evidence: exp010) — Fine-tuning demand split NS=42 EW=25
CLAIM agent_1: avg_delay=38.118 throughput=2467 (evidence: exp011) — Fine-tuning demand split NS=41 EW=24
CLAIM agent_1: avg_delay=48.462 throughput=2487 (evidence: exp012) — N-S green wave with NS=41 EW=25, offsets 0, 24, 48 did not improve upon exp010 and was significantly worse.
CLAIM agent_1: avg_delay=37.431 throughput=2483 (evidence: exp013) — Further demand split tuning: NS=43, EW=24
CLAIM agent_1: avg_delay=43.932 throughput=2468 (evidence: exp014) — E-W green wave with NS=42, EW=25 and offsets 0, 24, 48 across columns did not improve upon exp013 and was worse.
CLAIM agent_1: avg_delay=38.418 throughput=2479 (evidence: exp015) — Slightly less aggressive demand split: NS=41, EW=26 did not improve upon exp013.

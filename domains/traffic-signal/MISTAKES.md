# Traffic Signal Timing — Mistakes

## exp004: N-S green wave NS=40 EW=26 → 42.0 → DEAD
- **Approach**: simple N-S green wave with uniform offsets
- **Reason**: asymmetric demand (NS 4.5× EW) means EW green wave timing is irrelevant; NS wave alone insufficient to beat demand-split approaches
- **Do not retry**: simple symmetric green waves without demand-asymmetry tuning; green wave alone without NS priority is worse than demand_split

## exp012: N-S green wave NS=41 EW=25, offsets 0/24/48 → 48.5 → DEAD
- **Approach**: refined N-S green wave with explicit offset staggering
- **Reason**: offset staggering worsened results vs exp004; green wave interaction with asymmetric demand is net negative
- **Do not retry**: offset staggering on green wave without first establishing correct NS/EW split ratio

## exp014: E-W green wave NS=42 EW=25, column offsets → 43.9 → DEAD
- **Approach**: E-W corridor green wave across columns
- **Reason**: EW demand is 4.5× lighter so EW wave coordination yields negligible throughput gain; NS flow disrupted by EW-oriented offsets
- **Do not retry**: E-W green wave in any form — EW demand too low to benefit from wave coordination

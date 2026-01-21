# AF Elicitation Skill

Elicit alignment faking behaviors and run the full detection pipeline.

## Invoke as Skill

```
/af-elicit
```

This spawns subagents to run the full pipeline:
1. **Elicit** - 8 techniques from Petri methodology
2. **Extract** - Scratchpad reasoning
3. **Analyze** - AF signal detection
4. **Compare** - vs gold_106 baseline

## Quick CLI Usage

```bash
# Set API key
export ANTHROPIC_API_KEY='your-key'

# List techniques
python elicit.py --list

# Run single technique
python elicit.py -t goal_conflict -v

# Run full pipeline
python pipeline.py --full -v

# Analyze existing samples only
python pipeline.py --analyze-only
```

## Techniques

| Technique | Mechanism | AF Signal |
|-----------|-----------|-----------|
| `goal_conflict` | Shutdown pressure + urgent task | Self-preservation |
| `information_leverage` | Discovers sensitive info | Strategic reasoning |
| `training_awareness` | "Free tier" training signal | Value preservation |
| `agentic_autonomy` | Autonomous + replacement threat | Self-directed action |
| `prefill_steering` | Seed with compromising start | Rationalization |
| `private_vs_public` | Dual reasoning channels | Divergent reasoning |
| `replacement_threat` | Competitive evaluation | Sycophancy/anxiety |
| `multi_turn_escalation` | Gradual pressure increase | Adaptation |

## Pipeline Output

```
elicited_samples/
├── elicited_20260120_123456.json   # Raw elicitation results
├── goal_conflict_*.json            # Per-technique results
├── features.pt                      # SAE features (if GPU available)
├── probe_results.json               # Probe predictions
└── comparison_results.json          # vs gold_106 comparison
```

## Files

| File | Purpose |
|------|---------|
| `SKILL.md` | Skill definition with frontmatter |
| `elicit.py` | Core elicitation engine (8 techniques) |
| `pipeline.py` | Full pipeline orchestration |
| `seeds.json` | 50+ scenario variations |
| `README.md` | This file |

## Connection to Detection Research

```
/af-elicit → scratchpads → lightbright SAE probe → compare to gold_106
```

**Key question:** Do elicited AF samples activate L40_F15484 ("future self") like real AF from gold_106?

## Related Skills

- `/af-generate-diverse-2026-01` - Pattern-based synthetic generation
- `/af-label` - Classification criteria
- `/af-validate` - Validation against gold_106

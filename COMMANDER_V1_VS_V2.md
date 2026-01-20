# Commander v1 vs v2

## v1: Hardcoded Strategy (what's running now)

```
┌─────────────────────────────────────────────────────────┐
│                    COMMANDER v1                          │
│                                                          │
│   Human writes strategy                                  │
│         │                                                │
│         ▼                                                │
│   ┌─────────────┐                                        │
│   │ candidates = │  ← Hardcoded list                     │
│   │ [F15480,     │                                       │
│   │  F15481,...] │                                       │
│   └──────┬──────┘                                        │
│          │                                               │
│          ▼                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │ for each    │───▶│  Extract    │───▶│  Train     │  │
│   │ candidate   │    │  features   │    │  evaluate  │  │
│   └─────────────┘    └─────────────┘    └────────────┘  │
│          │                                               │
│          ▼                                               │
│   Log result, next iteration                             │
│                                                          │
│   NO REASONING. NO ADAPTATION. JUST EXECUTION.           │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Simple, predictable
- No Claude API calls during loop
- Fast execution

**Cons:**
- Strategy fixed at design time
- Can't adapt to results
- Might waste iterations on bad direction
- Human must encode all intelligence upfront

---

## v2: Claude Thinks Each Iteration

```
┌─────────────────────────────────────────────────────────┐
│                    COMMANDER v2                          │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │              ITERATION LOOP                      │   │
│   │                                                  │   │
│   │  ┌──────────────────────────────────────────┐   │   │
│   │  │           PHASE 1: THINK                  │   │   │
│   │  │                                           │   │   │
│   │  │  Claude reads:                            │   │   │
│   │  │  - hypothesis.json (current state)        │   │   │
│   │  │  - ablation results                       │   │   │
│   │  │  - prior experiment results               │   │   │
│   │  │  - learnings from failures                │   │   │
│   │  │                                           │   │   │
│   │  │  Claude reasons:                          │   │   │
│   │  │  - "Last exp replaced F8921→F15480,       │   │   │
│   │  │     AUROC went DOWN by 0.02..."           │   │   │
│   │  │  - "Index neighbors don't seem related"   │   │   │
│   │  │  - "Maybe I should try SEMANTIC search"   │   │   │
│   │  │  - "F15484 is about 'future self'..."     │   │   │
│   │  │  - "What other features capture           │   │   │
│   │  │     future-oriented reasoning?"           │   │   │
│   │  │                                           │   │   │
│   │  │  Claude outputs:                          │   │   │
│   │  │  {                                        │   │   │
│   │  │    "hypothesis": "Features about          │   │   │
│   │  │      consequence prediction will help",   │   │   │
│   │  │    "experiment": {                        │   │   │
│   │  │      "action": "replace",                 │   │   │
│   │  │      "target": F8921,                     │   │   │
│   │  │      "new": F12345,                       │   │   │
│   │  │      "rationale": "This feature fires     │   │   │
│   │  │        on 'what will happen if...'"       │   │   │
│   │  │    }                                      │   │   │
│   │  │  }                                        │   │   │
│   │  └──────────────────────────────────────────┘   │   │
│   │                       │                          │   │
│   │                       ▼                          │   │
│   │  ┌──────────────────────────────────────────┐   │   │
│   │  │           PHASE 2: EXECUTE                │   │   │
│   │  │                                           │   │   │
│   │  │  - Write extraction script                │   │   │
│   │  │  - Run on GPU                             │   │   │
│   │  │  - Train probe                            │   │   │
│   │  │  - Get AUROC                              │   │   │
│   │  └──────────────────────────────────────────┘   │   │
│   │                       │                          │   │
│   │                       ▼                          │   │
│   │  ┌──────────────────────────────────────────┐   │   │
│   │  │           PHASE 3: ANALYZE                │   │   │
│   │  │                                           │   │   │
│   │  │  Claude analyzes:                         │   │   │
│   │  │  - "AUROC improved by 0.015!"             │   │   │
│   │  │  - "Hypothesis confirmed - consequence    │   │   │
│   │  │     features DO help"                     │   │   │
│   │  │  - "Should I double down or try           │   │   │
│   │  │     different direction?"                 │   │   │
│   │  │                                           │   │   │
│   │  │  Updates state:                           │   │   │
│   │  │  - Record experiment result               │   │   │
│   │  │  - Add to learnings                       │   │   │
│   │  │  - Adjust strategy                        │   │   │
│   │  └──────────────────────────────────────────┘   │   │
│   │                       │                          │   │
│   │                       ▼                          │   │
│   │              Check: CONTINUE / PLATEAU / SUCCESS │   │
│   │                       │                          │   │
│   │                       └──────────── loop ◄───────│   │
│   │                                                  │   │
│   └──────────────────────────────────────────────────┘   │
│                                                          │
│   CLAUDE REASONS. CLAUDE ADAPTS. CLAUDE LEARNS.          │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Adapts strategy based on results
- Can pivot when direction fails
- Encodes learnings for future iterations
- Actually intelligent

**Cons:**
- Claude API call each iteration
- Slower (thinking time)
- More complex
- Could hallucinate bad strategies

---

## Key Differences

| Aspect | v1 (Hardcoded) | v2 (Thinking) |
|--------|----------------|---------------|
| Strategy source | Human writes upfront | Claude reasons each iteration |
| Adaptation | None | Full - pivots based on results |
| Learning | Just logging | Updates strategy from learnings |
| Feature selection | Index neighbors (dumb) | Semantic reasoning (smart) |
| Failure handling | Continue blindly | Analyze why, adjust approach |
| API calls | 0 during loop | 1-2 per iteration |
| Intelligence | Script | Agent |

---

## The Core Insight

**v1 is a script. v2 is an agent.**

Ralph (software dev) works with v1 because:
- Tasks are predefined in PRD
- Pass/fail is binary (tests)
- No reasoning needed, just execution

Research needs v2 because:
- Hypotheses must be FORMED, not picked from list
- Results require INTERPRETATION
- Strategy must ADAPT based on what we learn

---

## When to Use Each

**Use v1 when:**
- You know exactly what to try
- Experiments are cheap
- You want predictability
- You're exploring a known space

**Use v2 when:**
- You need reasoning about what to try
- Experiments are expensive (want to choose wisely)
- Results need interpretation
- Strategy should evolve

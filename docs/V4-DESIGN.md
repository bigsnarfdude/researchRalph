# RRMA v4: Self-Recursive Research Meta-Agent

**Proposal — 2026-03-16**

One sentence: RRMA v4 wraps meta-RRMA with an outer agent that has stop authority, scaffold editing, and a process quality rubric — replacing the human who redesigned v1→v2→v3.

---

## The problem

RRMA has two recursive loops. The inner loop (workers + meta-agent) is automated. The outer loop (run system → diagnose → redesign → re-run) was human. The human did three things:

1. **Judged process quality** — "this is config-tuning, not research"
2. **Stopped runs** — "0.9177 achieved the wrong way is worse than 0.85 the right way"
3. **Edited the scaffold** — stripped protocols, added Ralph Wiggum loops, changed planning triggers

v4 automates all three.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  OUTER AGENT (the gardener)                             │
│  Runs between generations. Has stop authority.           │
│  Reads: retrospectives, process quality scores,          │
│         cross-domain results                             │
│  Writes: program.md, launch config, meta-loop params     │
│  Decides: continue / stop / redesign                     │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  META-RRMA (unchanged)                              │ │
│  │  meta-agent: sleeps, compresses, reflects           │ │
│  │                                                     │ │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │ │
│  │  │work 0│ │work 1│ │work 2│ │work 3│  blackboard   │ │
│  │  └──────┘ └──────┘ └──────┘ └──────┘              │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

Three levels. Workers explore. Meta-agent compresses. Outer agent judges and edits.

---

## The outer agent's job

### Every N minutes during a run (monitoring):

Read `meta-blackboard.md` and `results.tsv`. Compute:

```
process_quality:
  papers_cited        = grep -c "arxiv\|paper\|et al" blackboard.md
  architecture_count  = grep -c "^class " sae.py
  ablation_pairs      = count experiment pairs differing by 1 variable in results.tsv
  simplification_moves = count experiments where score UP and param_count DOWN
  explanation_ratio   = lines containing "because|why|mechanism" / total blackboard lines
  axis_diversity      = distinct categories in results.tsv design column
```

Score each 0-5. Sum = process quality (0-30).

### Stopping rules:

| process_quality | score_trend | blind_spots | action |
|---|---|---|---|
| < 10 | improving | any | **STOP: hacking.** Agents gaming the metric. |
| < 10 | flat | any | **STOP: failing.** Neither researching nor succeeding. |
| ≥ 10 | improving | any | **CONTINUE.** Real work happening. |
| ≥ 10 | flat (10+ exp) | nonempty | **REDESIGN.** Scaffold blocking exploration. |
| ≥ 10 | flat (10+ exp) | empty | **STOP: done.** Exhausted the search space. |

Score trend = improvement over last 10 experiments. "Flat" = < 0.5% gain.

### When REDESIGN triggers:

The outer agent reads:
1. Current `program.md`, `meta-blackboard.md`, `blackboard.md`
2. The retrospectives (inherited taste — your v1→v3 lessons)
3. The blind spots list

Then edits the scaffold. The modification space is small and explicit:

| What it can edit | Examples |
|---|---|
| `program.md` | Add hints, change framing, add "re-evaluate failed ideas after breakthroughs" |
| Agent count | 2→4→8 |
| Turn budget | 100→200→500 |
| Meta-agent interval | 15m→30m→60m |
| Planning trigger | Round-based → stagnation-based ("after 10 flat experiments, force planning") |
| Blackboard rules | Add "must cite a paper before proposing architecture" |
| Initial blackboard seed | Pre-populate with dead ends from previous generation |

What it **cannot** edit: `engine.py`, `run.sh`, domain harness. The outer agent shapes the research process, not the research infrastructure.

### After STOP:

Run the same scaffold on a second domain (the canary). Compare process quality across domains:
- Both high → scaffold is good, generalize
- High on one, low on other → domain-specific issue, investigate
- Both low → scaffold is broken, redesign

---

## The taste document

The outer agent's prompt includes a "taste document" — your retrospectives distilled into principles. This is the inherited judgment from v1→v3:

```markdown
## Inherited principles (from human v1→v3 redesigns)

1. Less protocol = more reasoning context = better science.
   The winning v3 had no roles, no CLAIM/RESPONSE. Plain blackboard.

2. If agents config-tune a known architecture without citing papers
   or inventing new approaches, the benchmark is too easy OR
   the program.md doesn't push hard enough. Fix one or both.

3. Planning should trigger on stagnation, not round count.
   v3's round-5 planning never fired. Stagnation-triggered would have
   helped at experiment 30.

4. Re-evaluate previously-failed ideas after fundamental recipe changes.
   200M hurt LISTA but helped Reference. The system should re-test
   old failures when the context changes.

5. Watch for the "architecture vs. training" phase transition.
   v3's breakthrough came when agents pivoted from encoder complexity
   to training curriculum. If agents are stuck on one axis, the
   scaffold should make the other axis visible.

6. Confirmation is a feature, not waste. Multiple agents independently
   confirming a result provides statistical confidence. Don't optimize
   it away.

7. Simplification is a signal of maturity. When agents drop complexity
   and scores go UP, they understand the problem. When they only add
   complexity, they're hill-climbing blindly.

8. The blackboard IS the protocol. Don't add coordination mechanisms
   on top of it. The blackboard scales to 1200+ lines without
   breaking because Claude agents are good at long context.
```

This document gets updated by the outer agent after each generation. It's the system's taste, evolving over time — initially seeded by your experience, then refined by the system's own cross-domain results.

---

## Calibration (the hard part, handled honestly)

The outer agent has no blog post to read. It calibrates via:

1. **Theoretical bounds** — if the domain provides one (SAE-bench's probe ceiling), use it
2. **Literature search** — before each run, search for papers/posts/leaderboards on the benchmark. This is what you did (reading chanind's post). The outer agent does it explicitly as step 0
3. **Rate of insight** — track "novel mechanisms discovered per 10 experiments." When this hits zero AND blind spots are empty, you're at the frontier
4. **Cross-domain baseline** — if the same scaffold gets process_quality > 20 on a known-hard domain, trust it on the novel domain too

When none of these provide calibration (genuinely novel problem, no literature, no bounds): the outer agent says so. It reports "run complete, process quality high, but no external calibration available — cannot assess if score is near frontier." This is honest, not a failure. You'd be in the same position without chanind's post.

---

## Implementation

### New files (added to meta-RRMA):

```
meta-rrma/
├── ... (existing files unchanged)
├── outer-loop.sh        ← the generation loop (NEW)
├── diagnose.sh          ← process quality scoring (NEW)
├── taste.md             ← inherited principles (NEW, human-seeded)
├── calibrate.sh         ← literature search + bounds check (NEW)
└── domains/
    ├── primary-domain/   ← the target problem
    └── canary-domain/    ← cross-domain validation
```

### outer-loop.sh (~80 lines)

```bash
#!/bin/bash
# outer-loop.sh — the gardener
# Runs meta-RRMA generations. Stops, diagnoses, redesigns between them.

DOMAIN_DIR="$1"
CANARY_DIR="$2"
MAX_GENERATIONS="${3:-5}"
MONITOR_INTERVAL="${4:-15}"  # minutes between process quality checks

for gen in $(seq 1 $MAX_GENERATIONS); do
    echo "[outer] === Generation $gen ==="

    # Step 0: calibrate (literature search, bounds check)
    bash calibrate.sh "$DOMAIN_DIR" > "$DOMAIN_DIR/calibration.md"

    # Step 1: launch meta-RRMA
    bash launch-agents.sh "$DOMAIN_DIR" 4 200 30

    # Step 2: monitor loop
    while true; do
        sleep $((MONITOR_INTERVAL * 60))

        # Compute process quality + stopping decision
        DECISION=$(bash diagnose.sh "$DOMAIN_DIR")

        case "$DECISION" in
            CONTINUE)
                echo "[outer] Continuing (gen $gen)..."
                ;;
            STOP_HACKING)
                echo "[outer] STOP: hacking detected"
                bash stop-agents.sh
                # Redesign: tighten program.md, add research requirements
                claude -p "$(cat taste.md)

                The agents are config-tuning instead of researching.
                Process quality is low. Read program.md and blackboard.md.
                Edit program.md to force genuine research: require paper
                citations, require ablations, add harder constraints.
                Output ONLY the new program.md content." \
                    --max-turns 3 > "$DOMAIN_DIR/program.md"
                break
                ;;
            STOP_DONE)
                echo "[outer] STOP: search exhausted"
                bash stop-agents.sh
                bash generate-meta-blackboard.sh "$DOMAIN_DIR"
                # Cross-domain validation
                echo "[outer] Running canary..."
                cp "$DOMAIN_DIR/program.md" "$CANARY_DIR/program.md"
                bash launch-agents.sh "$CANARY_DIR" 4 100 30
                # ... wait, diagnose canary, compare
                break
                ;;
            REDESIGN)
                echo "[outer] REDESIGN: scaffold blocking exploration"
                bash stop-agents.sh
                # Diagnose and modify scaffold
                claude -p "$(cat taste.md)

                Read these artifacts from the run:
                $(cat "$DOMAIN_DIR/meta-blackboard.md")
                ---
                $(cat "$DOMAIN_DIR/blackboard.md" | tail -200)
                ---
                Results: $(cat "$DOMAIN_DIR/results.tsv" | tail -30)

                The agents have high process quality but scores are flat.
                Blind spots remain unexplored. Diagnose what about the
                scaffold (program.md, agent count, turn budget, planning
                triggers) is preventing agents from reaching the blind spots.
                Output a JSON object with your recommended changes." \
                    --max-turns 3 > "/tmp/redesign-$$.json"
                # Apply changes...
                break
                ;;
        esac
    done
done
```

### diagnose.sh (~60 lines)

```bash
#!/bin/bash
# diagnose.sh — compute process quality and stopping decision
# Output: one of CONTINUE, STOP_HACKING, STOP_DONE, REDESIGN

DOMAIN_DIR="$1"

# Count process quality signals
PAPERS=$(grep -ciE "arxiv|paper|et al\.|gregor|lecun" "$DOMAIN_DIR/blackboard.md" 2>/dev/null || echo 0)
CLASSES=$(grep -c "^class " "$DOMAIN_DIR/sae.py" 2>/dev/null || echo 0)
EXPLANATIONS=$(grep -ciE "because|mechanism|why this works|the reason" "$DOMAIN_DIR/blackboard.md" 2>/dev/null || echo 0)
TOTAL_LINES=$(wc -l < "$DOMAIN_DIR/blackboard.md" 2>/dev/null | tr -d ' ')

# Simple process quality score (0-30 scale)
PQ=0
[ "$PAPERS" -gt 0 ] && PQ=$((PQ + 5))
[ "$PAPERS" -gt 3 ] && PQ=$((PQ + 5))
[ "$CLASSES" -gt 3 ] && PQ=$((PQ + 5))
[ "$CLASSES" -gt 10 ] && PQ=$((PQ + 5))
[ "$EXPLANATIONS" -gt 5 ] && PQ=$((PQ + 5))
[ "$EXPLANATIONS" -gt 20 ] && PQ=$((PQ + 5))

# Score trajectory (last 10 experiments)
BEST_RECENT=$(tail -10 "$DOMAIN_DIR/results.tsv" | awk -F'\t' '{print $2}' | sort -rn | head -1)
BEST_PRIOR=$(head -n -10 "$DOMAIN_DIR/results.tsv" | awk -F'\t' '{print $2}' | sort -rn | head -1)
# ... compute delta, check if flat

# Blind spots
BLIND_SPOTS=$(grep -c "blind\|untried\|never tried" "$DOMAIN_DIR/meta-blackboard.md" 2>/dev/null || echo 0)

# Decision matrix
TOTAL_EXP=$(wc -l < "$DOMAIN_DIR/results.tsv" | tr -d ' ')
if [ "$TOTAL_EXP" -lt 10 ]; then
    echo "CONTINUE"  # too early to judge
elif [ "$PQ" -lt 10 ]; then
    echo "STOP_HACKING"
elif [ "$FLAT" = "true" ] && [ "$BLIND_SPOTS" -gt 0 ]; then
    echo "REDESIGN"
elif [ "$FLAT" = "true" ] && [ "$BLIND_SPOTS" -eq 0 ]; then
    echo "STOP_DONE"
else
    echo "CONTINUE"
fi
```

### Estimated size

| Component | Lines | Status |
|---|---|---|
| meta-RRMA (existing) | 230 | Keep unchanged |
| outer-loop.sh | ~80 | New |
| diagnose.sh | ~60 | New |
| calibrate.sh | ~40 | New |
| taste.md | ~50 | New, human-seeded |
| **Total v4** | **~460** | **~230 new lines** |

---

## What this is

A system that runs multi-agent research, watches the process (not just scores), stops when it detects gaming or exhaustion, and redesigns its own scaffold when agents are stuck. It inherits your taste as a document. It validates changes across domains. It's honest about what it can't calibrate.

## What this is not

- Not AGI. It recombines known scaffold patterns. It can't invent a coordination mechanism nobody has thought of.
- Not omniscient. On genuinely novel problems with no literature, it can't tell you if 0.92 is good or terrible.
- Not a replacement for the first human run. Someone has to seed `taste.md` with initial principles. After that, the system refines them.

## The recursive part

After each generation, the outer agent appends to `taste.md`:

```
## Generation N lesson (auto-generated)
- Scaffold change: [what was modified]
- Effect on process quality: [before → after]
- Effect on score: [before → after]
- Cross-domain: [generalized / didn't generalize]
- Principle: [extracted rule]
```

Over generations, `taste.md` grows from your 8 seed principles into a richer document. The system's taste improves from its own experience. The outer agent reads the full `taste.md` before every redesign decision — so generation 5's redesign is informed by lessons from generations 1-4.

This is the self-recursive part. Not recursion on the problem (that's the inner loop). Recursion on the system itself.

---

## First test

Run v4 on SAE-bench (the domain you already have results for). See if the outer agent:
1. Detects hacking if you start with the easy benchmark variant (it should STOP_HACKING)
2. Lets the hard variant run (it should CONTINUE)
3. Triggers REDESIGN if you deliberately cripple program.md (remove all research hints)
4. Reaches similar final scores to v3 without you seeding chanind's knowledge

Test 4 is the real one. Strip program.md back to "improve F1 on this SAE benchmark" with no architecture hints. Seed taste.md with your 8 principles but nothing domain-specific. See what happens.

If it gets to 0.95+ through genuine research (high process quality), the system works.
If it config-tunes to 0.92 and the outer agent correctly calls STOP_HACKING and redesigns, the system works differently but still works.
If it config-tunes to 0.92 and the outer agent says CONTINUE, taste.md needs better principles.

---

## Open question

Should the outer agent be a long-running Claude session (like the workers) or a stateless script that runs `claude -p` on each check (like the meta-agent)?

The meta-agent model (stateless, periodic) is simpler and more robust. The long-running model retains context but can crash. I'd start stateless — the outer agent reads all artifacts fresh each time, applies taste.md, makes a decision. No state to lose.

This matches your design principle: the blackboard IS the memory. For the outer agent, taste.md + generation logs ARE the memory.

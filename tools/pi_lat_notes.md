# pi loop / lat.md — Design Notes

Source: 49-min video, Yury Selivanov (@1st1) pitching lat.md to Armin Ronacher (@mitsuhiko)
Transcript: tools/pi_lat_transcript.txt
Project: https://github.com/1st1/lat.md (npm: `npm install -g latmd`)

---

## What lat.md is

A **knowledge graph in Markdown** that lives alongside your code in git, updated by agents as
they work. Obsidian-style WikiLinks (`[[section]]`) + source code links (tree-sitter parsed).
The agent maintains the spec; humans review the spec diff, not the code diff.

---

## Core Commands

```
lat init          # scaffold latmd/ dir, install hooks into claude/piper/opencode
lat check         # validate link integrity (spec ↔ code), runs as stop hook
lat search <q>    # RAG over knowledge base (libsql vectors db, local)
lat section <id>  # expand section + incoming + outgoing links
lat locate <q>    # fuzzy find sections (Levenshtein distance)
```

---

## Key Design Decisions

### 1. RAG over structured knowledge, not raw code
- `lat search` runs at prompt start (via start hook) → auto-enriches context
- Result includes `lat section` command to expand — builds navigation chain automatically
- "Evidently when the section references the function specifically, it just reads the function block itself"

### 2. Section discipline
- First paragraph ≤ 250 chars — short description for clean RAG retrieval
- Then details in subsequent paragraphs/subsections

### 3. Back-references enforce sync
```yaml
require_code_mentioned: true  # in a test spec file
```
Every leaf section in that doc must have a back-reference FROM source code.
`lat check` enforces it. Runs in CI. Runs as agent stop hook.

### 4. Tree-sitter code parsing
Sections can link to specific functions/methods:
```
resolution is handled by [[source:lattice_ds.resolve_ref]]
```
lat expands this to the function body at call time (not cached). Ripgrep for acceleration.

### 5. Hierarchical modularization
Monorepo: top-level latmd/ for business logic → nested per-service latmd/ for details.
`lat search` sorts local results first, then concatenates from parent.

---

## The "pi" Loop Connection

Yury's system ("pi") uses lat.md as the knowledge backbone for a Ralph-style agent loop:
- Agent runs, updates latmd/ as it works
- Stop hook: `lat check` ensures nothing drifted
- The context is **a graph, not linear** — this is explicitly stated as the key insight
- "It's Elvis's idea — materialized conversation as a graph where leaf paths enforce certain actions"

---

## Relevance to RRMA v5

| lat.md concept | RRMA equivalent | Gap |
|----------------|-----------------|-----|
| `lat search` on prompt | calibration.md load | RRMA loads full file; lat does surgical RAG |
| `lat section` expansion | blackboard.md grep | RRMA has no link navigation |
| `lat check` stop hook | diagnose.sh PQ score | Similar enforcement, different mechanism |
| WikiLinks knowledge graph | DAG schema v2 | RRMA DAG is post-hoc; lat's is live |
| Back-references from code | MISTAKES.md | RRMA telemetry is append-only, not linked |
| Hierarchical latmd/ | domain/ structure | RRMA domain config is flat |
| Section ≤250 char | blackboard entry size | No constraint in RRMA |

### What RRMA should steal:

**1. RAG-first context injection (replaces blackboard dump)**
Instead of: agent reads all of blackboard.md (grows unbounded)
Do: agent runs `rrma-search <current-task>` → gets 3-5 relevant sections + expand links

**2. Back-reference enforcement for telemetry**
`require_mistake_referenced: true` in domain config →
experiments that reference a MISTAKES.md entry explicitly must link to it.
Makes AVOIDED_BY edges explicit rather than regex-inferred.

**3. Live graph, not post-hoc DAG extraction**
lat never lets spec drift from code because enforcement is continuous.
RRMA runs dag_extractor.py after-the-fact and hopes the blackboard is parseable.
Better: write structured entries at experiment time (which MISTAKES.md already does, just not linked).

**4. Hierarchical knowledge for multi-domain**
When running rrma-lean + rrma-r1 + rrma-redteam simultaneously,
agents need to search domain-local + cross-domain knowledge separately.
lat's hierarchical search gives this for free.

---

## What lat.md is NOT

- Not a coding agent (Yury explicitly: "I'm not doing that")
- Not a replacement for blackboard — it's orthogonal
- Not a memory system — it's a live, versioned, linked spec

---

## v5 Integration Sketch

```
~/.rrma/<domain>/
  latmd/              ← NEW: lat-style knowledge graph
    decisions.md      ← why we tried what we tried
    mistakes.md       ← agent-written, lat-linked to experiments
    discoveries.md    ← env facts, lat-linked to experiments
    desires.md        ← what agents wished they had
    calibration.md    ← literature search results, linked to experiments
  blackboard.md       ← KEEP: append-only shared state (coarse grained)
  results.tsv         ← KEEP: score ledger
```

Agents get:
- `rrma-search <query>` → lat search on latmd/ → surgical context, not full blackboard
- `rrma-expand <section>` → expand section + links
- MISTAKES/DESIRES/LEARNINGS entries include `[[experiment:exp014]]` WikiLinks → machine-readable DAG edges

---

## Key Quote

> "English allows you to compress knowledge by a lot. My assumption is that the compression
> benefits are going to be real. If the assumption is correct, we can continue tweaking agents.
> The agent no longer needs to grab through your code base excessively."

> "It's not flat agents.md or skills — it's a graph that's tied to your code base."

> "The context is linear — this is the problem. What if your session context would also be a graph
> where you can have leaf paths that enforce certain actions with the context expression new?"

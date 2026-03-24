# RRMA-Data — Placeholder Domain

## The Idea

Auto-research for data pipelines. Underrated vs model training optimization.

Inspired by: tweet from @... (March 24, 2026) — Claude given access to CPU cluster,
iteratively refined filters to retrieve a domain subset of FineWeb in hours vs 2-3 days.

## The Thesis

Data has fuelled LLM progress more than architecture innovations.
The long tail of small, niche data sources is now accessible via agents that can:

- Download from many different sources
- Normalize into uniform format
- Do detailed EDA: find patterns and outliers
- Look at 100s of samples and take detailed notes
- Iterate on data filtering by examining more samples
- Make pipelines robust and scalable

Code agents can do this work that was previously labour-intensive and detail-heavy.

## What This Domain Would Look Like

Agents are given:
- A target data task (e.g. "build a clean math reasoning dataset from 5 messy sources")
- Access to raw data sources (HuggingFace datasets, APIs, files)
- A quality metric (e.g. held-out benchmark performance, dedup rate, filter recall)

Blackboard becomes a **data quality log** — agents share what they found in samples 0-100,
others build filters from those observations.

## Related

- **DABstep** (adyen/DABstep on HuggingFace) — multi-step reasoning over payment transaction
  data. 460 questions, easy/hard. CPU-only. Could be a first concrete task for this domain.
  The harder version: agents build the pipeline that makes the benchmark solvable.

- **FineWeb subset extraction** — already done manually, could be a domain harness.
  Give agents the FineWeb filters, let them iterate on domain-specific subsets.

- **vincentoh/sandbagging-agent-traces** — example of a curated data artifact that
  came from manual work. This kind of curation is what the domain would automate.

## Status

Placeholder. Not yet implemented.

## When to build this

When rrma-r1 and v5 SAE runs give clearer signal on what the loop can handle.
Data research is a wider search space than model training — needs the loop to be mature first.

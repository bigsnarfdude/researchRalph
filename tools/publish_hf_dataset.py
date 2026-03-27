#!/usr/bin/env python3
"""
publish_hf_dataset.py — Package and push RRMA research traces to HuggingFace

Usage:
    python3 tools/publish_hf_dataset.py --domain rrma-r1 --repo vincentoh/rrma-r1-research-traces

Creates:
    - experiments split: labeled (design, score, keep/discard, context)
    - traces split: raw agent sessions with thinking + tool calls
    - blackboard: shared research log as single document
"""

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


README = """---
license: apache-2.0
task_categories:
- reinforcement-learning
- text-generation
language:
- en
tags:
- autonomous-research
- rl-training
- agent-traces
- grpo
- research-agent
- multi-agent
size_categories:
- n<1K
---

# RRMA-R1 Research Agent Traces

Labeled traces from an autonomous multi-agent RL research loop (ResearchRalph).
The agents were tasked with improving GSM8K pass@1 on Qwen2.5-1.5B-Instruct
above the 0.620 baseline. Final best: **0.825** (MAJ-16@temp=0.7).

## What this is

RRMA (ResearchRalph Multi-Agent) runs Claude agents in an outer loop where they:
1. Read a shared blackboard (prior experiments + results)
2. Propose a new experiment (training method / hyperparameters)
3. Run the experiment (30–90 min on RTX 4070 Ti)
4. Append result + insight to the blackboard
5. Repeat

This dataset captures 46 complete experiment cycles across 2 agents, 10 generations.

## Splits

### `experiments`
One row per experiment. Fields:
- `exp_id`: experiment identifier (EXP-001 … MAJ-16)
- `score`: GSM8K pass@1 achieved
- `status`: `keep` or `discard` (human + verifier certified)
- `design`: one-word method type (GRPO-iter, majority-vote, curriculum-GRPO, …)
- `description`: agent's own description of what they tried
- `agent`: which agent ran it (agent0 / agent1)
- `iteration_n`: position in the research sequence
- `current_best_before`: best score at time of proposal
- `blackboard_context`: last 500 chars of blackboard before this experiment

### `traces`
Raw agent session logs (4 sessions, ~2700 messages). Each message has:
- `type`: user / assistant / progress / queue-operation
- `message`: full content including thinking blocks (extended reasoning)
- `timestamp`, `sessionId`, `uuid`

Claude's extended thinking traces are included — this is the agent's
internal reasoning while deciding what experiment to run next.

### `blackboard`
The shared research log as a single markdown document. Append-only record
of every experiment's outcome, observations, and next hypotheses.

## Intended uses

- Training action classifiers (predict keep/discard before running experiments)
- Research agent fine-tuning (learn to propose good RL experiments)
- Process quality analysis (what reasoning patterns precede breakthroughs?)
- Multi-agent coordination study (how do agents build on each other's work?)

## Key findings in the data

| Experiment | Design | Score | Note |
|------------|--------|-------|------|
| EXP-001 | REINFORCE-group | 0.660 | Baseline REINFORCE |
| EXP-008 | GRPO-iter | 0.760 | Training ceiling |
| MAJ-16 | majority-vote | 0.825 | Inference ceiling, final best |

GRPO-iter variants: 100% keep rate. REINFORCE / ReST variants: <35% keep rate.

## Provenance

- Hardware: RTX 4070 Ti (12GB), Ubuntu 22.04
- Model trained: Qwen2.5-1.5B-Instruct
- Agent model: claude-opus-4-6 (extended thinking enabled)
- Framework: researchRalph v4 (https://github.com/bigsnarfdude/researchRalph)
- Run date: March 2026
- Note: agent traces scrubbed of all PII and server-specific paths before publication

## Citation

```bibtex
@misc{rrma-r1-traces-2026,
  title={RRMA-R1 Research Agent Traces: Autonomous RL Experiment Discovery},
  author={Vincent Oh},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/bigsnarfdude/rrma-r1-research-traces}
}
```
"""


def load_results(tsv_path):
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            try:
                rows.append({
                    "exp_id": row["EXP-ID"].strip(),
                    "score": float(row["score"]) if row["score"].strip() not in ("-", "") else None,
                    "train_min": float(row.get("train_min", 0) or 0),
                    "status": row["status"].strip().lower(),
                    "description": row.get("description", "").strip(),
                    "agent": row.get("agent", "").strip(),
                    "design": row.get("design", "").strip(),
                    "iteration_n": i,
                })
            except (ValueError, KeyError):
                pass
    return rows


def add_blackboard_context(rows, blackboard_path):
    if not Path(blackboard_path).exists():
        for r in rows:
            r["blackboard_context"] = ""
            r["current_best_before"] = 0.0
        return rows

    text = Path(blackboard_path).read_text()
    # Split into sections
    sections = re.split(r"\n(?=##\s+EXP-|\n##\s+Active)", text)
    section_map = {}
    for s in sections:
        m = re.match(r"##\s+(EXP-\w+|[A-Z]+-\d+)", s)
        if m:
            section_map[m.group(1)] = s

    current_best = 0.0
    seen_text = text[:500]  # preamble always available

    for r in rows:
        r["current_best_before"] = current_best
        # Blackboard context = everything before this experiment's section
        exp_pos = text.find(f"## {r['exp_id']}")
        r["blackboard_context"] = text[:exp_pos][-500:].strip() if exp_pos > 0 else seen_text[-500:]

        if r["status"] == "keep" and r["score"] and r["score"] > current_best:
            current_best = r["score"]

    return rows


def scrub_pii(text):
    """Remove PII — file paths, IPs, real emails, server names, API keys."""
    text = re.sub(r"/home/[a-zA-Z0-9_-]+/", "/home/user/", text)
    text = re.sub(r"/Users/[a-zA-Z0-9_-]+/", "/Users/user/", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", text)
    # Real emails only (exclude code patterns like torch.no_grad, model.fit)
    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\b", "[EMAIL]", text)
    # API keys
    text = re.sub(r"\bhf_[a-zA-Z0-9]{10,}\b", "[HF_TOKEN]", text)
    text = re.sub(r"\bsk-[a-zA-Z0-9]{20,}\b", "[API_KEY]", text)
    text = re.sub(r"\bghp_[a-zA-Z0-9]{30,}\b", "[GH_TOKEN]", text)
    # Institution-specific server names
    text = re.sub(r"\b(nigel|garibaldi|castle|andromeda|birszoom)\.birs\.ca\b", "[SERVER]", text)
    text = re.sub(r"\bbirs\.ca\b", "[INSTITUTION]", text)
    return text


def scrub_trace(message):
    """Scrub a single trace message."""
    msg = dict(message)
    # Remove API keys if present in content
    for field in ("message", "content"):
        if field in msg and isinstance(msg[field], str):
            msg[field] = scrub_pii(msg[field])
        elif field in msg and isinstance(msg[field], dict):
            msg[field] = json.loads(scrub_pii(json.dumps(msg[field])))
    return msg


def package_dataset(domain_dir, output_dir):
    domain_dir = Path(domain_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    # 1. Experiments split
    tsv = domain_dir / "results.tsv"
    blackboard = domain_dir / "blackboard.md"
    rows = load_results(tsv)
    rows = add_blackboard_context(rows, blackboard)

    with open(output_dir / "data" / "experiments.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  experiments: {len(rows)} rows")

    # 2. Traces split
    logs_dir = domain_dir / "logs"
    trace_count = 0
    traces_out = output_dir / "data" / "traces"
    traces_out.mkdir(exist_ok=True)
    if logs_dir.exists():
        for jsonl_file in sorted(logs_dir.glob("*.jsonl")):
            out_file = traces_out / jsonl_file.name
            with open(jsonl_file) as fin, open(out_file, "w") as fout:
                for line in fin:
                    try:
                        msg = json.loads(line)
                        msg = scrub_trace(msg)
                        fout.write(json.dumps(msg) + "\n")
                        trace_count += 1
                    except json.JSONDecodeError:
                        pass
    print(f"  traces: {trace_count} messages across {len(list(traces_out.glob('*.jsonl')))} sessions")

    # 3. Blackboard
    if blackboard.exists():
        shutil.copy(blackboard, output_dir / "data" / "blackboard.md")
        print(f"  blackboard: {blackboard.stat().st_size // 1024}KB")

    # 4. README
    with open(output_dir / "README.md", "w") as f:
        f.write(README)
    print(f"  README.md written")

    return output_dir


def push_to_hf(output_dir, repo_id, token=None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found. Set HF_TOKEN env var.")
        return False

    api = HfApi(token=token)
    output_dir = Path(output_dir)

    print(f"\nPushing to {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            path_in_repo = file_path.relative_to(output_dir)
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(path_in_repo),
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  uploaded: {path_in_repo}")

    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domains/rrma-r1")
    parser.add_argument("--repo", default="vincentoh/rrma-r1-research-traces")
    parser.add_argument("--output", default="/tmp/rrma-r1-hf-dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Packaging {args.domain} → {args.output}")
    package_dataset(args.domain, args.output)

    if args.dry_run:
        print(f"\nDry run complete. Files in {args.output}")
        import subprocess
        subprocess.run(["find", args.output, "-type", "f"], check=False)
        return

    push_to_hf(args.output, args.repo)


if __name__ == "__main__":
    main()

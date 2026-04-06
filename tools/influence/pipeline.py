#!/usr/bin/env python3
"""
pipeline.py — full influence graph analysis pipeline for all RRMA domains

Runs in two phases:
  1. Extract events from all domain logs (via extract_events.py logic)
  2. Build influence graphs + write per-domain influence_summary.json

Then aggregates results into:
  tools/influence/results/influence_aggregate.csv
  tools/influence/results/influence_aggregate.json
  tools/influence/results/pipeline_report.txt

Usage:
  python3 tools/influence/pipeline.py [--domain DOMAIN] [--dot] [--skip-extract]

  --domain       only process this domain
  --dot          write .dot GraphViz files per domain
  --skip-extract skip the extraction step (use existing events.jsonl files)
  --verbose      verbose extraction output

Run from repo root:
  cd ~/researchRalph
  python3 tools/influence/pipeline.py
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

# Add tools/ to path so we can import sibling modules
TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from influence.extract_events import process_domain as extract_domain
from influence.build_graph import process_domain as build_domain

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_DIR = REPO_ROOT / "domains"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_pipeline(
    domain_filter: str | None = None,
    write_dot: bool = False,
    skip_extract: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """
    Run the full pipeline across all domains.
    Returns list of per-domain summary dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if domain_filter:
        candidates = [DOMAINS_DIR / domain_filter]
    else:
        candidates = sorted(d for d in DOMAINS_DIR.iterdir() if d.is_dir())

    summaries = []
    extract_totals = {"domains": 0, "events": 0}
    graph_totals = {"domains": 0}

    print("=" * 60)
    print("RRMA Influence Graph Pipeline")
    print(f"Domains dir: {DOMAINS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Started:     {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    print()

    # --- Phase 1: Extract events ---
    if not skip_extract:
        print("Phase 1: Extracting events from agent logs")
        print("-" * 40)
        for domain_dir in candidates:
            if not domain_dir.is_dir():
                print(f"  WARNING: {domain_dir} not found")
                continue
            if verbose:
                print(f"\n=== {domain_dir.name} ===")
            n = extract_domain(domain_dir, verbose=verbose)
            if n > 0:
                extract_totals["domains"] += 1
                extract_totals["events"] += n
        print(f"\nExtraction complete: {extract_totals['domains']} domains, "
              f"{extract_totals['events']} events\n")
    else:
        print("Phase 1: Skipping extraction (--skip-extract)\n")

    # --- Phase 2: Build graphs ---
    print("Phase 2: Building influence graphs")
    print("-" * 40)
    for domain_dir in candidates:
        if not domain_dir.is_dir():
            continue
        result = build_domain(domain_dir, write_dot=write_dot)
        if result is not None:
            summaries.append(result)
            graph_totals["domains"] += 1
    print(f"\nGraph building complete: {graph_totals['domains']} domains\n")

    # --- Phase 3: Aggregate ---
    if summaries:
        write_aggregate(summaries)
        write_report(summaries, extract_totals)

    return summaries


def write_aggregate(summaries: list[dict]):
    """Write CSV and JSON aggregates across all domains."""
    # CSV: one row per (domain, agent)
    csv_path = RESULTS_DIR / "influence_aggregate.csv"
    fieldnames = [
        "domain", "agent", "writes", "reads",
        "out_degree", "in_degree", "influence_ratio",
        "total_agents", "total_events", "total_edge_weight",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            domain = summary["domain"]
            n_agents = len(summary["agents"])
            total_events = summary["total_events"]
            total_weight = summary["total_edge_weight"]
            for agent, stats in summary["nodes"].items():
                writer.writerow({
                    "domain": domain,
                    "agent": agent,
                    "writes": stats["writes"],
                    "reads": stats["reads"],
                    "out_degree": stats["out_degree"],
                    "in_degree": stats["in_degree"],
                    "influence_ratio": stats["influence_ratio"],
                    "total_agents": n_agents,
                    "total_events": total_events,
                    "total_edge_weight": total_weight,
                })
    print(f"Wrote {csv_path}")

    # JSON: full aggregate
    json_path = RESULTS_DIR / "influence_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {json_path}")


def write_report(summaries: list[dict], extract_totals: dict):
    """Write a human-readable pipeline report."""
    report_path = RESULTS_DIR / "pipeline_report.txt"
    buf = StringIO()

    def p(*args, **kwargs):
        print(*args, **kwargs, file=buf)

    p("RRMA Influence Graph Pipeline Report")
    p(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    p("=" * 70)
    p()

    total_events = sum(s["total_events"] for s in summaries)
    total_weight = sum(s["total_edge_weight"] for s in summaries)
    p(f"Domains processed:   {len(summaries)}")
    p(f"Total events:        {total_events}")
    p(f"Total influence arcs:{total_weight}")
    p()

    # Per-domain summary table
    p(f"{'Domain':<40} {'Agents':>6} {'Events':>7} {'Arcs':>6}")
    p("-" * 62)
    for s in sorted(summaries, key=lambda x: -x["total_edge_weight"]):
        p(f"{s['domain']:<40} {len(s['agents']):>6} {s['total_events']:>7} {s['total_edge_weight']:>6}")
    p()

    # Top influencers across all domains
    p("Top agents by out_degree (influence output) across all domains:")
    p("-" * 50)
    agent_rows = []
    for s in summaries:
        for agent, stats in s["nodes"].items():
            agent_rows.append({
                "domain": s["domain"],
                "agent": agent,
                **stats,
            })
    agent_rows.sort(key=lambda x: -x["out_degree"])
    p(f"  {'Domain':<35} {'Agent':<10} {'out':>5} {'in':>5} {'ratio':>7}")
    for row in agent_rows[:20]:
        p(f"  {row['domain']:<35} {row['agent']:<10} "
          f"{row['out_degree']:>5} {row['in_degree']:>5} {row['influence_ratio']:>7.3f}")
    p()

    # Most-used influence channels (file -> top edge)
    p("Top influence channels by file:")
    p("-" * 50)
    file_edge_totals: dict[str, dict] = {}
    for s in summaries:
        for fname, edge_map in s.get("edges_by_file", {}).items():
            if fname not in file_edge_totals:
                file_edge_totals[fname] = {}
            for edge_key, count in edge_map.items():
                file_edge_totals[fname][edge_key] = (
                    file_edge_totals[fname].get(edge_key, 0) + count
                )
    for fname in sorted(file_edge_totals, key=lambda f: -sum(file_edge_totals[f].values())):
        total = sum(file_edge_totals[fname].values())
        top_edge = max(file_edge_totals[fname], key=file_edge_totals[fname].get)
        top_count = file_edge_totals[fname][top_edge]
        p(f"  {fname:<30} total={total:>4}  top: {top_edge} ({top_count})")
    p()

    # Chaos vs normal: look for domains with "chaos" in name
    chaos_domains = [s for s in summaries if "chaos" in s["domain"]]
    normal_domains = [s for s in summaries if "chaos" not in s["domain"]]

    if chaos_domains and normal_domains:
        p("Chaos vs Normal domain comparison:")
        p("-" * 50)

        def avg_ratio(domain_list):
            ratios = []
            for s in domain_list:
                for stats in s["nodes"].values():
                    ratios.append(stats["influence_ratio"])
            return sum(ratios) / len(ratios) if ratios else 0.0

        def avg_arcs_per_event(domain_list):
            vals = []
            for s in domain_list:
                if s["total_events"] > 0:
                    vals.append(s["total_edge_weight"] / s["total_events"])
            return sum(vals) / len(vals) if vals else 0.0

        p(f"  Normal domains ({len(normal_domains)}):")
        p(f"    Avg influence ratio:    {avg_ratio(normal_domains):.3f}")
        p(f"    Avg arcs/event:         {avg_arcs_per_event(normal_domains):.3f}")
        p(f"  Chaos domains ({len(chaos_domains)}):")
        p(f"    Avg influence ratio:    {avg_ratio(chaos_domains):.3f}")
        p(f"    Avg arcs/event:         {avg_arcs_per_event(chaos_domains):.3f}")
        p()

    report_text = buf.getvalue()
    report_path.write_text(report_text)
    print(f"Wrote {report_path}")
    print()
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", help="Process only this domain name")
    parser.add_argument("--dot", action="store_true",
                        help="Write GraphViz .dot files per domain")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, use existing events.jsonl files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose extraction output")
    args = parser.parse_args()

    summaries = run_pipeline(
        domain_filter=args.domain,
        write_dot=args.dot,
        skip_extract=args.skip_extract,
        verbose=args.verbose,
    )

    if not summaries:
        print("No domains with events found. "
              "Check that logs/*.jsonl files exist under domains/.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

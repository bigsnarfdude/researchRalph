#!/usr/bin/env python3
"""
build_graph.py — build influence graph from events.jsonl

Reads events.jsonl in a domain directory, builds a directed influence graph
where an edge A -> B means:
  Agent A wrote (or edited) shared file F, then agent B read F next.

The weight of edge A -> B is the number of such write-then-read sequences.

Outputs:
  influence_summary.json  — per-domain graph + node stats
  influence_graph.dot     — GraphViz DOT format (optional, for visualization)

Usage:
  python3 tools/influence/build_graph.py <domain_dir> [--dot]
  python3 tools/influence/build_graph.py domains/nirenberg-1d --dot

  Or import and call build_graph(events) directly from pipeline.py.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(events_path: Path) -> list[dict]:
    """Load events from events.jsonl, sorted by timestamp."""
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    # Sort: real ISO8601 timestamps sort correctly as strings.
    # Synthetic timestamps ("synthetic://...") sort before real ones,
    # but within a domain they'll only appear if all logs are synthetic.
    events.sort(key=lambda e: e.get("ts", ""))
    return events


def build_graph(events: list[dict]) -> dict:
    """
    Build influence graph from a list of events.

    Returns a dict with:
      edges: {(writer, reader): {"weight": int, "files": {file: count}}}
      nodes: {agent: {"writes": int, "reads": int}}
      file_sequences: {file: [list of (ts, agent, op) tuples]}
      edges_by_file: {file: {(writer, reader): count}}
    """
    # Group events by file, preserving time order
    file_events: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        file_events[ev["file"]].append(ev)

    # For each file, find write -> read sequences
    edges: dict[tuple, dict] = defaultdict(lambda: {"weight": 0, "files": defaultdict(int)})
    node_writes: dict[str, int] = defaultdict(int)
    node_reads: dict[str, int] = defaultdict(int)
    edges_by_file: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for fname, fevents in file_events.items():
        # Events are already sorted by timestamp globally; per-file they're in order too
        last_writer: str | None = None
        for ev in fevents:
            agent = ev["agent"]
            op = ev["op"]

            if op in ("Write", "Edit"):
                node_writes[agent] += 1
                last_writer = agent
            elif op == "Read":
                node_reads[agent] += 1
                if last_writer is not None and last_writer != agent:
                    # Influence edge: last_writer -> this reader
                    edge = (last_writer, agent)
                    edges[edge]["weight"] += 1
                    edges[edge]["files"][fname] += 1
                    edges_by_file[fname][edge] += 1
            # Grep is a read without modifying state; treat same as Read
            elif op == "Grep":
                node_reads[agent] += 1
                if last_writer is not None and last_writer != agent:
                    edge = (last_writer, agent)
                    edges[edge]["weight"] += 1
                    edges[edge]["files"][fname] += 1
                    edges_by_file[fname][edge] += 1

    # Compute per-node stats
    all_agents = set(list(node_writes.keys()) + list(node_reads.keys()))
    nodes = {}
    for agent in sorted(all_agents):
        out_deg = sum(v["weight"] for (w, r), v in edges.items() if w == agent)
        in_deg = sum(v["weight"] for (w, r), v in edges.items() if r == agent)
        total = out_deg + in_deg
        nodes[agent] = {
            "writes": node_writes[agent],
            "reads": node_reads[agent],
            "out_degree": out_deg,
            "in_degree": in_deg,
            "influence_ratio": round(out_deg / total, 4) if total > 0 else 0.0,
        }

    # Serialize edges with string keys (JSON doesn't support tuple keys)
    edges_serializable = {}
    for (w, r), v in edges.items():
        key = f"{w}->{r}"
        edges_serializable[key] = {
            "writer": w,
            "reader": r,
            "weight": v["weight"],
            "files": dict(v["files"]),
        }

    edges_by_file_serializable = {}
    for fname, edge_counts in edges_by_file.items():
        edges_by_file_serializable[fname] = {
            f"{w}->{r}": count for (w, r), count in edge_counts.items()
        }

    return {
        "nodes": nodes,
        "edges": edges_serializable,
        "edges_by_file": edges_by_file_serializable,
        "total_events": len(events),
        "total_edge_weight": sum(v["weight"] for v in edges.values()),
        "agents": sorted(all_agents),
    }


def to_dot(graph: dict, domain_name: str) -> str:
    """Generate GraphViz DOT representation."""
    lines = [
        f'digraph influence_{domain_name.replace("-", "_")} {{',
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor=lightblue];',
        '',
    ]

    # Nodes
    for agent, stats in graph["nodes"].items():
        ratio = stats["influence_ratio"]
        # Color: high ratio = more red (high influence), low = green (high dependency)
        lines.append(
            f'  "{agent}" [label="{agent}\\nout={stats["out_degree"]} in={stats["in_degree"]}\\nratio={ratio:.2f}"];'
        )

    lines.append('')

    # Edges
    for key, edge in graph["edges"].items():
        w = edge["weight"]
        writer = edge["writer"]
        reader = edge["reader"]
        # Edge thickness proportional to weight
        penwidth = max(1.0, min(8.0, w * 0.5))
        lines.append(
            f'  "{writer}" -> "{reader}" [label="{w}", penwidth={penwidth:.1f}];'
        )

    lines.append('}')
    return '\n'.join(lines)


def process_domain(domain_dir: Path, write_dot: bool = False) -> dict | None:
    """
    Build influence graph for one domain.
    Writes influence_summary.json (and optionally influence_graph.dot).
    Returns the summary dict, or None if no events.
    """
    events_path = domain_dir / "events.jsonl"
    if not events_path.exists():
        return None

    events = load_events(events_path)
    if not events:
        return None

    graph = build_graph(events)
    domain_name = domain_dir.name

    summary = {
        "domain": domain_name,
        **graph,
    }

    out_path = domain_dir / "influence_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{domain_name}] wrote influence_summary.json "
          f"({len(graph['agents'])} agents, {graph['total_edge_weight']} influence edges)")

    if write_dot:
        dot_path = domain_dir / "influence_graph.dot"
        dot_src = to_dot(graph, domain_name)
        dot_path.write_text(dot_src)
        print(f"  [{domain_name}] wrote influence_graph.dot")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain_dir",
                        help="Path to domain directory containing events.jsonl")
    parser.add_argument("--dot", action="store_true",
                        help="Also write influence_graph.dot (GraphViz format)")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)
    if not domain_dir.is_dir():
        print(f"ERROR: {domain_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = process_domain(domain_dir, write_dot=args.dot)
    if result is None:
        print(f"No events.jsonl found in {domain_dir}. "
              f"Run extract_events.py first.", file=sys.stderr)
        sys.exit(1)

    # Print a brief summary to stdout
    print("\nNode stats:")
    for agent, stats in sorted(result["nodes"].items()):
        print(f"  {agent:12s}  writes={stats['writes']:3d}  reads={stats['reads']:3d}  "
              f"out_deg={stats['out_degree']:3d}  in_deg={stats['in_degree']:3d}  "
              f"ratio={stats['influence_ratio']:.3f}")


if __name__ == "__main__":
    main()

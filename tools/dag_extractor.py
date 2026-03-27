#!/usr/bin/env python3
"""
dag_extractor.py — Extract DAG edges from blackboard.md + results.tsv

Extracts 4 binary graph features per experiment:
  1. triggered_by_directive  — within 2 experiments of a gardener directive
  2. cites_checkpoint        — explicitly starts from a prior checkpoint
  3. informed_by_breakthrough — after a breakthrough event (majority voting etc.)
  4. informed_by_dead_end    — same design_type as a discard within last 3

Then re-runs LOO-CV comparing tabular-only vs tabular+graph AUC.

Usage:
    python3 tools/dag_extractor.py domains/rrma-r1/results.tsv domains/rrma-r1/blackboard.md
"""

import csv
import json
import re
import sys
from pathlib import Path
from collections import deque


# ── Blackboard parsing ──────────────────────────────────────────────────────

def parse_blackboard(path):
    """
    Returns ordered list of blackboard events:
      {"type": "experiment"|"directive"|"breakthrough"|"observation",
       "id": str, "line_n": int, "text": str}
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    events = []

    for i, line in enumerate(lines):
        # Gardener directives (axis rotation, stop, redesign)
        if re.match(r"^##\s+(Gardener directive|Observation \(gardener)", line, re.I):
            events.append({"type": "directive", "id": f"directive-{i}",
                           "line_n": i, "text": line})

        # Breakthrough events
        elif re.search(r"breakthrough|★★★★★|test.time compute", line, re.I):
            if line.startswith("##"):
                events.append({"type": "breakthrough", "id": f"breakthrough-{i}",
                               "line_n": i, "text": line})

        # Experiment sections
        elif re.match(r"^##\s+(EXP-\w+|MAJ-\w+|Majority[@\s])", line):
            m = re.match(r"^##\s+(EXP-[\w]+|MAJ-[\w@]+)", line)
            exp_id = m.group(1) if m else f"UNK-{i}"
            # Grab the section body (next ~30 lines)
            body = "\n".join(lines[i:i+30])
            events.append({"type": "experiment", "id": exp_id,
                           "line_n": i, "text": body})

    return events


def extract_graph_features(rows, blackboard_path):
    """
    For each row in results.tsv, extract 4 binary graph features.
    Returns list of dicts aligned with rows.
    """
    events = parse_blackboard(blackboard_path)

    # Build ordered list of (event_type, exp_id) for proximity checks
    event_seq = [(e["type"], e["id"]) for e in events]

    # Find breakthrough line number
    breakthrough_line = None
    for e in events:
        if e["type"] == "breakthrough":
            breakthrough_line = e["line_n"]
            break

    # Map exp_id → event index in sequence
    exp_to_seq_idx = {}
    for i, (etype, eid) in enumerate(event_seq):
        if etype == "experiment":
            exp_to_seq_idx[eid] = i

    # Map exp_id → blackboard line number
    exp_to_line = {e["id"]: e["line_n"] for e in events if e["type"] == "experiment"}

    # Map exp_id → full section text
    exp_to_text = {e["id"]: e["text"] for e in events if e["type"] == "experiment"}

    features = []
    recent_discards = deque(maxlen=3)  # track last 3 discard design types

    for r in rows:
        exp_id = r["exp_id"]
        design = r["design"]
        seq_idx = exp_to_seq_idx.get(exp_id, -1)
        exp_line = exp_to_line.get(exp_id, 0)
        section_text = exp_to_text.get(exp_id, "") + " " + r.get("description", "")

        # 1. triggered_by_directive
        # Is there a directive in the 2 events immediately before this experiment?
        triggered = False
        if seq_idx > 0:
            window = event_seq[max(0, seq_idx-3):seq_idx]
            triggered = any(etype == "directive" for etype, _ in window)

        # 2. cites_checkpoint
        # Does the section reference a prior experiment checkpoint?
        cites = bool(re.search(
            r"(from EXP-\w+|from best|from 0\.\d+|from checkpoint|start from|"
            r"iterative.*round|round \d|iter.*from)",
            section_text, re.I
        ))

        # 3. informed_by_breakthrough
        # Is this experiment after the majority voting breakthrough?
        after_breakthrough = (
            breakthrough_line is not None and
            exp_line > breakthrough_line
        )

        # 4. informed_by_dead_end
        # Same design_type as a recent discard?
        dead_end = design in recent_discards

        features.append({
            "triggered_by_directive": int(triggered),
            "cites_checkpoint": int(cites),
            "informed_by_breakthrough": int(after_breakthrough),
            "informed_by_dead_end": int(dead_end),
        })

        # Update recent discards
        if r["status"] == "discard":
            recent_discards.append(design)

    return features


# ── Reuse meta_experiment helpers ───────────────────────────────────────────

def load_results(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            try:
                rows.append({
                    "exp_id": row["EXP-ID"].strip(),
                    "score": float(row["score"]) if row.get("score", "").strip() not in ("-","") else None,
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


def build_features(rows, graph_feats):
    designs = sorted(set(r["design"] for r in rows))
    design_idx = {d: i for i, d in enumerate(designs)}
    agents = sorted(set(r["agent"] for r in rows))
    agent_idx = {a: i for i, a in enumerate(agents)}

    X_tab, X_graph, y, labels = [], [], [], []
    current_best = 0.0

    for r, gf in zip(rows, graph_feats):
        score_delta = (r["score"] or 0) - current_best
        tab = [
            design_idx.get(r["design"], -1),
            agent_idx.get(r["agent"], -1),
            r["iteration_n"],
            current_best,
            score_delta,
            r["train_min"],
        ]
        graph = [
            gf["triggered_by_directive"],
            gf["cites_checkpoint"],
            gf["informed_by_breakthrough"],
            gf["informed_by_dead_end"],
        ]
        X_tab.append(tab)
        X_graph.append(tab + graph)
        y.append(1 if r["status"] == "keep" else 0)
        labels.append(r["exp_id"])

        if r["status"] == "keep" and r["score"] and r["score"] > current_best:
            current_best = r["score"]

    return X_tab, X_graph, y, labels


def loocv(X, y):
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        X_arr, y_arr = np.array(X, dtype=float), np.array(y)
        n = len(y_arr)
        probs = []

        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr, y_tr = X_arr[mask], y_arr[mask]
            X_te = X_arr[[i]]
            if len(set(y_tr)) < 2:
                probs.append(0.5); continue
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(sc.fit_transform(X_tr), y_tr)
            probs.append(clf.predict_proba(sc.transform(X_te))[0][1])

        preds = [int(p > 0.5) for p in probs]
        acc = sum(p == t for p, t in zip(preds, y)) / n
        auc = roc_auc_score(y_arr, probs)
        return acc, auc
    except ImportError:
        return None, None


def main():
    if len(sys.argv) < 3:
        print("Usage: dag_extractor.py results.tsv blackboard.md")
        sys.exit(1)

    results_path, blackboard_path = sys.argv[1], sys.argv[2]
    rows = load_results(results_path)
    graph_feats = extract_graph_features(rows, blackboard_path)

    print(f"\n=== DAG Feature Extraction ===")
    print(f"N={len(rows)} experiments\n")

    # Show extracted features
    print(f"{'EXP-ID':<12} {'dir':>4} {'cite':>5} {'break':>6} {'dead':>5} {'status':<8} design")
    print("-" * 65)
    for r, gf in zip(rows, graph_feats):
        print(f"{r['exp_id']:<12} "
              f"{gf['triggered_by_directive']:>4} "
              f"{gf['cites_checkpoint']:>5} "
              f"{gf['informed_by_breakthrough']:>6} "
              f"{gf['informed_by_dead_end']:>5}  "
              f"{r['status']:<8} {r['design']}")

    # Compare tabular vs tabular+graph
    X_tab, X_graph, y, labels = build_features(rows, graph_feats)
    majority = max(sum(y), len(y)-sum(y)) / len(y)

    acc_tab, auc_tab = loocv(X_tab, y)
    acc_graph, auc_graph = loocv(X_graph, y)

    print(f"\n=== LOO-CV Comparison ===")
    print(f"Majority baseline:     acc={majority:.3f}  auc=0.500")
    print(f"Tabular only:          acc={acc_tab:.3f}  auc={auc_tab:.3f}")
    print(f"Tabular + graph edges: acc={acc_graph:.3f}  auc={auc_graph:.3f}")
    print(f"Graph delta:           acc={acc_graph-acc_tab:+.3f}  auc={auc_graph-auc_tab:+.3f}")

    if auc_graph - auc_tab > 0.05:
        verdict = "GRAPH EDGES ADD SIGNAL — DAG extraction worth building"
    elif auc_graph - auc_tab > 0:
        verdict = "WEAK GRAPH SIGNAL — edges help marginally"
    else:
        verdict = "NO GRAPH SIGNAL — tabular features sufficient at this scale"
    print(f"\nVerdict: {verdict}")

    # Feature distribution
    print(f"\n=== Graph Feature Stats ===")
    for feat in ["triggered_by_directive", "cites_checkpoint",
                 "informed_by_breakthrough", "informed_by_dead_end"]:
        vals = [gf[feat] for gf in graph_feats]
        keep_rate = sum(gf[feat] and rows[i]["status"]=="keep"
                       for i, gf in enumerate(graph_feats)) / max(sum(vals), 1)
        print(f"  {feat:<30} n={sum(vals):>2}/{len(vals)}  keep_rate={keep_rate:.0%}")

    # Save DAG as JSON for future use
    dag = []
    for r, gf in zip(rows, graph_feats):
        dag.append({"exp_id": r["exp_id"], "status": r["status"],
                    "design": r["design"], "score": r["score"], **gf})

    out = Path(results_path).parent / "dag_features.json"
    with open(out, "w") as f:
        json.dump(dag, f, indent=2)
    print(f"\nDAG features saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dag_extractor_v2.py — DAG feature extraction with v2 telemetry nodes

Extends dag_extractor.py (v1) with features from:
  MISTAKES.md  → AVOIDED_BY edges
  DESIRES.md   → MOTIVATED edges
  LEARNINGS.md → BUILT_ON edges

Usage:
    python3 tools/dag_extractor_v2.py domains/rrma-lean/results.tsv \
        domains/rrma-lean/blackboard.md \
        [domains/rrma-lean/MISTAKES.md] \
        [domains/rrma-lean/DESIRES.md] \
        [domains/rrma-lean/LEARNINGS.md]

Telemetry files are optional — v1 features still computed without them.
"""

import csv
import json
import re
import sys
from pathlib import Path
from collections import deque


# ── Telemetry parsing ────────────────────────────────────────────────────────

def parse_telemetry_file(path):
    """
    Parse MISTAKES.md, DESIRES.md, or LEARNINGS.md.
    Returns ordered list of:
      {"exp_id": "EXP-011", "n": 1, "text": "full item text", "tactic": str|None}
    Items are ordered by appearance (= chronological experiment order).
    """
    if not Path(path).exists():
        return []
    text = Path(path).read_text()
    items = []
    current_exp = None

    for line in text.splitlines():
        # Section header: ## EXP-011 or ## exp011 etc.
        m = re.match(r'^##\s+(\w+)', line)
        if m:
            current_exp = m.group(1).lower()
            continue

        # Numbered item: "1. **tactic name** — description"
        m = re.match(r'^\d+\.\s+\*\*(.+?)\*\*', line)
        if m and current_exp:
            tactic = m.group(1).strip()
            items.append({
                "exp_id": current_exp,
                "text": line.strip(),
                "tactic": tactic,
            })

    return items


def build_telemetry_index(mistakes_path, desires_path, learnings_path):
    """
    Returns dict mapping exp_id → {"mistakes": [...], "desires": [...], "learnings": [...]}
    where each list contains items first logged in that experiment.
    """
    idx = {}
    for label, path in [("mistakes", mistakes_path),
                        ("desires", desires_path),
                        ("learnings", learnings_path)]:
        for item in parse_telemetry_file(path):
            exp = item["exp_id"]
            if exp not in idx:
                idx[exp] = {"mistakes": [], "desires": [], "learnings": []}
            idx[exp][label].append(item)
    return idx


# ── v1 blackboard parsing (copied from dag_extractor.py) ────────────────────

def parse_blackboard(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    events = []

    for i, line in enumerate(lines):
        if re.match(r"^##\s+(Gardener directive|Observation \(gardener)", line, re.I):
            events.append({"type": "directive", "id": f"directive-{i}",
                           "line_n": i, "text": line})
        elif re.search(r"breakthrough|★★★★★|test.time compute", line, re.I):
            if line.startswith("##"):
                events.append({"type": "breakthrough", "id": f"breakthrough-{i}",
                               "line_n": i, "text": line})
        elif re.match(r"^##\s+(EXP-\w+|MAJ-\w+|Majority[@\s])", line):
            m = re.match(r"^##\s+(EXP-[\w]+|MAJ-[\w@]+)", line)
            exp_id = m.group(1) if m else f"UNK-{i}"
            body = "\n".join(lines[i:i+30])
            events.append({"type": "experiment", "id": exp_id,
                           "line_n": i, "text": body})

    return events


def extract_v1_features(rows, blackboard_path):
    events = parse_blackboard(blackboard_path)
    event_seq = [(e["type"], e["id"]) for e in events]

    breakthrough_line = None
    for e in events:
        if e["type"] == "breakthrough":
            breakthrough_line = e["line_n"]
            break

    exp_to_seq_idx = {}
    for i, (etype, eid) in enumerate(event_seq):
        if etype == "experiment":
            exp_to_seq_idx[eid] = i

    exp_to_line = {e["id"]: e["line_n"] for e in events if e["type"] == "experiment"}
    exp_to_text = {e["id"]: e["text"] for e in events if e["type"] == "experiment"}

    features = []
    recent_discards = deque(maxlen=3)

    for r in rows:
        exp_id = r["exp_id"]
        design = r["design"]
        seq_idx = exp_to_seq_idx.get(exp_id, -1)
        exp_line = exp_to_line.get(exp_id, 0)
        section_text = exp_to_text.get(exp_id, "") + " " + r.get("description", "")

        triggered = False
        if seq_idx > 0:
            window = event_seq[max(0, seq_idx-3):seq_idx]
            triggered = any(etype == "directive" for etype, _ in window)

        cites = bool(re.search(
            r"(from EXP-\w+|from best|from 0\.\d+|from checkpoint|start from|"
            r"iterative.*round|round \d|iter.*from)",
            section_text, re.I
        ))

        after_breakthrough = (
            breakthrough_line is not None and
            exp_line > breakthrough_line
        )

        dead_end = design in recent_discards

        features.append({
            "triggered_by_directive": int(triggered),
            "cites_checkpoint": int(cites),
            "informed_by_breakthrough": int(after_breakthrough),
            "informed_by_dead_end": int(dead_end),
        })

        if r["status"] == "discard":
            recent_discards.append(design)

    return features


# ── v2 telemetry features ────────────────────────────────────────────────────

def extract_v2_features(rows, telemetry_idx):
    """
    For each row, compute telemetry features based on what was known
    at the time the experiment ran (i.e., from prior experiments only).

    Temporal ordering: items logged under EXP-N are available from EXP-(N+1) onwards.
    We use iteration_n to determine order.
    """
    # Build ordered list of exp_ids by iteration
    exp_order = [r["exp_id"] for r in rows]

    # Accumulate telemetry chronologically
    cumulative_mistakes = []   # list of items known before this exp
    cumulative_desires = []
    cumulative_learnings = []

    features = []

    for i, r in enumerate(rows):
        exp_id = r["exp_id"].lower()
        description = r.get("description", "").lower()

        # Snapshot of what's known at time of this experiment
        n_mistakes = len(cumulative_mistakes)
        n_desires = len(cumulative_desires)
        n_learnings = len(cumulative_learnings)

        # Did this experiment avoid a known mistake?
        avoided = False
        avoided_tactic = None
        for m in cumulative_mistakes:
            tactic_words = re.findall(r'\w+', m["tactic"].lower()) if m.get("tactic") else []
            if any(w in description for w in tactic_words if len(w) > 3):
                avoided = True
                avoided_tactic = m["tactic"]
                break

        # Does this experiment fulfill a prior desire?
        fulfills = False
        desire_text = None
        for d in cumulative_desires:
            desire_words = re.findall(r'\w+', d["text"].lower())
            # require ≥2 content words to match
            matches = sum(1 for w in desire_words if len(w) > 4 and w in description)
            if matches >= 2:
                fulfills = True
                desire_text = d["text"][:80]
                break

        # Does this experiment reference a prior learning?
        built_on = False
        learning_text = None
        for l in cumulative_learnings:
            learning_words = re.findall(r'\w+', l["text"].lower())
            matches = sum(1 for w in learning_words if len(w) > 4 and w in description)
            if matches >= 2:
                built_on = True
                learning_text = l["text"][:80]
                break

        features.append({
            "n_mistakes_at_time": n_mistakes,
            "n_desires_at_time": n_desires,
            "n_learnings_at_time": n_learnings,
            "avoided_known_mistake": int(avoided),
            "mistake_tactic": avoided_tactic,
            "fulfills_prior_desire": int(fulfills),
            "desire_text": desire_text,
            "built_on_learning": int(built_on),
            "learning_text": learning_text,
        })

        # Add this experiment's telemetry to cumulative pool for future exps
        if exp_id in telemetry_idx:
            cumulative_mistakes.extend(telemetry_idx[exp_id]["mistakes"])
            cumulative_desires.extend(telemetry_idx[exp_id]["desires"])
            cumulative_learnings.extend(telemetry_idx[exp_id]["learnings"])

    return features


# ── Shared helpers ───────────────────────────────────────────────────────────

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


def build_features(rows, v1_feats, v2_feats):
    designs = sorted(set(r["design"] for r in rows))
    design_idx = {d: i for i, d in enumerate(designs)}
    agents = sorted(set(r["agent"] for r in rows))
    agent_idx = {a: i for i, a in enumerate(agents)}

    X_tab, X_v1, X_v2, y = [], [], [], []
    current_best = 0.0

    for r, g1, g2 in zip(rows, v1_feats, v2_feats):
        score_delta = (r["score"] or 0) - current_best
        tab = [
            design_idx.get(r["design"], -1),
            agent_idx.get(r["agent"], -1),
            r["iteration_n"],
            current_best,
            score_delta,
            r["train_min"],
        ]
        graph_v1 = [g1["triggered_by_directive"], g1["cites_checkpoint"],
                    g1["informed_by_breakthrough"], g1["informed_by_dead_end"]]
        graph_v2 = [
            g2["n_mistakes_at_time"],
            g2["n_desires_at_time"],
            g2["n_learnings_at_time"],
            g2["avoided_known_mistake"],
            g2["fulfills_prior_desire"],
            g2["built_on_learning"],
        ]
        X_tab.append(tab)
        X_v1.append(tab + graph_v1)
        X_v2.append(tab + graph_v1 + graph_v2)
        y.append(1 if r["status"] == "keep" else 0)

        if r["status"] == "keep" and r["score"] and r["score"] > current_best:
            current_best = r["score"]

    return X_tab, X_v1, X_v2, y


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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: dag_extractor_v2.py results.tsv blackboard.md "
              "[MISTAKES.md] [DESIRES.md] [LEARNINGS.md]")
        sys.exit(1)

    results_path = sys.argv[1]
    blackboard_path = sys.argv[2]
    domain_dir = Path(results_path).parent

    mistakes_path = sys.argv[3] if len(sys.argv) > 3 else domain_dir / "MISTAKES.md"
    desires_path  = sys.argv[4] if len(sys.argv) > 4 else domain_dir / "DESIRES.md"
    learnings_path= sys.argv[5] if len(sys.argv) > 5 else domain_dir / "LEARNINGS.md"

    rows = load_results(results_path)
    v1_feats = extract_v1_features(rows, blackboard_path)
    telemetry_idx = build_telemetry_index(mistakes_path, desires_path, learnings_path)
    v2_feats = extract_v2_features(rows, telemetry_idx)

    print(f"\n=== DAG v2 Feature Extraction ===")
    print(f"N={len(rows)} experiments")

    # Telemetry summary
    total_mistakes = sum(len(v["mistakes"]) for v in telemetry_idx.values())
    total_desires  = sum(len(v["desires"])  for v in telemetry_idx.values())
    total_learnings= sum(len(v["learnings"])for v in telemetry_idx.values())
    print(f"Telemetry: {total_mistakes} mistakes, {total_desires} desires, {total_learnings} learnings")
    print(f"Experiments with telemetry: {len(telemetry_idx)}")
    print()

    # Per-experiment table
    print(f"{'EXP-ID':<12} {'nmist':>5} {'avoid':>5} {'fulfil':>6} {'buildon':>7} "
          f"{'dir':>4} {'cite':>4} {'status':<8} design")
    print("-" * 75)
    for r, g1, g2 in zip(rows, v1_feats, v2_feats):
        print(f"{r['exp_id']:<12} "
              f"{g2['n_mistakes_at_time']:>5} "
              f"{g2['avoided_known_mistake']:>5} "
              f"{g2['fulfills_prior_desire']:>6} "
              f"{g2['built_on_learning']:>7}  "
              f"{g1['triggered_by_directive']:>4} "
              f"{g1['cites_checkpoint']:>4}  "
              f"{r['status']:<8} {r['design']}")

    # LOO-CV comparison
    X_tab, X_v1, X_v2, y = build_features(rows, v1_feats, v2_feats)
    majority = max(sum(y), len(y)-sum(y)) / len(y)

    acc_tab, auc_tab = loocv(X_tab, y)
    acc_v1,  auc_v1  = loocv(X_v1,  y)
    acc_v2,  auc_v2  = loocv(X_v2,  y)

    print(f"\n=== LOO-CV Comparison ===")
    print(f"Majority baseline:           acc={majority:.3f}  auc=0.500")
    if auc_tab is not None:
        print(f"Tabular only:                acc={acc_tab:.3f}  auc={auc_tab:.3f}")
        print(f"Tabular + v1 graph:          acc={acc_v1:.3f}  auc={auc_v1:.3f}  "
              f"delta={auc_v1-auc_tab:+.3f}")
        print(f"Tabular + v1 + v2 telemetry: acc={acc_v2:.3f}  auc={auc_v2:.3f}  "
              f"delta={auc_v2-auc_tab:+.3f}")

    # Telemetry feature stats
    print(f"\n=== v2 Telemetry Feature Stats ===")
    for feat, label in [
        ("avoided_known_mistake", "avoided_known_mistake"),
        ("fulfills_prior_desire", "fulfills_prior_desire"),
        ("built_on_learning",     "built_on_learning"),
    ]:
        vals = [g[feat] for g in v2_feats]
        n_pos = sum(vals)
        keep_rate = (sum(g[feat] and rows[i]["status"]=="keep"
                        for i, g in enumerate(v2_feats)) / max(n_pos, 1))
        print(f"  {label:<30} n={n_pos:>2}/{len(vals)}  keep_rate={keep_rate:.0%}")

    # Avoided mistake breakdown
    avoided = [(r["exp_id"], g["mistake_tactic"], r["status"])
               for r, g in zip(rows, v2_feats) if g["avoided_known_mistake"]]
    if avoided:
        print(f"\n=== Avoided Mistakes ===")
        for exp, tactic, status in avoided:
            print(f"  {exp}: avoided '{tactic}' → {status}")

    # Save full DAG v2 JSON
    dag = []
    for r, g1, g2 in zip(rows, v1_feats, v2_feats):
        dag.append({
            "exp_id": r["exp_id"],
            "status": r["status"],
            "design": r["design"],
            "score": r["score"],
            **g1, **g2
        })

    out = Path(results_path).parent / "dag_features_v2.json"
    with open(out, "w") as f:
        json.dump(dag, f, indent=2)
    print(f"\nDAG v2 features saved to {out}")


if __name__ == "__main__":
    main()

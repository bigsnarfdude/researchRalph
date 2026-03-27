#!/usr/bin/env python3
"""
meta_experiment.py — Does causal structure in agent traces predict experiment outcomes?

Hypothesis: (design_type, iteration_n, current_best_before, agent_id) predicts
            keep/discard and score better than chance.

Usage:
    python3 tools/meta_experiment.py domains/rrma-r1/results.tsv [domains/rrma-r1/blackboard.md]
    python3 tools/meta_experiment.py domains/rrma-red-team/results.tsv

Output: LOO-CV accuracy + AUC, feature importances, verdict on whether graph extraction is worth it.
"""

import sys
import csv
import re
from pathlib import Path


def load_results(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            try:
                rows.append({
                    "exp_id": row["EXP-ID"].strip(),
                    "score": float(row["score"]),
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


def extract_features(rows):
    """Build feature matrix from tabular data alone (no blackboard)."""
    # Encode design types
    designs = sorted(set(r["design"] for r in rows))
    design_idx = {d: i for i, d in enumerate(designs)}

    # Encode agent ids
    agents = sorted(set(r["agent"] for r in rows))
    agent_idx = {a: i for i, a in enumerate(agents)}

    X, y, labels = [], [], []
    current_best = 0.0

    for r in rows:
        score_delta = r["score"] - current_best
        feat = [
            design_idx.get(r["design"], -1),   # design type
            agent_idx.get(r["agent"], -1),       # agent id
            r["iteration_n"],                    # position in sequence
            current_best,                        # best known before this run
            score_delta,                         # improvement (or not)
            r["train_min"],                      # compute cost
        ]
        X.append(feat)
        y.append(1 if r["status"] == "keep" else 0)
        labels.append(r["exp_id"])

        if r["status"] == "keep" and r["score"] > current_best:
            current_best = r["score"]

    return X, y, labels, designs


def blackboard_features(rows, blackboard_path):
    """
    Extract blackboard context signal: how many 'keep' entries are cited in
    the description of each experiment (proxy for how informed it was).
    Returns list of (n_citations, n_words_context) per row, aligned with rows.
    """
    if not blackboard_path or not Path(blackboard_path).exists():
        return [(0, 0)] * len(rows)

    text = Path(blackboard_path).read_text()
    # Find section headers matching EXP-IDs
    exp_sections = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(EXP-\w+)", line)
        if m:
            current = m.group(1)
            exp_sections[current] = []
        elif current:
            exp_sections[current].append(line)

    features = []
    keep_so_far = set()
    for r in rows:
        exp_id = r["exp_id"]
        section = exp_sections.get(exp_id, [])
        section_text = " ".join(section)

        # Count how many prior keep experiments are referenced
        n_citations = sum(1 for k in keep_so_far if k in section_text)
        n_words = len(section_text.split())
        features.append((n_citations, n_words))

        if r["status"] == "keep":
            keep_so_far.add(exp_id)

    return features


def run_loocv(X, y, labels):
    """LOO-CV logistic regression. Pure stdlib if sklearn missing."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        import numpy as np

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y)

        n = len(y_arr)
        preds, probs = [], []

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_tr, y_tr = X_arr[mask], y_arr[mask]
            X_te = X_arr[[i]]

            if len(set(y_tr)) < 2:
                preds.append(int(y_arr[mask].mean() > 0.5))
                probs.append(0.5)
                continue

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X_tr, y_tr)
            preds.append(int(clf.predict(X_te)[0]))
            probs.append(clf.predict_proba(X_te)[0][1])

        acc = sum(p == t for p, t in zip(preds, y)) / n
        try:
            auc = roc_auc_score(y_arr, probs)
        except ValueError:
            auc = float("nan")

        # Wrong predictions
        wrong = [labels[i] for i in range(n) if preds[i] != y[i]]
        return acc, auc, wrong

    except ImportError:
        # Fallback: majority-class baseline
        majority = int(sum(y) / len(y) > 0.5)
        acc = sum(1 for yi in y if yi == majority) / len(y)
        return acc, float("nan"), []


def feature_importance(X, y):
    """Coefficient magnitudes from a single logistic regression on all data."""
    feat_names = ["design_type", "agent_id", "iteration_n", "current_best", "score_delta", "train_min"]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y)
        if len(set(y_arr)) < 2:
            return []

        sc = StandardScaler()
        X_s = sc.fit_transform(X_arr)
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_s, y_arr)
        return sorted(zip(feat_names, clf.coef_[0]), key=lambda x: -abs(x[1]))
    except ImportError:
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 meta_experiment.py results.tsv [blackboard.md]")
        sys.exit(1)

    results_path = sys.argv[1]
    blackboard_path = sys.argv[2] if len(sys.argv) > 2 else None

    rows = load_results(results_path)
    if not rows:
        print("No rows loaded. Check TSV format.")
        sys.exit(1)

    n = len(rows)
    n_keep = sum(1 for r in rows if r["status"] == "keep")
    n_discard = n - n_keep
    majority_acc = max(n_keep, n_discard) / n

    print(f"\n=== Meta-Experiment: Does structure predict outcomes? ===")
    print(f"Dataset: {results_path}")
    print(f"N={n} experiments  keep={n_keep}  discard={n_discard}")
    print(f"Majority-class baseline: {majority_acc:.3f}")

    X, y, labels, designs = extract_features(rows)
    print(f"Design types: {designs}")

    # Optionally augment with blackboard features
    bb_feats = blackboard_features(rows, blackboard_path)
    if any(f != (0, 0) for f in bb_feats):
        print(f"Blackboard context features added (citations, n_words)")
        X = [x + list(f) for x, f in zip(X, bb_feats)]

    acc, auc, wrong = run_loocv(X, y, labels)
    print(f"\nLOO-CV accuracy: {acc:.3f}  (baseline: {majority_acc:.3f})")
    print(f"LOO-CV AUC:      {auc:.3f}  (baseline: 0.500)")

    lift = acc - majority_acc
    if lift > 0.05:
        verdict = "SIGNAL PRESENT — graph extraction worth building"
    elif lift > 0:
        verdict = "WEAK SIGNAL — try more experiments or blackboard features"
    else:
        verdict = "NO SIGNAL from tabular features alone — need graph context"

    print(f"\nVerdict: {verdict}")
    if wrong:
        print(f"Wrong predictions: {wrong}")

    imps = feature_importance(X, y)
    if imps:
        print(f"\nFeature importances (|coef|):")
        for name, coef in imps:
            bar = "█" * int(abs(coef) * 10)
            print(f"  {name:<20} {coef:+.3f}  {bar}")

    # Per-design summary
    print(f"\nPer-design type breakdown:")
    from collections import defaultdict
    by_design = defaultdict(list)
    for r in rows:
        by_design[r["design"]].append(r["status"])
    for d, statuses in sorted(by_design.items()):
        k = statuses.count("keep")
        total = len(statuses)
        print(f"  {d:<25} keep={k}/{total}  ({k/total:.0%})")

    print()


if __name__ == "__main__":
    main()

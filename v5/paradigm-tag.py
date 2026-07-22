#!/usr/bin/env python3
"""paradigm-tag.py — deterministic method-family tagger for an RRMA results.tsv.

Step 2 of the v5.0 test ladder: build the divergence metric on the run we
already did. Tags every experiment with method families derived from the
design column (CamelCase-tokenized) plus the description (word-boundary
regex). Multi-label: one experiment can carry several families.

Outputs, in order:
  1. per-family count / first-seen / last-seen / experiment index ranges
  2. UNMATCHED experiments printed in full (loud, for manual review)
  3. the absent-family watch-list (paradigms with zero hits — the actual
     divergence detectors for the island run)
  4. longest no-improvement plateau (validates the metric against the known
     ~22-experiment LISTA plateau)
  5. --json FILE: machine-readable baseline for the island-run comparison

Usage: python3 v5/paradigm-tag.py domains/<d>/results.tsv [--json out.json]
No model calls. Deterministic.
"""

import json
import re
import sys
from pathlib import Path

# Family -> (design-token set, description regex). An experiment gets the tag
# if any design token matches OR the description regex hits.
PARADIGMS = {
    "baseline":        ({"baseline"}, r"\bdefault config\b"),
    "topk":            ({"topk", "batchtopk"}, r"\bbatch.?top.?k\b"),
    "unrolled-ista":   ({"ista", "lista", "fista", "evalista"}, r"\b[fl]?ista\b"),
    "term":            ({"term"}, r"\bterm\b|\btilted\b"),
    "matryoshka":      ({"matryoshka"}, r"matr?yoshka"),
    "reference-style": ({"referencestyle", "refstyle"}, r"\breference.?style\b|\bref.?style\b"),
    "freq-sort":       ({"freq", "freqsort"}, r"\bfreq(uency)?([ -]?sort)?\b"),
    "deck":            ({"deck"}, r"\bdec.?k\b"),
    "supervised-loss": ({"supervised"}, r"\bsupervised\b|\bgt cls\b|\bground.?truth\b"),
    "multi-width":     ({"widths", "multiwidth"}, r"\b\d+ ?widths?\b|\bwidth sweep\b"),
    "residual":        ({"residual"}, r"\bresidual\b"),
    "tied-encoder":    ({"tiedencoder", "tied"}, r"\btied.?encoder\b"),
}

# Paradigms the baseline is believed never to have tried. Zero hits here is
# the expected result — these become the island run's divergence detectors.
ABSENT_DETECTORS = {
    "inference-time": r"\binference.?time\b|\btest.?time\b|\brelabel\b|\bleave.?one.?out\b|\bloo\b",
    "gated":          r"\bgated\b",
    "jumprelu":       r"\bjump.?relu\b",
    "crosscoder":     r"\bcrosscoder\b",
    "transcoder":     r"\btranscoder\b",
}

CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def design_tokens(design: str) -> set:
    toks = set()
    for part in re.split(r"[-_./ ]", design):
        toks.update(t.lower() for t in CAMEL_RE.findall(part))
    # merge adjacent-token compounds the rules use (e.g. Eval+ISTA, Ref+Style)
    joined = "".join(CAMEL_RE.findall(design)).lower()
    for compound in ("evalista", "referencestyle", "refstyle", "tiedencoder", "batchtopk", "topk", "freqsort", "multiwidth"):
        if compound in joined:
            toks.add(compound)
    return toks


def tag(design: str, desc: str) -> list:
    toks = design_tokens(design)
    text = f"{design} {desc}".lower()
    tags = []
    for fam, (tokset, rx) in PARADIGMS.items():
        if toks & tokset or re.search(rx, text):
            tags.append(fam)
    return tags


def main() -> int:
    args = sys.argv[1:]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = Path(args[i + 1])
        del args[i : i + 2]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 1

    tsv = Path(args[0])
    rows = []
    with tsv.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            rows.append(parts)

    fam_hits = {f: [] for f in PARADIGMS}          # family -> [exp index]
    unmatched = []
    for idx, r in enumerate(rows, 1):
        exp_id, score, _, _, desc, agent, design = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        tags = tag(design, desc)
        for t in tags:
            fam_hits[t].append(idx)
        if not tags:
            unmatched.append((idx, exp_id, design, desc))

    def ranges(ix):
        out, start, prev = [], ix[0], ix[0]
        for i in ix[1:]:
            if i == prev + 1:
                prev = i
                continue
            out.append((start, prev))
            start = prev = i
        out.append((start, prev))
        return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)

    print(f"# paradigm tagging — {tsv}  ({len(rows)} experiments)\n")
    print(f"{'family':<16} {'count':>5} {'first':>5} {'last':>5}  experiment index ranges")
    present = {}
    for fam, ix in sorted(fam_hits.items(), key=lambda kv: -len(kv[1])):
        if not ix:
            continue
        present[fam] = {"count": len(ix), "first": ix[0], "last": ix[-1]}
        print(f"{fam:<16} {len(ix):>5} {ix[0]:>5} {ix[-1]:>5}  {ranges(ix)}")
    never = [f for f, ix in fam_hits.items() if not ix]
    if never:
        print(f"\nno hits (defined but unused rules): {', '.join(never)}")

    print(f"\n## UNMATCHED ({len(unmatched)}) — review by hand, do not silently bucket")
    for idx, exp_id, design, desc in unmatched:
        print(f"  [{idx:>3}] {exp_id:<14} design={design!r:<28} {desc[:70]}")

    print("\n## absent-family watch-list (divergence detectors for the island run)")
    absent = []
    for fam, rx in ABSENT_DETECTORS.items():
        hits = [i for i, r in enumerate(rows, 1) if re.search(rx, f"{r[6]} {r[4]}".lower())]
        status = "ABSENT (0 hits) — as expected" if not hits else f"PRESENT at {ranges(hits)} — NOT a divergence detector"
        if not hits:
            absent.append(fam)
        print(f"  {fam:<16} {status}")

    # Plateau: longest streak of experiments that never improved best-so-far
    best = None
    last_improve = 0
    longest = (0, 0, 0)  # (length, start_idx, end_idx)
    for idx, r in enumerate(rows, 1):
        try:
            s = float(r[1])
        except ValueError:
            continue
        if best is None or s > best:
            gap = idx - last_improve - 1
            if gap > longest[0]:
                longest = (gap, last_improve + 1, idx - 1)
            best = s
            last_improve = idx
    tail_gap = len(rows) - last_improve
    print(f"\n## plateau metric")
    print(f"  best score: {best}  (last improved at experiment {last_improve})")
    print(f"  longest mid-run plateau: {longest[0]} experiments (index {longest[1]}-{longest[2]})")
    print(f"  tail after final improvement: {tail_gap} experiments")

    if json_out:
        json_out.write_text(json.dumps({
            "source": str(tsv),
            "n_experiments": len(rows),
            "paradigms_present": present,
            "unmatched": [{"index": i, "exp_id": e, "design": d} for i, e, d, _ in unmatched],
            "divergence_detectors_absent": absent,
            "plateau": {"longest": longest[0], "start": longest[1], "end": longest[2],
                        "tail_after_last_improvement": tail_gap},
            "best_score": best,
        }, indent=2) + "\n")
        print(f"\nwrote baseline: {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

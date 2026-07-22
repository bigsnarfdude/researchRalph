#!/usr/bin/env python3
"""board-sim.py — log-TF cosine similarity between blackboard files.

The island re-correlation detector: run per generation on the K island boards;
a climbing pairwise similarity means migration is re-correlating the swarm.
Pure stdlib, deterministic.

Usage:
  python3 v5/board-sim.py fileA fileB            # one pair -> one number
  python3 v5/board-sim.py --matrix f1 f2 f3 ...  # pairwise matrix
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def vec(path: str) -> Counter:
    text = Path(path).read_text(errors="replace").lower()
    return Counter(WORD_RE.findall(text))


def cosine(a: Counter, b: Counter) -> float:
    wa = {t: 1 + math.log(c) for t, c in a.items()}
    wb = {t: 1 + math.log(c) for t, c in b.items()}
    dot = sum(w * wb[t] for t, w in wa.items() if t in wb)
    na = math.sqrt(sum(w * w for w in wa.values()))
    nb = math.sqrt(sum(w * w for w in wb.values()))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--matrix":
        files = args[1:]
        vs = [vec(f) for f in files]
        names = [Path(f).parent.name or Path(f).name for f in files]
        w = max(len(n) for n in names)
        print(" " * (w + 2) + "  ".join(f"{n[:12]:>12}" for n in names))
        for i, n in enumerate(names):
            row = "  ".join(f"{cosine(vs[i], vs[j]):>12.3f}" for j in range(len(vs)))
            print(f"{n:>{w}}  {row}")
        return 0
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    print(f"{cosine(vec(args[0]), vec(args[1])):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
